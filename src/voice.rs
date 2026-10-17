// Reference voices and WAV in/out.
//
// A registered voice is a small file holding the 256 numbers of its embedding (engine.rs embed_voice): "OVV1", the count as u32,
// then the numbers as f32, all little endian. The audio it was made from is not kept.

use anyhow::{bail, Context, Result};
use std::io::Cursor;
use std::path::Path;

const MAGIC: &[u8; 4] = b"OVV1";
pub const EMBEDDING_SIZE: usize = 256;

pub fn save(path: &Path, embedding: &[f32]) -> Result<()> {
    if embedding.len() != EMBEDDING_SIZE {
        bail!("an embedding has {EMBEDDING_SIZE} numbers, this one {}", embedding.len());
    }
    let mut bytes = Vec::with_capacity(8 + embedding.len() * 4);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(embedding.len() as u32).to_le_bytes());
    for v in embedding {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, bytes).with_context(|| format!("writing {}", path.display()))
}

pub fn load(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading the voice {}", path.display()))?;
    if bytes.len() < 8 || &bytes[..4] != MAGIC {
        bail!("{} is not a voice file", path.display());
    }
    let count = u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize;
    if count != EMBEDDING_SIZE || bytes.len() != 8 + count * 4 {
        bail!("{} is damaged", path.display());
    }
    Ok(bytes[8..].chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect())
}

// What a reference voice may be. hound accepts any sample rate that agrees with the byte rate field, so a damaged or hostile file
// could ask for a resampling by a factor of a million; the limits keep it to what a real recording is.
pub const MIN_RATE: u32 = 8_000;
pub const MAX_RATE: u32 = 192_000;
pub const MAX_SECONDS: u32 = 120;

// A WAV file's samples as mono floats in -1..1, and its sample rate. 8/16/24/32 bit integer and 32 bit float are read.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes)).context("that is not a WAV file")?;
    let spec = reader.spec();
    if !(MIN_RATE..=MAX_RATE).contains(&spec.sample_rate) {
        bail!("a sample rate of {} Hz is not accepted ({MIN_RATE}-{MAX_RATE} Hz)", spec.sample_rate);
    }
    if !(1..=32).contains(&spec.bits_per_sample) || spec.channels == 0 {
        bail!("{} bit, {} channel audio is not accepted", spec.bits_per_sample, spec.channels);
    }
    let frames = reader.duration() as u64;
    if frames > spec.sample_rate as u64 * MAX_SECONDS as u64 {
        bail!("the recording is longer than {MAX_SECONDS} seconds");
    }
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.map(|v| v as f32 / scale)).collect::<Result<_, _>>()?
        }
    };
    let channels = spec.channels as usize;
    let mono: Vec<f32> = samples.chunks(channels).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect();
    if mono.iter().any(|v| !v.is_finite()) {
        bail!("the recording holds samples that are not numbers");
    }
    Ok((mono, spec.sample_rate))
}

// 16-bit mono WAV: what the app's player takes. Samples beyond full scale are clipped, not wrapped.
pub fn encode_wav(audio: &[f32], rate: u32) -> Vec<u8> {
    let mut cursor = Cursor::new(Vec::with_capacity(44 + audio.len() * 2));
    {
        let spec = hound::WavSpec { channels: 1, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = hound::WavWriter::new(&mut cursor, spec).expect("writing to memory");
        for &v in audio {
            writer.write_sample((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).expect("writing to memory");
        }
        writer.finalize().expect("writing to memory");
    }
    cursor.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_voice_file_round_trips_and_a_damaged_one_is_refused() {
        let dir = std::env::temp_dir().join(format!("openvoice-voice-test-{}", std::process::id()));
        let path = dir.join("a.ovvoice");
        let embedding: Vec<f32> = (0..EMBEDDING_SIZE).map(|i| i as f32 * 0.5 - 20.0).collect();
        save(&path, &embedding).unwrap();
        assert_eq!(load(&path).unwrap(), embedding);
        let mut damaged = std::fs::read(&path).unwrap();
        damaged.pop();
        std::fs::write(&path, damaged).unwrap();
        assert!(load(&path).is_err());
        std::fs::write(&path, b"nope").unwrap();
        assert!(load(&path).is_err());
        assert!(save(&path, &[0.0; 3]).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn wav_round_trips_and_clips() {
        let audio = vec![0.0, 0.5, -0.5, 2.0, -2.0];
        let wav = encode_wav(&audio, 22050);
        let (back, rate) = decode_wav(&wav).unwrap();
        assert_eq!(rate, 22050);
        assert_eq!(back.len(), 5);
        assert!((back[1] - 0.5).abs() < 1e-3 && (back[2] + 0.5).abs() < 1e-3);
        assert!(back[3] > 0.999 && back[4] < -0.999, "out of range samples are clipped, not wrapped");
    }

    fn wav_with_rate(rate: u32, samples: usize) -> Vec<u8> {
        let mut wav = encode_wav(&vec![0.1; samples], 22050);
        wav[24..28].copy_from_slice(&rate.to_le_bytes());
        wav[28..32].copy_from_slice(&(rate.wrapping_mul(2)).to_le_bytes()); // 16-bit mono: byte rate = rate * 2
        wav
    }

    #[test]
    fn a_sample_rate_that_is_not_a_real_recording_is_refused() {
        for rate in [0u32, 1, 100, 7_999, 192_001, 200_000_000, 1_000_000_000] {
            assert!(decode_wav(&wav_with_rate(rate, 4096)).is_err(), "rate {rate} was accepted");
        }
        for rate in [8_000u32, 22_050, 44_100, 48_000, 192_000] {
            assert!(decode_wav(&wav_with_rate(rate, 4096)).is_ok(), "rate {rate} was refused");
        }
    }

    #[test]
    fn a_recording_longer_than_the_limit_is_refused() {
        let wav = wav_with_rate(8_000, 8_000 * (MAX_SECONDS as usize + 1));
        assert!(decode_wav(&wav).is_err());
    }

    #[test]
    fn samples_that_are_not_numbers_are_refused() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec { channels: 1, sample_rate: 22050, bits_per_sample: 32, sample_format: hound::SampleFormat::Float };
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for v in [0.1f32, f32::NAN, 0.2] {
                writer.write_sample(v).unwrap();
            }
            writer.finalize().unwrap();
        }
        assert!(decode_wav(&cursor.into_inner()).is_err());
    }

    #[test]
    fn stereo_is_averaged_to_mono() {
        let mut cursor = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec { channels: 2, sample_rate: 8000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for (l, r) in [(1000i16, 3000i16), (-2000, 0)] {
                writer.write_sample(l).unwrap();
                writer.write_sample(r).unwrap();
            }
            writer.finalize().unwrap();
        }
        let (mono, _) = decode_wav(&cursor.into_inner()).unwrap();
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 2000.0 / 32768.0).abs() < 1e-6 && (mono[1] + 1000.0 / 32768.0).abs() < 1e-6);
    }
}
