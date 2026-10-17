// Audio helpers between the models: the magnitude spectrogram the tone color converter takes (OpenVoice's
// spectrogram_torch with center=False) and the sample rate conversion from the base voice (44.1 kHz) to the converter (22.05 kHz).

use realfft::RealFftPlanner;

pub const CONVERTER_RATE: u32 = 22050;
const N_FFT: usize = 1024;
const HOP: usize = 256;
pub const BINS: usize = N_FFT / 2 + 1;

// [BINS, frames] in row-major order (all frames of bin 0, then bin 1, ...): the layout of the converter's "spec" input.
pub struct Spectrogram {
    pub data: Vec<f32>,
    pub frames: usize,
}

// torch: pad (n_fft - hop) / 2 on both sides with the signal mirrored (no repeated edge sample), a periodic Hann window,
// no centering, and sqrt(re^2 + im^2 + 1e-6) as the magnitude.
pub fn spectrogram(audio: &[f32]) -> Spectrogram {
    let pad = (N_FFT - HOP) / 2;
    let mut padded = Vec::with_capacity(audio.len() + 2 * pad);
    let len = audio.len();
    let at = |i: isize| -> f32 {
        // reflect: -1 -> 1, -2 -> 2 ... and len -> len - 2 ...; a signal shorter than the pad is a case the callers avoid
        let mut i = i;
        let last = len as isize - 1;
        if last <= 0 {
            return audio.first().copied().unwrap_or(0.0);
        }
        loop {
            if i < 0 {
                i = -i;
            } else if i > last {
                i = 2 * last - i;
            } else {
                return audio[i as usize];
            }
        }
    };
    for i in 0..(len + 2 * pad) {
        padded.push(at(i as isize - pad as isize));
    }
    let frames = if padded.len() >= N_FFT { (padded.len() - N_FFT) / HOP + 1 } else { 0 };
    let window: Vec<f32> = (0..N_FFT).map(|n| 0.5 - 0.5 * (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos()).collect();
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
    let mut input = fft.make_input_vec();
    let mut output = fft.make_output_vec();
    let mut data = vec![0.0f32; BINS * frames];
    for t in 0..frames {
        for n in 0..N_FFT {
            input[n] = padded[t * HOP + n] * window[n];
        }
        fft.process(&mut input, &mut output).expect("buffers are sized by the plan");
        for (k, value) in output.iter().enumerate() {
            data[k * frames + t] = (value.re * value.re + value.im * value.im + 1e-6).sqrt();
        }
    }
    Spectrogram { data, frames }
}

// Zeroth order modified Bessel function of the first kind, by its power series (enough terms for the beta used here).
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..200 {
        term *= (half / k as f64) * (half / k as f64);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

fn sinc(x: f64) -> f64 {
    if x == 0.0 {
        1.0
    } else {
        (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x)
    }
}

// resampy's "kaiser_best" filter, the one librosa.load uses when it changes the sample rate: the right half of a sinc
// (50 zero crossings, 8192 table entries each, cut-off at 0.9173 of Nyquist) under a Kaiser window (beta 12.98). The numbers
// are what resampy's data file holds (checked against it to 4e-16).
// The converter and the reference voice encoder react to the faintest content of the spectrum (they see sqrt(power + 1e-6) of
// every bin), so a resampler with a slightly different transition band gives a quite different voice; this one is a
// step by step port of resampy so that it does not.
const NUM_ZEROS: usize = 50;
const NUM_TABLE: usize = 8192;
const ROLLOFF: f64 = 0.9173473712608761;
const BETA: f64 = 12.984585247040012;

struct Filter {
    table: Vec<f64>,
    delta: Vec<f64>,
}

fn kaiser_best() -> &'static Filter {
    static FILTER: std::sync::OnceLock<Filter> = std::sync::OnceLock::new();
    FILTER.get_or_init(|| {
        let n = NUM_TABLE * NUM_ZEROS;
        let denominator = bessel_i0(BETA);
        let table: Vec<f64> = (0..=n)
            .map(|i| {
                let sinc_win = ROLLOFF * sinc(ROLLOFF * (i as f64 * NUM_ZEROS as f64 / n as f64));
                // the second half of the symmetric Kaiser window of length 2n + 1: positions n..=2n
                let r = i as f64 / n as f64;
                let taper = bessel_i0(BETA * (1.0 - r * r).max(0.0).sqrt()) / denominator;
                taper * sinc_win
            })
            .collect();
        let mut delta = vec![0.0; table.len()];
        for i in 0..table.len() - 1 {
            delta[i] = table[i + 1] - table[i];
        }
        Filter { table, delta }
    })
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// The filter weights for one phase: where the output sample falls between two input samples, as a fraction p / q (q = the
// rate ratio's denominator). The weights depend only on that, not on which output sample it is, so they are worked out once
// per phase: `left[i]` multiplies the input i samples before the sample just before the output position, `right[k]` the
// k-th one after it. The values are what resampy computes in its loop (including the linear interpolation in the table).
struct Phase {
    left: Vec<f32>,
    right: Vec<f32>,
}

fn phase_weights(frac_num: u64, denominator: u64, scale: f64, gain: f64) -> Phase {
    let filter = kaiser_best();
    let nwin = filter.table.len();
    let index_step = (scale * NUM_TABLE as f64) as usize;
    let frac = scale * (frac_num as f64 / denominator as f64);
    let weights = |frac: f64| -> Vec<f32> {
        let index_frac = frac * NUM_TABLE as f64;
        let offset = index_frac as usize;
        let eta = index_frac - offset as f64;
        (0..(nwin - offset) / index_step)
            .map(|i| {
                let at = offset + i * index_step;
                ((filter.table[at] + eta * filter.delta[at]) * gain) as f32
            })
            .collect()
    };
    Phase { left: weights(frac), right: weights(scale - frac) }
}

// resampy.resample + librosa's fix_length: the same samples as librosa.load(..., sr=to) gives from audio at `from`.
// An empty result for a rate of 0 and for a conversion that would produce an absurd amount of audio (the callers refuse such
// rates before this; this keeps a missed one from dividing by zero or asking for gigabytes).
const MAX_OUTPUT_SAMPLES: u64 = 20_000_000;

pub fn resample(audio: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == 0 || to == 0 {
        return Vec::new();
    }
    if from == to {
        return audio.to_vec();
    }
    if audio.len() as u64 * to as u64 / from as u64 > MAX_OUTPUT_SAMPLES {
        return Vec::new();
    }
    let sample_ratio = to as f64 / from as f64;
    let scale = sample_ratio.min(1.0);
    if (scale * NUM_TABLE as f64) < 1.0 {
        return Vec::new(); // a ratio below 1/8192: the filter table has no entry per input sample
    }
    // resampy multiplies the whole table by the ratio when going down in rate; the sum is linear in it
    let gain = if sample_ratio < 1.0 { sample_ratio } else { 1.0 };
    let n_orig = audio.len();
    let n_out = (n_orig as u64 * to as u64 / from as u64) as usize; // resampy truncates ...
    // Output sample t sits at input position t * from / to: its whole part and its fraction, in exact integers
    let divisor = gcd(from as u64, to as u64);
    let denominator = to as u64 / divisor;
    let mut phases: std::collections::HashMap<u64, Phase> = std::collections::HashMap::new();
    let mut out = vec![0.0f32; n_out];
    for (t, slot) in out.iter_mut().enumerate() {
        let position = t as u64 * from as u64;
        let n = (position / to as u64) as usize;
        let phase_index = (position % to as u64) / divisor;
        let phase = phases.entry(phase_index).or_insert_with(|| phase_weights(phase_index, denominator, scale, gain));
        let mut sum = 0.0f32;
        let left_count = (n + 1).min(phase.left.len());
        // left wing: the input going backwards from n
        for (weight, sample) in phase.left[..left_count].iter().zip(audio[n + 1 - left_count..=n].iter().rev()) {
            sum += weight * sample;
        }
        // right wing: the input going forwards from n + 1
        let right_count = (n_orig - n - 1).min(phase.right.len());
        for (weight, sample) in phase.right[..right_count].iter().zip(&audio[n + 1..n + 1 + right_count]) {
            sum += weight * sample;
        }
        *slot = sum;
    }
    // ... and librosa then pads (or cuts) to the rounded-up length
    out.resize((n_orig as f64 * to as f64 / from as f64).ceil() as usize, 0.0);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn golden_f32(name: &str) -> Vec<f32> {
        let path = format!("{}/tests/golden/{name}", env!("CARGO_MANIFEST_DIR"));
        std::fs::read(path).expect("golden file").chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    fn base_44k() -> Vec<f32> {
        let path = format!("{}/tests/golden/dsp_base_44k.wav", env!("CARGO_MANIFEST_DIR"));
        let mut reader = hound::WavReader::open(path).expect("golden wav");
        assert_eq!(reader.spec().sample_rate, 44100);
        reader.samples::<f32>().map(|s| s.expect("a sample")).collect()
    }

    fn correlation(a: &[f32], b: &[f32]) -> f64 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let ma = a.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let mb = b.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let (x, y) = (a[i] as f64 - ma, b[i] as f64 - mb);
            sab += x * y;
            saa += x * x;
            sbb += y * y;
        }
        sab / (saa * sbb).sqrt()
    }

    #[test]
    fn spectrogram_matches_torch_on_the_same_samples() {
        // The input is librosa's resampled audio, so only the STFT is under test here.
        let audio = golden_f32("dsp_audio_22k.f32");
        let want = golden_f32("dsp_spec.f32");
        let got = spectrogram(&audio);
        assert_eq!(got.data.len(), want.len(), "same number of bins and frames");
        let worst = got.data.iter().zip(&want).map(|(a, b)| (a - b).abs() / (1.0 + b.abs())).fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "worst relative difference {worst}");
    }

    #[test]
    fn resampling_follows_librosa() {
        let want = golden_f32("dsp_audio_22k.f32");
        let got = resample(&base_44k(), 44100, 22050);
        assert_eq!(got.len(), want.len());
        let c = correlation(&got, &want);
        assert!(c > 0.99999, "correlation with librosa {c}");
    }

    #[test]
    fn resampling_a_24k_reference_follows_librosa() {
        let path = format!("{}/tests/golden/ref_en.wav", env!("CARGO_MANIFEST_DIR"));
        let mut reader = hound::WavReader::open(path).expect("reference wav");
        assert_eq!(reader.spec().sample_rate, 24000);
        let audio: Vec<f32> = reader.samples::<f32>().map(|s| s.expect("a sample")).collect();
        let want = golden_f32("ref_22k.f32");
        let got = resample(&audio, 24000, 22050);
        assert_eq!(got.len(), want.len());
        let c = correlation(&got, &want);
        assert!(c > 0.99999, "correlation with librosa {c}");
    }

    #[test]
    fn absurd_rates_give_nothing_instead_of_panicking() {
        let audio = vec![0.1f32; 4096];
        assert!(resample(&audio, 0, 22050).is_empty());
        assert!(resample(&audio, 22050, 0).is_empty());
        assert!(resample(&audio, 200_000_000, 22050).len() < 4096);
        assert!(resample(&audio, 1, 22050).is_empty(), "an input at 1 Hz would need an enormous output");
    }

    #[test]
    fn resampling_keeps_a_tone_and_its_level() {
        let tone: Vec<f32> = (0..44100).map(|n| (2.0 * std::f32::consts::PI * 1000.0 * n as f32 / 44100.0).sin() * 0.5).collect();
        let out = resample(&tone, 44100, 22050);
        let peak = out[1000..out.len() - 1000].iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!((peak - 0.5).abs() < 0.01, "peak {peak}");
    }
}
