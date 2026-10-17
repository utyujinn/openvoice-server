// The engine: English text in, speech in the reference voice out, all with ONNX Runtime (through `ort`, loading the
// DLL at run time) and the Rust text and audio code of this crate.
//
//   text -> pieces -> [text front end] -> phoneme ids + BERT features -> base voice model (44.1 kHz)
//        -> resample to 22.05 kHz -> spectrogram -> tone color converter (+ the reference voice's embedding) -> audio
//
// What is random in it (the duration predictor, the noise of the base voice model, the converter's `tau`) is a parameter, so a
// test can switch it all off and compare with the Python original sample by sample.

use crate::dsp::{self, CONVERTER_RATE};
use crate::text::frontend::{Encoded, Frontend};
use anyhow::{anyhow, Context, Result};
use ort::environment::Environment;
use ort::ep;
use ort::session::{Session, SessionOutputs};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub struct EngineConfig {
    // Folder of the ONNX models: melo.onnx, converter.onnx, bert10.onnx, ref_enc.onnx
    pub models_dir: PathBuf,
    // Folder of the text data: dictionaries, tokenizer, speaker embeddings (se_*.bin)
    pub data_dir: PathBuf,
    // onnxruntime.dll (the build matching the provider DLLs next to it)
    pub ort_dll: PathBuf,
    pub use_cuda: bool,
}

// Settings of one synthesis. The defaults are MeloTTS' and OpenVoice's own.
#[derive(Clone, Debug)]
pub struct Params {
    pub speaker: String,
    pub sdp_ratio: f32,
    pub noise_scale: f32,
    pub noise_scale_w: f32,
    pub speed: f32,
    pub tau: f32,
}

impl Params {
    // What the callers may ask for. Outside these (or not finite) the model gets nonsense: a speed of 0 makes every duration
    // infinite, which turns into an enormous allocation.
    pub fn validate(&self) -> Result<()> {
        let within = |v: f32, low: f32, high: f32| v.is_finite() && v >= low && v <= high;
        anyhow::ensure!(within(self.speed, 0.25, 4.0), "speed must be between 0.25 and 4 (it is {})", self.speed);
        anyhow::ensure!(
            within(self.tau, 0.0, 2.0) && within(self.noise_scale, 0.0, 2.0) && within(self.noise_scale_w, 0.0, 2.0) && within(self.sdp_ratio, 0.0, 1.0),
            "tau, noise_scale and noise_scale_w must be between 0 and 2, sdp_ratio between 0 and 1"
        );
        Ok(())
    }
}

impl Default for Params {
    fn default() -> Self {
        Params { speaker: "EN-US".to_string(), sdp_ratio: 0.2, noise_scale: 0.6, noise_scale_w: 0.8, speed: 1.0, tau: 0.3 }
    }
}

// Where the time of the last piece went, in milliseconds.
#[derive(Clone, Debug, Default)]
pub struct Timings {
    pub front: f64,
    pub bert: f64,
    pub melo: f64,
    pub resample: f64,
    pub spectrogram: f64,
    pub converter: f64,
}

pub struct Engine {
    pub last_timings: Timings,
    frontend: Frontend,
    bert: Session,
    melo: Session,
    converter: Session,
    ref_enc: Session,
    // base voice embeddings by MeloTTS speaker name ("EN-US" ...)
    base_voices: HashMap<String, Vec<f32>>,
}

// ONNX Runtime allows one environment per process, ever: once built (or failed to build) it is what every later `load` gets, so the
// DLL path of the first call is the one that counts, and a failed first call is reported again by the following ones.
fn environment(dll: &Path) -> Result<&'static Environment> {
    static ENVIRONMENT: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENVIRONMENT
        .get_or_init(|| {
            ort::init_from(dll)
                .map_err(|e| format!("{}: {e}", dll.display()))
                .and_then(|builder| builder.with_name("openvoice").build().map_err(|e| e.to_string()))
        })
        .as_ref()
        .map_err(|e| anyhow!("{e}"))
}

// The data of a named output, or an error naming it: a model that was swapped for another (or a damaged download) must not
// panic the process through the `outputs["name"]` indexing, which does.
fn output_f32<'a>(outputs: &'a SessionOutputs<'_>, name: &str) -> Result<&'a [f32]> {
    let value = outputs.get(name).ok_or_else(|| anyhow!("the model gave no output named {name}"))?;
    let (_, data) = value.try_extract_tensor::<f32>().map_err(|e| anyhow!("{e}"))?;
    Ok(data)
}

fn session(environment: &Environment, path: &Path, use_cuda: bool) -> Result<Session> {
    let mut builder = Session::builder(environment).map_err(|e| anyhow!("{e}"))?;
    // The engine runs next to a game: one busy-waiting thread per core in each of the four sessions would take CPU time from it.
    builder = builder.with_intra_threads(if use_cuda { 1 } else { 2 }).map_err(|e| anyhow!("{e}"))?;
    if use_cuda {
        // The convolution algorithm is picked by heuristics: the default benchmark-per-shape search made the first
        // call of every new input length slow, and every sentence has a new length.
        // SameAsRequested: the default arena rounds every growth up to a power of two and never shrinks, so one long piece of text
        // would leave the process holding far more VRAM than it needs while VRChat runs.
        let cuda = ep::CUDA::default()
            .with_conv_algorithm_search(ep::cuda::ConvAlgorithmSearch::Heuristic)
            .with_arena_extend_strategy(ep::ArenaExtendStrategy::SameAsRequested)
            .build()
            .error_on_failure();
        builder = builder.with_execution_providers([cuda]).map_err(|e| anyhow!("{e}"))?;
    }
    builder.commit_from_file(path).map_err(|e| anyhow!("{}: {e}", path.display()))
}

fn tensor_f32(shape: Vec<usize>, data: Vec<f32>) -> Result<Tensor<f32>> {
    Tensor::from_array((shape, data)).map_err(|e| anyhow!("{e}"))
}

fn tensor_i64(shape: Vec<usize>, data: Vec<i64>) -> Result<Tensor<i64>> {
    Tensor::from_array((shape, data)).map_err(|e| anyhow!("{e}"))
}

fn scalar(value: f32) -> Result<Tensor<f32>> {
    tensor_f32(vec![], vec![value])
}

impl Engine {
    pub fn load(config: &EngineConfig) -> Result<Engine> {
        let environment = environment(&config.ort_dll)?;
        let frontend = Frontend::load(&config.data_dir).context("loading the text data")?;
        let mut base_voices = HashMap::new();
        for name in frontend.speakers.keys() {
            let file = config.data_dir.join(format!("se_{}.bin", name.to_lowercase().replace('_', "-")));
            if let Ok(bytes) = std::fs::read(&file) {
                base_voices.insert(name.clone(), bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
            }
        }
        Ok(Engine {
            last_timings: Timings::default(),
            bert: session(environment, &config.models_dir.join("bert10.onnx"), config.use_cuda)?,
            melo: session(environment, &config.models_dir.join("melo.onnx"), config.use_cuda)?,
            converter: session(environment, &config.models_dir.join("converter.onnx"), config.use_cuda)?,
            ref_enc: session(environment, &config.models_dir.join("ref_enc.onnx"), false)?,
            frontend,
            base_voices,
        })
    }

    // One throw-away synthesis (two pieces, so two input lengths): the first call of a graph on the GPU loads kernels and
    // takes a second or more, which is not time a user should wait. Done while loading, before the app is told it is ready.
    pub fn warm_up(&mut self) -> Result<()> {
        let neutral = vec![0.0f32; crate::voice::EMBEDDING_SIZE];
        for text in ["Hello.", "Hello, this is a short test of the voice."] {
            self.synthesize_piece(text, &neutral, &Params::default())?;
        }
        Ok(())
    }

    pub fn speakers(&self) -> Vec<String> {
        let mut names: Vec<String> = self.base_voices.keys().cloned().collect();
        names.sort();
        names
    }

    // The embedding (256 numbers) of a reference voice: what the converter needs to make the base voice sound like it.
    pub fn embed_voice(&mut self, audio: &[f32], rate: u32) -> Result<Vec<f32>> {
        anyhow::ensure!((crate::voice::MIN_RATE..=crate::voice::MAX_RATE).contains(&rate), "a sample rate of {rate} Hz is not accepted");
        let audio = dsp::resample(audio, rate, CONVERTER_RATE);
        anyhow::ensure!(audio.len() > 2048, "the reference voice is too short");
        let spec = dsp::spectrogram(&audio);
        // The model wants [1, frames, bins]: the spectrogram turned around
        let mut transposed = vec![0.0f32; spec.data.len()];
        for k in 0..dsp::BINS {
            for t in 0..spec.frames {
                transposed[t * dsp::BINS + k] = spec.data[k * spec.frames + t];
            }
        }
        let outputs = self.ref_enc.run(ort::inputs!["spec_t" => tensor_f32(vec![1, spec.frames, dsp::BINS], transposed)?]).map_err(|e| anyhow!("{e}"))?;
        let data = output_f32(&outputs, "g")?;
        anyhow::ensure!(data.len() == crate::voice::EMBEDDING_SIZE && data.iter().all(|v| v.is_finite()), "the reference voice gave no usable embedding");
        Ok(data.to_vec())
    }

    pub fn pieces(&self, text: &str) -> Vec<String> {
        self.frontend.pieces(text)
    }

    // BERT features per phoneme, [768, phonemes] flattened: the hidden state of every word piece repeated over its phonemes.
    fn bert_features(&mut self, encoded: &Encoded) -> Result<Vec<f32>> {
        let length = encoded.bert_ids.len();
        let ids: Vec<i64> = encoded.bert_ids.iter().map(|&i| i as i64).collect();
        let outputs = self
            .bert
            .run(ort::inputs!["input_ids" => tensor_i64(vec![1, length], ids)?, "attention_mask" => tensor_i64(vec![1, length], vec![1; length])?])
            .map_err(|e| anyhow!("{e}"))?;
        let hidden = output_f32(&outputs, "hidden")?;
        anyhow::ensure!(hidden.len() == length * 768, "BERT gave {} numbers for {length} tokens", hidden.len());
        let phonemes: usize = encoded.word2ph.iter().sum();
        anyhow::ensure!(phonemes == encoded.x.len(), "{phonemes} BERT positions for {} symbols", encoded.x.len());
        let mut features = vec![0.0f32; 768 * phonemes];
        let mut at = 0;
        for (token, &count) in encoded.word2ph.iter().enumerate() {
            let row = &hidden[token * 768..(token + 1) * 768];
            for _ in 0..count {
                for (d, &v) in row.iter().enumerate() {
                    features[d * phonemes + at] = v;
                }
                at += 1;
            }
        }
        Ok(features)
    }

    // One piece of text to speech at 22.05 kHz in the voice of `voice` (an embedding from embed_voice).
    pub fn synthesize_piece(&mut self, piece: &str, voice: &[f32], params: &Params) -> Result<Vec<f32>> {
        params.validate()?;
        anyhow::ensure!(voice.len() == crate::voice::EMBEDDING_SIZE, "a voice has {} numbers, this one {}", crate::voice::EMBEDDING_SIZE, voice.len());
        let started = std::time::Instant::now();
        let ms = |since: std::time::Instant| since.elapsed().as_secs_f64() * 1000.0;
        let encoded = self.frontend.encode(piece)?;
        self.last_timings.front = ms(started);
        let mark = std::time::Instant::now();
        let features = self.bert_features(&encoded)?;
        self.last_timings.bert = ms(mark);
        let mark = std::time::Instant::now();
        let length = encoded.x.len();
        let speaker = *self.frontend.speakers.get(&params.speaker).ok_or_else(|| anyhow!("unknown speaker {}", params.speaker))?;
        let base_se = self.base_voices.get(&params.speaker).ok_or_else(|| anyhow!("no embedding for speaker {}", params.speaker))?.clone();

        let outputs = self
            .melo
            .run(ort::inputs![
                "x" => tensor_i64(vec![1, length], encoded.x.clone())?,
                "x_lengths" => tensor_i64(vec![1], vec![length as i64])?,
                "sid" => tensor_i64(vec![1], vec![speaker])?,
                "tone" => tensor_i64(vec![1, length], encoded.tone.clone())?,
                "language" => tensor_i64(vec![1, length], encoded.lang.clone())?,
                "bert" => tensor_f32(vec![1, 1024, length], vec![0.0; 1024 * length])?,
                "ja_bert" => tensor_f32(vec![1, 768, length], features)?,
                "noise_scale" => scalar(params.noise_scale)?,
                "length_scale" => scalar(1.0 / params.speed)?,
                "noise_scale_w" => scalar(params.noise_scale_w)?,
                "sdp_ratio" => scalar(params.sdp_ratio)?,
            ])
            .map_err(|e| anyhow!("{e}"))?;
        let base = output_f32(&outputs, "audio")?;
        self.last_timings.melo = ms(mark);
        let mark = std::time::Instant::now();
        let base22 = dsp::resample(base, self.frontend.sampling_rate, CONVERTER_RATE);
        drop(outputs);
        self.last_timings.resample = ms(mark);
        let mark = std::time::Instant::now();

        let spec = dsp::spectrogram(&base22);
        anyhow::ensure!(spec.frames > 0, "the base voice came out too short");
        self.last_timings.spectrogram = ms(mark);
        let mark = std::time::Instant::now();
        let converted = self
            .converter
            .run(ort::inputs![
                "spec" => tensor_f32(vec![1, dsp::BINS, spec.frames], spec.data)?,
                "spec_lengths" => tensor_i64(vec![1], vec![spec.frames as i64])?,
                "g_src" => tensor_f32(vec![1, 256, 1], base_se)?,
                "g_tgt" => tensor_f32(vec![1, 256, 1], voice.to_vec())?,
                "tau" => scalar(params.tau)?,
            ])
            .map_err(|e| anyhow!("{e}"))?;
        let audio = output_f32(&converted, "audio")?;
        self.last_timings.converter = ms(mark);
        Ok(audio.to_vec())
    }

    // The whole text, piece by piece, with the 50 ms of silence between pieces the original puts between them.
    pub fn synthesize(&mut self, text: &str, voice: &[f32], params: &Params) -> Result<Vec<f32>> {
        params.validate()?;
        let mut out = Vec::new();
        for piece in self.pieces(text) {
            out.extend(self.synthesize_piece(&piece, voice, params)?);
            out.extend(std::iter::repeat(0.0f32).take((CONVERTER_RATE as f32 * 0.05 / params.speed) as usize));
        }
        Ok(out)
    }
}
