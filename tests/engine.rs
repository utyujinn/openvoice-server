// The whole engine against the Python original (PyTorch OpenVoice + MeloTTS with every random element off).
//
// Needs the ONNX models and an onnxruntime build, which are not in the repository, so it only runs when asked:
//   OPENVOICE_ORT_DLL=<path to onnxruntime.dll>  (with the CUDA libraries on PATH when OPENVOICE_CUDA=1)
//   cargo test --release --test engine -- --ignored --nocapture
// The models are read from ./models (see tests/golden/README.md).

use openvoice_server::engine::{Engine, EngineConfig, Params};
use std::path::PathBuf;
use std::time::Instant;

fn golden(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name)
}

fn read_f32(path: PathBuf) -> Vec<f32> {
    std::fs::read(path).expect("golden file").chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

fn correlation(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let (mut sab, mut saa, mut sbb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        sab += x * y;
        saa += x * x;
        sbb += y * y;
    }
    sab / (saa * sbb).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    correlation(a, b)
}

fn load_engine() -> Engine {
    let dll = std::env::var("OPENVOICE_ORT_DLL").expect("set OPENVOICE_ORT_DLL");
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    Engine::load(&EngineConfig { models_dir: root.join("models"), data_dir: root.join("data"), ort_dll: dll.into(), use_cuda: std::env::var("OPENVOICE_CUDA").is_ok_and(|v| v == "1") })
        .expect("the engine loads")
}

fn reference_voice() -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(golden("ref_en.wav")).expect("reference wav");
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader.samples::<i32>().map(|s| s.unwrap() as f32 / scale).collect()
        }
    };
    let mono: Vec<f32> = samples.chunks(spec.channels as usize).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect();
    (mono, spec.sample_rate)
}

#[test]
#[ignore = "needs the ONNX models and onnxruntime.dll, see the top of this file"]
fn embedding_of_the_reference_voice_matches_pytorch() {
    let mut engine = load_engine();
    let (audio, rate) = reference_voice();
    let got = engine.embed_voice(&audio, rate).unwrap();
    let want = read_f32(golden("engine_tgt_se.f32"));
    assert_eq!(got.len(), want.len());
    let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("embedding: cosine {:.6}, worst difference {worst:.5}", cosine(&got, &want));
    assert!(cosine(&got, &want) > 0.999);
}

#[test]
#[ignore = "needs the ONNX models and onnxruntime.dll, see the top of this file"]
fn speech_matches_the_python_original_when_nothing_is_random() {
    let mut engine = load_engine();
    let voice = read_f32(golden("engine_tgt_se.f32")); // PyTorch's own embedding, so only the synthesis is compared
    let meta: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(golden("engine_meta.json")).unwrap()).unwrap();
    let params = Params { sdp_ratio: 0.0, noise_scale: 0.0, noise_scale_w: 0.0, tau: 0.0, ..Params::default() };
    // one throw-away run: the first call loads kernels and is not what a user waits for
    engine.synthesize("Hello.", &voice, &params).unwrap();
    for (i, item) in meta["texts"].as_array().unwrap().iter().enumerate() {
        let text = item["text"].as_str().unwrap();
        let want = read_f32(golden(&format!("engine_{i}.f32")));
        let started = Instant::now();
        let got = engine.synthesize(text, &voice, &params).unwrap();
        let took = started.elapsed().as_secs_f64();
        let c = correlation(&got, &want);
        println!("  {:?}", engine.last_timings);
        println!("{text:?}: {} samples (python {}), correlation {c:.5}, {:.0} ms for {:.1} s of speech (RTF {:.3})", got.len(), want.len(), took * 1000.0, got.len() as f64 / 22050.0, took / (got.len() as f64 / 22050.0));
        assert!((got.len() as i64 - want.len() as i64).abs() <= 512, "length differs by more than one frame");
        // With CUDA the rounding of the predicted durations can differ by a frame from the CPU/PyTorch one, which shifts the rest of
        // the sentence a little and lowers the correlation without a difference anyone could hear.
        assert!(c > 0.95, "correlation {c}");
    }
}
