# openvoice-server

English text to speech in a registered reference voice, for Mutelink: [MeloTTS](https://github.com/myshell-ai/MeloTTS) English
(the base voice) and [OpenVoice](https://github.com/myshell-ai/OpenVoice) V2 (the tone color converter), both MIT licensed, run with
ONNX Runtime. It is a child process of the app, like `irodori-server`: it talks over stdin / stdout (`src/protocol.rs`), so the app
can end it to give back all the GPU memory.

It is its own Cargo workspace on purpose: it needs the upstream `ort` crate, and the app's graph already holds voicevox_core's fork
of it (both declare `links = "onnxruntime"`). Build and test it here, not from `tauri-src/src-tauri`.

```
cargo build --release          # target/release/openvoice-server.exe
cargo test --release           # text front end, audio helpers, protocol: no models needed (18 tests)
```

The engine tests need the ONNX models (`models/`, not in git) and an `onnxruntime.dll` (with the CUDA libraries on PATH for `OPENVOICE_CUDA=1`):

```
OPENVOICE_ORT_DLL=<onnxruntime.dll> OPENVOICE_CUDA=1 cargo test --release --test engine --test server -- --ignored --nocapture
```

## What is where

- `src/text/` — the English front end of MeloTTS in Rust, exact to the Python on all the golden sentences (`tests/golden/README.md`):
  `normalize` (numbers, times, abbreviations), `numwords` (inflect's number words), `split` (sentence splitting), `g2p` (CMU dictionary and
  g2p_en's network for unknown words), `frontend` (ids for the models).
- `src/dsp.rs` — the spectrogram and resampy's `kaiser_best` resampler. The converter and the reference voice encoder see the faintest
  content of the spectrum, so the resampler had to be an exact port (a plain windowed sinc gave a different voice).
- `src/engine.rs` — BERT, base voice, converter, reference voice embedding.
- `src/main.rs` — the commands: `load`, `register_voice`, `synthesize` (one answer per piece of text, so playback can start early).
- `data/` — dictionaries, tokenizer, base voice embeddings, g2p weights (small; the app downloads them).
- `models/` — the ONNX graphs, 426 MB in all: `melo.onnx` (162 MB), `converter.onnx` (61 MB), `bert10.onnx` (181 MB, BERT's first 10 layers: MeloTTS reads the
  output of layer 10 of 12), `ref_enc.onnx` (2 MB). Made by `tools/export_*.py` and `tools/pipeline_onnx.py` from the original weights, then
  `tools/shrink_models.py` stores the large weights of all but `melo.onnx` as float16 (cast back to float32 inside the graph): half the size,
  the same arithmetic. Int8 quantization was tried and was too inexact; `melo.onnx` stays float32 because its duration rounding is sensitive.
  `melo.onnx` is exported with the spline rewritten without boolean-mask indexing (`tools/export_melo_nz.py`): the mask indexing became `NonZero` ops,
  each a wait for the GPU, which a game filling the GPU stretches to hundreds of milliseconds.

## Known limits

- English only (MeloTTS has Spanish, French, Chinese, Japanese and Korean bases too; not ported or judged).
- The 18 rare plural homographs that g2p_en resolves by part of speech (e.g. "retakes") are not; they are spelled out by the network.
- Speed depends on how busy the GPU is: with VRChat running, the real-time factor is about 0.1 - 0.25 on an RTX 4070.

## Release

The executable is built by GitHub Actions (`.github/workflows/release.yml`). Raise `version` in `Cargo.toml`, commit, and push a tag `server-vX.Y.Z` with the same number: the workflow runs the unit tests, builds with `--locked`, and publishes the exe as the release of that tag with its SHA-256 in the notes (Mutelink pins that checksum). "Run workflow" on the Actions page only builds and tests, and keeps the exe as an artifact. The models and data are the assets of release `models-v1` (made with `tools/shrink_models.py`), uploaded by hand.

## License

MIT ([LICENSE](LICENSE)). The models, data and the software this is ported from are under their own licenses: see [NOTICE.md](NOTICE.md) and [MODEL_LICENSES.md](MODEL_LICENSES.md).
