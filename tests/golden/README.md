# Golden data

Everything in this folder was made by running the Python original (MeloTTS + OpenVoice V2 + inflect + librosa), and the Rust code
is checked against it. The scripts are in `../../tools/`; they expect the layout of the spike where they were written (`ROOT` at the
top of each: a folder with `MeloTTS/`, `OpenVoice/`, `hf_home/`, `nltk_data/` and a Python environment with torch, onnx and
onnxruntime-gpu).

| file | made by | checked by |
|---|---|---|
| `numbers.json` | `export_numbers_golden.py` | `text::numwords`, `text::normalize` tests |
| `text_en.json` | `export_text_assets.py` | `text::split`, `text::frontend` tests |
| `text_en_fuzz.json` (not in git, 2.8 MB) | `fuzz_golden.py` | `text::frontend::tests::random_texts_match_the_python` (skipped when absent) |
| `ref_en.wav`, `ref_22k.f32` | `make_reference.py` (8 s of a public domain LibriVox recording) |
| `dsp_*` | `export_dsp_golden.py`, `export_engine_golden.py` | `dsp` tests, `tests/engine.rs` |
| `engine_*` | `export_engine_golden.py` (original PyTorch, every random element off) | `tests/engine.rs` (needs the models) |
