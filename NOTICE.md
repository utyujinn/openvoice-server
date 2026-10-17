# Third-party models, data and software

This repository holds the Rust runtime and the scripts that convert the models below to ONNX. It contains **no model weights**: the
converted ONNX files are published as assets of a GitHub release, together with [MODEL_LICENSES.md](MODEL_LICENSES.md), which holds
the original license text and copyright notice of each component.

| Component | License | Used for |
|---|---|---|
| [myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS) | MIT | English base voice (`melo.onnx`), text normalization and sentence splitting (ported to Rust) |
| [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice) V2 ([weights](https://huggingface.co/myshell-ai/OpenVoiceV2)) | MIT | tone color converter (`converter.onnx`), reference voice encoder (`ref_enc.onnx`), the accents' speaker embeddings (`data/se_*.bin`) |
| [google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) | Apache-2.0 | first 10 layers as `bert10.onnx`; its tokenizer (`data/bert_tokenizer.json`) |
| [Kyubyong/g2p](https://github.com/Kyubyong/g2p) (`g2p_en`) | Apache-2.0 | grapheme-to-phoneme network and logic (ported to Rust; weights in `data/g2p_en.bin`) |
| [CMU Pronouncing Dictionary](http://www.speech.cs.cmu.edu/cgi-bin/cmudict) | BSD-style (see MODEL_LICENSES.md) | `data/cmudict_*.tsv` |
| [jaraco/inflect](https://github.com/jaraco/inflect) | MIT | number-to-words rules (ported to Rust) |
| [bmcfee/resampy](https://github.com/bmcfee/resampy) | ISC | the `kaiser_best` resampler (ported to Rust, filter parameters taken from its data) |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT | inference |
| NVIDIA CUDA / cuDNN runtime | NVIDIA proprietary licenses | **Not redistributed by this repository.** Check the CUDA EULA / cuDNN SLA before shipping them. |

The reference voice of the tests (`tests/golden/ref_en.wav`) is 8 seconds cut from the LibriVox recording of *Pride and Prejudice*
(read by Elizabeth Klett), which is in the public domain. See `tools/make_reference.py`.

## Voice cloning

The converter makes speech in the voice of whatever reference recording it is given. Applications built on this repository should
tell their users to register only their own voice, or a voice whose owner has clearly agreed, and should not use it to impersonate
anyone or to mislead others. This runtime does not add a watermark to the audio.
