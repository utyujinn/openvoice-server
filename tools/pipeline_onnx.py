"""End to end with ONNX Runtime only for the neural parts: text -> [BERT, base voice, tone color converter] -> wav.
The text front end (normalizing, phonemes with tones, BERT tokenizer) is still the Python one: it is what a Rust
port would have to replace. Prints the time of every stage per sentence and writes the wavs for listening.

BERT: MeloTTS takes hidden_states[-3] of bert-base-uncased, i.e. the output of layer 10 of 12, so the exported graph
keeps only the first 10 layers (two fewer to compute, same numbers).
"""
import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT / "MeloTTS"))
sys.path.insert(0, str(ROOT / "OpenVoice"))

import numpy as np
import soundfile as sf
import librosa
import torch
import onnxruntime as ort

OUT = HERE / "out"
OUT_WAV = ROOT.parent / "listen_en" / "openvoice_onnx"
OUT_WAV.mkdir(parents=True, exist_ok=True)


def export_bert():
    path = OUT / "bert10.onnx"
    if path.exists():
        return path
    from transformers import AutoModel

    model = AutoModel.from_pretrained("bert-base-uncased").eval()
    model.encoder.layer = model.encoder.layer[:10]

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    ids = torch.randint(1000, 2000, (1, 12))
    torch.onnx.export(Wrap(model), (ids, torch.ones_like(ids)), str(path), opset_version=17, dynamo=False,
                      input_names=["input_ids", "attention_mask"], output_names=["hidden"], dynamic_axes={"input_ids": {1: "L"}, "attention_mask": {1: "L"}, "hidden": {1: "L"}})
    return path


def providers():
    # HEURISTIC: the default (EXHAUSTIVE) searches for the fastest convolution algorithm again for every new input
    # length, which is every sentence here.
    return [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"]


def main():
    from melo.text import cleaner
    from melo import commons, utils
    from melo.api import TTS
    from openvoice.mel_processing import spectrogram_torch
    from transformers import AutoTokenizer

    tts = TTS(language="EN", device="cpu")  # only for hps and symbols: the neural part runs in onnxruntime
    hps = tts.hps
    spk = hps.data.spk2id["EN-US"]
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert = ort.InferenceSession(str(export_bert()), providers=providers())
    melo = ort.InferenceSession(str(OUT / "melo.onnx"), providers=providers())
    conv = ort.InferenceSession(str(OUT / "converter.onnx"), providers=providers())
    ref_enc = ort.InferenceSession(str(OUT / "ref_enc.onnx"), providers=["CPUExecutionProvider"])

    from huggingface_hub import snapshot_download
    ckpt = Path(snapshot_download("myshell-ai/OpenVoiceV2"))
    src_se = torch.load(ckpt / "base_speakers" / "ses" / "en-us.pth", map_location="cpu").numpy()
    conv_hps = json.loads((ckpt / "converter" / "config.json").read_text())["data"]
    ref_audio, _ = librosa.load(str(ROOT.parent / "common" / "ref_en.wav"), sr=conv_hps["sampling_rate"])
    ref_spec = spectrogram_torch(torch.FloatTensor(ref_audio).unsqueeze(0), conv_hps["filter_length"], conv_hps["sampling_rate"], conv_hps["hop_length"], conv_hps["win_length"], center=False)
    tgt_se = ref_enc.run(None, {"spec_t": ref_spec.transpose(1, 2).numpy()})[0][..., None]

    def synth(text, tau=0.3, noise=(0.6, 0.8, 0.2)):
        t0 = time.perf_counter()
        norm, phones, tones, word2ph = cleaner.clean_text(text, "EN")
        phone, tone, lang = utils.cleaned_text_to_sequence(phones, tones, "EN", tts.symbol_to_id)
        phone, tone, lang = commons.intersperse(phone, 0), commons.intersperse(tone, 0), commons.intersperse(lang, 0)
        word2ph = [w * 2 for w in word2ph]
        word2ph[0] += 1
        t_front = time.perf_counter()

        enc = tok(norm, return_tensors="np")
        hidden = bert.run(None, {"input_ids": enc["input_ids"].astype(np.int64), "attention_mask": enc["attention_mask"].astype(np.int64)})[0][0]
        feat = np.concatenate([np.repeat(hidden[i:i + 1], word2ph[i], axis=0) for i in range(len(word2ph))], axis=0).T.astype(np.float32)
        t_bert = time.perf_counter()

        L = len(phone)
        feed = {"x": np.array([phone], np.int64), "x_lengths": np.array([L], np.int64), "sid": np.array([spk], np.int64), "tone": np.array([tone], np.int64),
                "language": np.array([lang], np.int64), "bert": np.zeros((1, 1024, L), np.float32), "ja_bert": feat[None], "noise_scale": np.array(noise[0], np.float32),
                "length_scale": np.array(1.0, np.float32), "noise_scale_w": np.array(noise[1], np.float32), "sdp_ratio": np.array(noise[2], np.float32)}
        base = melo.run(None, feed)[0][0, 0]
        t_melo = time.perf_counter()

        base22 = librosa.resample(base, orig_sr=hps.data.sampling_rate, target_sr=conv_hps["sampling_rate"])
        spec = spectrogram_torch(torch.FloatTensor(base22).unsqueeze(0), conv_hps["filter_length"], conv_hps["sampling_rate"], conv_hps["hop_length"], conv_hps["win_length"], center=False).numpy()
        t_spec = time.perf_counter()

        out = conv.run(None, {"spec": spec, "spec_lengths": np.array([spec.shape[-1]], np.int64), "g_src": src_se, "g_tgt": tgt_se, "tau": np.array(tau, np.float32)})[0][0, 0]
        t_end = time.perf_counter()
        return out, base, dict(front=t_front - t0, bert=t_bert - t_front, melo=t_melo - t_bert, spec=t_spec - t_melo, conv=t_end - t_spec, total=t_end - t0)

    sentences = json.loads((ROOT.parent / "common" / "sentences.json").read_text(encoding="utf-8"))
    synth("Hello, nice to meet you.")  # warm-up
    rows = []
    for s in sentences:
        out, base, t = synth(s["text"])
        audio_s = len(out) / conv_hps["sampling_rate"]
        sf.write(str(OUT_WAV / f"{s['id']:02d}.wav"), out, conv_hps["sampling_rate"])
        rows.append((s["id"], audio_s, t))
        print(f"{s['id']:>2} {audio_s:4.1f}s  total {t['total']*1000:4.0f}ms (RTF {t['total']/audio_s:.3f})  front {t['front']*1000:3.0f} bert {t['bert']*1000:3.0f} melo {t['melo']*1000:3.0f} spec {t['spec']*1000:3.0f} conv {t['conv']*1000:3.0f}")
    rtf = [r[2]["total"] / r[1] for r in rows]
    print("PIPELINE RTF median %.3f max %.3f" % (float(np.median(rtf)), max(rtf)))
    for k in ("front", "bert", "melo", "spec", "conv"):
        print(f"  mean {k}: {np.mean([r[2][k] for r in rows])*1000:.0f} ms")
    print("wrote", OUT_WAV)


if __name__ == "__main__":
    main()
