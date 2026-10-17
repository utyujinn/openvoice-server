"""Spike: export MeloTTS (English) SynthesizerTrn.infer to ONNX and compare it with PyTorch.

  melo.onnx : (x, x_lengths, sid, tone, language, bert, ja_bert, noise_scale, length_scale, noise_scale_w, sdp_ratio)
              -> audio[1,1,N]
Run deterministically (noise_scale = noise_scale_w = 0, sdp_ratio = 0: only the plain duration predictor) so the
two outputs can be compared sample by sample. The text front end (cleaning, phonemes, BERT features) is the
Python one here; this spike only answers whether the neural part exports and runs.
"""
import os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT / "MeloTTS"))

import numpy as np
import soundfile as sf
import torch

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

TEXT = "Could you pass me that, please? Thank you so much."


class MeloGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, x_lengths, sid, tone, language, bert, ja_bert, noise_scale, length_scale, noise_scale_w, sdp_ratio):
        return self.model.infer(x, x_lengths, sid, tone, language, bert, ja_bert, noise_scale=noise_scale, length_scale=length_scale,
                                noise_scale_w=noise_scale_w, sdp_ratio=sdp_ratio)[0]


def main():
    from melo.api import TTS
    from melo import utils

    tts = TTS(language="EN", device="cuda")
    model = tts.model.eval()
    spk_id = tts.hps.data.spk2id["EN-US"]
    bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(TEXT, "EN", tts.hps, "cuda", tts.symbol_to_id)
    x = phones.cuda().unsqueeze(0)
    tone = tones.cuda().unsqueeze(0)
    lang = lang_ids.cuda().unsqueeze(0)
    bert = bert.cuda().unsqueeze(0)
    ja_bert = ja_bert.cuda().unsqueeze(0)
    x_len = torch.LongTensor([phones.size(0)]).cuda()
    sid = torch.LongTensor([spk_id]).cuda()
    ns, ls, nsw, sdp = (torch.tensor(v, dtype=torch.float32).cuda() for v in (0.0, 1.0, 0.0, 0.0))
    print("inputs: x", tuple(x.shape), "bert", tuple(bert.shape), "ja_bert", tuple(ja_bert.shape), "phones", int(x_len))

    graph = MeloGraph(model).eval()
    with torch.no_grad():
        want = graph(x, x_len, sid, tone, lang, bert, ja_bert, ns, ls, nsw, sdp).cpu().numpy()
    print("pytorch out", want.shape)

    path = OUT / "melo.onnx"
    torch.onnx.export(
        graph, (x, x_len, sid, tone, lang, bert, ja_bert, ns, ls, nsw, sdp), str(path), opset_version=17, dynamo=False,
        input_names=["x", "x_lengths", "sid", "tone", "language", "bert", "ja_bert", "noise_scale", "length_scale", "noise_scale_w", "sdp_ratio"],
        output_names=["audio"],
        dynamic_axes={"x": {1: "L"}, "tone": {1: "L"}, "language": {1: "L"}, "bert": {2: "L"}, "ja_bert": {2: "L"}, "audio": {2: "N"}},
    )
    print("exported", path, round(path.stat().st_size / 2**20, 1), "MiB")

    import onnxruntime as ort

    feed = {
        "x": x.cpu().numpy(), "x_lengths": x_len.cpu().numpy(), "sid": sid.cpu().numpy(), "tone": tone.cpu().numpy(), "language": lang.cpu().numpy(),
        "bert": bert.cpu().numpy(), "ja_bert": ja_bert.cpu().numpy(), "noise_scale": np.array(0.0, np.float32), "length_scale": np.array(1.0, np.float32),
        "noise_scale_w": np.array(0.0, np.float32), "sdp_ratio": np.array(0.0, np.float32),
    }
    for providers in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
        sess = ort.InferenceSession(str(path), providers=providers)
        got = sess.run(None, feed)[0]
        n = min(got.shape[-1], want.shape[-1])
        diff = np.abs(got[..., :n] - want[..., :n])
        corr = float(np.corrcoef(got[..., :n].ravel(), want[..., :n].ravel())[0, 1])
        print(providers[0], "len onnx/torch", got.shape[-1], want.shape[-1], "max|diff|", float(diff.max()), "corr", round(corr, 6))
        for _ in range(2):
            sess.run(None, feed)
        t = time.perf_counter()
        reps = 10
        for _ in range(reps):
            sess.run(None, feed)
        dt = (time.perf_counter() - t) / reps
        audio_s = got.shape[-1] / 44100
        print(f"RESULT {providers[0]}: {dt*1000:.0f} ms for {audio_s:.1f} s of base audio -> RTF {dt/audio_s:.3f}")
        sf.write(str(OUT / f"melo_{providers[0][:4]}.wav"), got[0, 0], 44100)
    sf.write(str(OUT / "melo_torch.wav"), want[0, 0], 44100)


if __name__ == "__main__":
    main()
