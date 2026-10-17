"""Spike: export OpenVoice V2's tone color converter to ONNX and compare it with PyTorch.

Two graphs:
  converter.onnx : (spec[1,513,T], spec_lengths[1], g_src[1,256,1], g_tgt[1,256,1], tau[]) -> audio[1,1,N]
  ref_enc.onnx   : spec^T[1,T,513] -> g[1,256]      (the speaker embedding of a reference clip, once per voice)
The spectrogram itself (STFT magnitude) stays outside the graphs here; in the app it is a small FFT routine.
tau = 0 makes the converter deterministic, so PyTorch and ONNX outputs can be compared sample by sample.
"""
import os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
sys.path.insert(0, str(ROOT / "OpenVoice"))

import numpy as np
import soundfile as sf
import librosa
import torch
from huggingface_hub import snapshot_download
from openvoice.api import ToneColorConverter, OpenVoiceBaseClass
from openvoice.mel_processing import spectrogram_torch

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


class NoWM(ToneColorConverter):
    def __init__(self, *a, **k):
        OpenVoiceBaseClass.__init__(self, *a, **k)
        self.watermark_model = None
        self.version = getattr(self.hps, "_version_", "v1")


class ConverterGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, spec, spec_lengths, g_src, g_tgt, tau):
        return self.model.voice_conversion(spec, spec_lengths, sid_src=g_src, sid_tgt=g_tgt, tau=tau)[0]


class RefEncGraph(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ref_enc = model.ref_enc

    def forward(self, spec_t):
        return self.ref_enc(spec_t)


def spec_of(conv, wav_path):
    hps = conv.hps
    audio, _ = librosa.load(wav_path, sr=hps.data.sampling_rate)
    y = torch.FloatTensor(audio).unsqueeze(0).to(conv.device)
    return spectrogram_torch(y, hps.data.filter_length, hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length, center=False)


def main():
    ckpt = Path(snapshot_download("myshell-ai/OpenVoiceV2"))
    conv = NoWM(str(ckpt / "converter" / "config.json"), device="cuda")
    conv.load_ckpt(str(ckpt / "converter" / "checkpoint.pth"))
    conv.model.eval()

    base = ROOT.parent / "listen_en" / "openvoice_sweep" / "_base_EN-US" / "01.wav"
    ref = ROOT.parent / "common" / "ref_en.wav"
    spec = spec_of(conv, str(base))
    spec_lengths = torch.LongTensor([spec.size(-1)]).cuda()
    src_se = torch.load(ckpt / "base_speakers" / "ses" / "en-us.pth", map_location="cuda")
    tgt_se = conv.extract_se(str(ref))
    tau = torch.tensor(0.0).cuda()

    graph = ConverterGraph(conv.model).eval()
    with torch.no_grad():
        want = graph(spec, spec_lengths, src_se, tgt_se, tau).cpu().numpy()
    print("pytorch out", want.shape)

    path = OUT / "converter.onnx"
    torch.onnx.export(
        graph, (spec, spec_lengths, src_se, tgt_se, tau), str(path), opset_version=17, dynamo=False,
        input_names=["spec", "spec_lengths", "g_src", "g_tgt", "tau"], output_names=["audio"],
        dynamic_axes={"spec": {2: "T"}, "audio": {2: "N"}},
    )
    print("exported", path, round(path.stat().st_size / 2**20, 1), "MiB")

    spec_t = spec.transpose(1, 2)
    ref_graph = RefEncGraph(conv.model).eval()
    ref_spec = spec_of(conv, str(ref)).transpose(1, 2)
    with torch.no_grad():
        want_g = ref_graph(ref_spec).cpu().numpy()
    torch.onnx.export(ref_graph, (ref_spec,), str(OUT / "ref_enc.onnx"), opset_version=17, dynamo=False,
                      input_names=["spec_t"], output_names=["g"], dynamic_axes={"spec_t": {1: "T"}})
    print("exported ref_enc.onnx")

    import onnxruntime as ort
    for providers in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
        sess = ort.InferenceSession(str(path), providers=providers)
        feed = {"spec": spec.cpu().numpy(), "spec_lengths": spec_lengths.cpu().numpy(), "g_src": src_se.cpu().numpy(), "g_tgt": tgt_se.cpu().numpy(), "tau": np.array(0.0, dtype=np.float32)}
        got = sess.run(None, feed)[0]
        diff = np.abs(got - want)
        corr = float(np.corrcoef(got.ravel(), want.ravel())[0, 1])
        print(providers[0], "shape", got.shape, "max|diff|", float(diff.max()), "mean|diff|", float(diff.mean()), "corr", round(corr, 6))
        for _ in range(2):
            sess.run(None, feed)
        t = time.perf_counter()
        n = 10
        for _ in range(n):
            sess.run(None, feed)
        dt = (time.perf_counter() - t) / n
        audio_s = got.shape[-1] / 22050
        print(f"  {providers[0]}: {dt*1000:.0f} ms for {audio_s:.1f} s of audio -> RTF {dt/audio_s:.3f}")
        sf.write(str(OUT / f"converted_{providers[0][:4]}.wav"), got[0, 0], 22050)

    sess = ort.InferenceSession(str(OUT / "ref_enc.onnx"), providers=["CPUExecutionProvider"])
    got_g = sess.run(None, {"spec_t": ref_spec.cpu().numpy()})[0]
    print("ref_enc max|diff|", float(np.abs(got_g - want_g).max()))


if __name__ == "__main__":
    main()
