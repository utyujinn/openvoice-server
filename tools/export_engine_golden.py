"""End to end golden audio from the ORIGINAL PyTorch OpenVoice + MeloTTS, with every random element switched off
(noise_scale = noise_scale_w = sdp_ratio = 0, tau = 0), so the Rust engine can be compared with it. Also the speaker
embedding of the reference clip as PyTorch computes it."""
import json, os, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD = Path(r"C:\Users\Utyujin\Dev\Mutelink") / "openvoice-server" / "tests" / "golden"
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT / "OpenVoice"))
sys.path.insert(0, str(ROOT / "MeloTTS"))

import numpy as np
import soundfile as sf
import librosa
import torch

TEXTS = [
    "Hello, nice to meet you.",
    "Could you pass me that, please? Thank you so much.",
    "Ahaha! Oh my gosh, that's so funny, I can't stop laughing!",
]


def main():
    from huggingface_hub import snapshot_download
    from melo.api import TTS
    from openvoice.api import ToneColorConverter, OpenVoiceBaseClass

    class NoWM(ToneColorConverter):
        def __init__(self, *a, **k):
            OpenVoiceBaseClass.__init__(self, *a, **k)
            self.watermark_model = None
            self.version = getattr(self.hps, "_version_", "v1")

    ckpt = Path(snapshot_download("myshell-ai/OpenVoiceV2"))
    tts = TTS(language="EN", device="cuda")
    conv = NoWM(str(ckpt / "converter" / "config.json"), device="cuda")
    conv.load_ckpt(str(ckpt / "converter" / "checkpoint.pth"))
    spk = tts.hps.data.spk2id["EN-US"]
    src_se = torch.load(ckpt / "base_speakers" / "ses" / "en-us.pth", map_location="cuda")
    # 8 s cut from LibriVox's Pride and Prejudice (public domain), made by make_reference.py: 24 kHz float mono
    ref = ROOT.parent / "common" / "ref_en_librivox.wav"
    shutil.copy(ref, GOLD / "ref_en.wav")
    tgt_se = conv.extract_se(str(ref))
    (GOLD / "engine_tgt_se.f32").write_bytes(tgt_se.flatten().cpu().numpy().astype("<f4").tobytes())

    tmp = Path(tempfile.mkdtemp())
    meta = []
    for i, text in enumerate(TEXTS):
        base = tmp / f"base{i}.wav"
        # sdp_ratio / noise_scale / noise_scale_w all 0: deterministic. The pieces are joined with 50 ms of silence, as the library does.
        tts.tts_to_file(text, spk, str(base), sdp_ratio=0.0, noise_scale=0.0, noise_scale_w=0.0, quiet=True)
        out = tmp / f"out{i}.wav"
        conv.convert(str(base), src_se, tgt_se, output_path=str(out), tau=0.0, message="")
        audio, sr = sf.read(str(out), dtype="float32")
        assert sr == 22050
        (GOLD / f"engine_{i}.f32").write_bytes(audio.astype("<f4").tobytes())
        meta.append({"text": text, "samples": int(len(audio))})
    (GOLD / "engine_meta.json").write_text(json.dumps({"texts": meta, "speaker": "EN-US"}))
    print("wrote", meta)


if __name__ == "__main__":
    main()
