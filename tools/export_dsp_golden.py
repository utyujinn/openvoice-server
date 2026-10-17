"""Golden data for the Rust audio helpers: the spectrogram the converter takes (OpenVoice's spectrogram_torch) and the
44.1 kHz -> 22.05 kHz resampling librosa does between the base voice and the converter."""
import json, os, struct, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLD = Path(r"C:\Users\Utyujin\Dev\Mutelink") / "openvoice-server" / "tests" / "golden"
sys.path.insert(0, str(ROOT / "OpenVoice"))

import numpy as np
import librosa
import soundfile as sf
import torch
from openvoice.mel_processing import spectrogram_torch


def main():
    src = ROOT.parent / "listen_en" / "openvoice_sweep" / "_base_EN-US" / "04.wav"
    audio44, sr = sf.read(str(src), dtype="float32")
    assert sr == 44100, sr
    audio44 = audio44[: 44100 * 3]
    sf.write(str(GOLD / "dsp_base_44k.wav"), audio44, 44100, subtype="FLOAT")
    audio22 = librosa.resample(audio44, orig_sr=44100, target_sr=22050)
    spec = spectrogram_torch(torch.FloatTensor(audio22).unsqueeze(0), 1024, 22050, 256, 1024, center=False)[0].numpy()  # [513, T]
    (GOLD / "dsp_audio_22k.f32").write_bytes(audio22.astype("<f4").tobytes())
    (GOLD / "dsp_spec.f32").write_bytes(np.ascontiguousarray(spec).astype("<f4").tobytes())
    (GOLD / "dsp_meta.json").write_text(json.dumps({"n22": int(len(audio22)), "bins": int(spec.shape[0]), "frames": int(spec.shape[1])}))
    print("wrote", len(audio44), "->", len(audio22), "samples; spec", spec.shape)


if __name__ == "__main__":
    main()
