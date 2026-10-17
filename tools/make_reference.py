# The reference voice of the tests: 8 s cut from LibriVox's recording of Pride and Prejudice, chapter 1 (read by Elizabeth Klett; LibriVox
# recordings are in the public domain). https://archive.org/details/prideandprejudice_1005_librivox
# Written as a 24 kHz float WAV (what the app gets from a browser recording) with librosa's 22.05 kHz version beside it, which
# the resampler test compares with.
#   python make_reference.py <prideandprejudice_01_austen_64kb.mp3> <common dir> <tests/golden dir>
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

mp3, common, golden = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
START, SECONDS = 80.0, 8.0
y, _ = librosa.load(mp3, sr=24000, mono=True, offset=START, duration=SECONDS)
y = (y / np.abs(y).max() * 0.9).astype(np.float32)
sf.write(common / "ref_en_librivox.wav", y, 24000, subtype="FLOAT")
sf.write(golden / "ref_en.wav", y, 24000, subtype="FLOAT")
y22 = librosa.resample(y, orig_sr=24000, target_sr=22050, res_type="kaiser_best")
(golden / "ref_22k.f32").write_bytes(y22.astype("<f4").tobytes())
print("reference", len(y) / 24000, "s;", len(y22), "samples at 22.05 kHz")
