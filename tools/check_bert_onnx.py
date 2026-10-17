"""Does bert10.onnx give the features MeloTTS takes from the full BERT (hidden_states[-3])?"""
import os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForMaskedLM

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
full = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval()
sess = ort.InferenceSession(str(HERE / "out" / "bert10.onnx"), providers=["CPUExecutionProvider"])
for text in ("hello, nice to meet you.", "could you pass me that, please? thank you so much.", "i have three cats and twenty-one dogs."):
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        want = full(**enc, output_hidden_states=True)["hidden_states"][-3][0].numpy()
    got = sess.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0][0]
    print(f"{text[:30]!r}: shapes {got.shape} {want.shape} max|diff| {np.abs(got - want).max():.5f} corr {np.corrcoef(got.ravel(), want.ravel())[0, 1]:.6f}")
