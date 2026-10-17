"""Re-export melo.onnx without the data dependent `NonZero` ops.

MeloTTS' rational quadratic spline (used by the stochastic duration predictor) picks the elements inside the interval with
boolean mask indexing, which ONNX turns into NonZero (+ GatherND / ScatterND). NonZero hands its result size back to the CPU,
a synchronisation with the GPU; with a game (VRChat) filling the GPU it took 280-310 ms per sentence for some sentences.
Here the spline is evaluated for every element, with the inputs clamped into the interval, and torch.where keeps the original
value of the elements that were outside it: the same numbers, with no shape that depends on the data.
"""
import os, shutil, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault("HF_HOME", str(ROOT / "hf_home"))
os.environ.setdefault("NLTK_DATA", str(ROOT / "nltk_data"))
sys.path.insert(0, str(ROOT / "MeloTTS"))
sys.path.insert(0, str(HERE))

import numpy as np
import torch
import torch.nn.functional as F
import melo.transforms as T


def unconstrained_rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, inverse=False, tails="linear", tail_bound=1.0,
                                            min_bin_width=T.DEFAULT_MIN_BIN_WIDTH, min_bin_height=T.DEFAULT_MIN_BIN_HEIGHT, min_derivative=T.DEFAULT_MIN_DERIVATIVE):
    if tails != "linear":
        raise RuntimeError("{} tails are not implemented.".format(tails))
    inside = (inputs >= -tail_bound) & (inputs <= tail_bound)
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant
    outputs, logabsdet = T.rational_quadratic_spline(
        inputs=inputs.clamp(-tail_bound, tail_bound), unnormalized_widths=unnormalized_widths, unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives, inverse=inverse, left=-tail_bound, right=tail_bound, bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width, min_bin_height=min_bin_height, min_derivative=min_derivative)
    return torch.where(inside, outputs, inputs), torch.where(inside, logabsdet, torch.zeros_like(logabsdet))


def main():
    out = HERE / "out"
    if (out / "melo.onnx").exists() and not (out / "melo_old.onnx").exists():
        shutil.copy(out / "melo.onnx", out / "melo_old.onnx")
    T.unconstrained_rational_quadratic_spline = unconstrained_rational_quadratic_spline
    import export_melo
    export_melo.main()


if __name__ == "__main__":
    main()
