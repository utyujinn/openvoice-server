# Halves the size of the ONNX files: every large float32 weight is stored as float16 and cast back to float32 as the first step of
# the graph, so the arithmetic is unchanged and the outputs move only by the rounding of the stored weights. (Real int8 quantization
# was tried on BERT and moved its features too far: worst cosine 0.90 against fp32; this keeps it at 1.00000.)
#   python shrink_models.py <models dir (fp32)> <output dir>
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MIN_ELEMENTS = 4096
# melo.onnx stays as it is: its duration predictor rounds the length of every sound up, so fp16 weights changed the length of a
# sentence by 334 samples in the golden test and the speech no longer lined up with the original (correlation 0.07).
KEEP = {"melo.onnx"}


def shrink(src: Path, dst: Path):
    m = onnx.load(str(src))
    casts, kept = [], []
    for init in m.graph.initializer:
        if init.data_type == TensorProto.FLOAT and int(np.prod(init.dims)) >= MIN_ELEMENTS:
            name16 = init.name + "__fp16"
            kept.append(numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), name16))
            casts.append(helper.make_node("Cast", [name16], [init.name], to=TensorProto.FLOAT, name="cast_" + init.name.replace("/", "_")))
        else:
            kept.append(init)
    nodes = list(m.graph.node)
    del m.graph.initializer[:]
    m.graph.initializer.extend(kept)
    del m.graph.node[:]
    m.graph.node.extend(casts + nodes)
    onnx.save(m, str(dst))


if __name__ == "__main__":
    src_dir, dst_dir = Path(sys.argv[1]), Path(sys.argv[2])
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.glob("*.onnx")):
        if f.name in KEEP:
            (dst_dir / f.name).write_bytes(f.read_bytes())
        else:
            shrink(f, dst_dir / f.name)
        print(f"{f.name}: {f.stat().st_size / 1e6:.0f} MB -> {(dst_dir / f.name).stat().st_size / 1e6:.0f} MB")
