"""Record runtime quantizer bytes and the probe's D table before the freeze."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

from experiments.numerics_rows.common import check_runtime, check_source, sha256, write_json

PHASES = (1, 1 + 2**-3, 1 + 2**-2, 1 + 3 * 2**-3, 1 + 2**-1,
          1 + 5 * 2**-3, 1 + 3 * 2**-2, 1 + 7 * 2**-3, 1 + 2**-7, 1 + 3 * 2**-7)


def dom(limit=10.0):
    import numpy as np

    values = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    values = values[np.isfinite(values) & (values < limit)]
    return np.sort(np.append(values, np.float32(limit)))


def artifact_scales(root):
    from experiments.dense_fused_oracle import Store

    store = Store(str(root))
    result = {}
    for key in store.index:
        if key.endswith(".input_global_scale"):
            value = float(store.get(key).float().reshape(-1)[0])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid artifact scale: {key}")
            result[key] = value
    if not result:
        raise ValueError("artifact has no input_global_scale tensors")
    return result


def inputs(family, gs):
    import torch

    values, labels = [0.0], [{"zero": True}]
    exponents = range(-133, 128) if family == "e2m1" else (range(-133, 1) if family == "e4m3" else (0, 64))
    for j in exponents:
        base = 6 * 2.0**j / gs if family == "e2m1" else 2.0**j
        for phase in PHASES:
            for sign in (1, -1):
                value = sign * base * phase
                if abs(value) <= float(torch.finfo(torch.bfloat16).max):
                    values.append(value)
                    labels.append({"j": j, "phase": phase, "sign": sign})
    x = torch.zeros(len(values), 256, dtype=torch.bfloat16, device="cuda")
    x[:, 0] = torch.tensor(values, dtype=torch.bfloat16, device="cuda")
    return x, labels


def quantize(family, x, gs=1.0):
    import torch
    from tessera.serving.native_ops import native_fp8_quant
    from vllm import _custom_ops as ops

    if family == "e4m3":
        return native_fp8_quant(x.contiguous())
    if family == "e2m1":
        return ops.scaled_fp4_quant(x.contiguous(), torch.tensor([gs], device=x.device, dtype=torch.float32),
                                   is_sf_swizzled_layout=False)
    return x.clone(), torch.ones((x.shape[0], 1), dtype=torch.float32, device=x.device)


def record_quantizer(family, gs, out, boundaries):
    import numpy as np
    import torch
    from experiments.numerics_rows.common import BoundaryRecorder

    x, labels = inputs(family, gs)
    recorder = BoundaryRecorder(torch.cuda.synchronize, boundaries)
    first = recorder.call(f"{family}:{gs}:first", "quantizer", lambda: quantize(family, x, gs))
    second = recorder.call(f"{family}:{gs}:repeat", "quantizer", lambda: quantize(family, x, gs))
    repeat = all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(first, second))
    codes, scales = first
    path = out / f"{family}-{struct.pack('<f', gs).hex()}.npz"
    np.savez(path, input_bits=x.view(torch.int16).cpu().numpy(),
             code_bytes=codes.view(torch.uint8).cpu().numpy(), scale_bytes=scales.view(torch.uint8).cpu().numpy())
    scale_values = scales.view(torch.float8_e4m3fn).float() if family == "e2m1" else scales.float()
    scale_values = scale_values.reshape(x.shape[0], -1)
    normal = (scale_values[:, 0] >= (2**-6 if family == "e2m1" else 2**-126)) & torch.isfinite(scale_values[:, 0])
    saturation = (x[:, 0].float().abs() * gs > 6 * 448) if family == "e2m1" else torch.zeros_like(normal)
    candidates = [label["j"] for i, label in enumerate(labels)
                  if label.get("phase") == 1 and label.get("sign") == 1 and normal[i] and not saturation[i]
                  and x[i, 0] != 0]
    result = {"family": family, "gs": gs, "path": str(path), "sha256": sha256(path), "labels": labels,
              "repeat_equal": repeat, "finite_scales": bool(torch.isfinite(scale_values).all()),
              "saturated_rows": saturation.nonzero().flatten().cpu().tolist(),
              "zero_rows": (x.float().abs().amax(1) == 0).nonzero().flatten().cpu().tolist(),
              "smallest_normal_unsaturated_j": min(candidates) if candidates else None}
    if not repeat:
        raise ValueError("quantizer calls gave different bytes")
    return result


def probe_table(library, out, limit=10.0):
    import numpy as np

    for label in ("library", "probe"):
        if sha256(library[label]["path"]) != library[label]["sha256"]:
            raise ValueError(f"changed {label} binary")
    gf = dom(limit)
    source, target = out / "dom.f32", out / f"D-{library['key']}.f32"
    gf.astype("<f4").tofile(source)
    subprocess.run([library["probe"]["path"], str(source), str(target)], check=True)
    d = np.fromfile(target, dtype="<f4")
    if d.shape != gf.shape or not np.isfinite(d).all():
        raise ValueError("probe returned a nonfinite or incomplete D table")
    mask = gf >= 0
    return {"key": library["key"], "L_bits": struct.unpack("<I", struct.pack("<f", limit))[0],
            "dom_size": len(gf), "dom_path": str(source), "dom_sha256": sha256(source),
            "d_table_path": str(target), "d_table_sha256": sha256(target),
            "d_nondecreasing_0_L": bool(np.all(d[mask][1:] >= d[mask][:-1])),
            "library_so_sha256": library["library"]["sha256"], "probe_sha256": library["probe"]["sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--image", choices=("G", "N"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--artifact", type=Path)
    args = p.parse_args()
    check_source(args.source)
    check_runtime(args.image)
    sys.path.insert(0, str(args.source / "src"))
    import tessera
    import torch

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    built = json.loads(args.build.read_text())
    scales = artifact_scales(args.artifact) if args.image == "G" else {}
    records, boundaries = [], []
    for family in (("e4m3", "e2m1") if args.image == "G" else ("value",)):
        for gs in sorted(set(scales.values())) if family == "e2m1" else (1.0,):
            records.append(record_quantizer(family, gs, args.out, boundaries))
    write_json(args.out / "q1.json", {"quantizers": records, "artifact_scales": scales,
                                      "tables": [probe_table(lib, args.out) for lib in built["libraries"]],
                                      "boundaries": boundaries})


if __name__ == "__main__":
    main()
