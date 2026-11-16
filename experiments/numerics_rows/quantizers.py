"""Record runtime quantizer bytes and the probe's D table before the freeze."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

from experiments.numerics_rows.common import artifact_limit, check_runtime, check_source, register_native_ops, sha256, write_json

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
    from tessera.serving.nvfp4_moe_route import _static_input_global
    import torch
    for module, scheme in store.schemes.items():
        if scheme.get("structure") != "routed_moe":
            continue
        for name, projections in (("gs13", ("gate_proj", "up_proj")), ("gs2", ("down_proj",))):
            values = [result[f"{module}.{e}.{projection}.input_global_scale"]
                      for e in range(scheme["experts"]) for projection in projections]
            result[f"{module}:{name}:runtime_reduction"] = float(_static_input_global(torch.tensor(values, dtype=torch.float32), "cpu"))
    return result


def inputs(family, gs):
    import torch

    values, labels = [0.0, 1.0, -1.0], [{"zero": True}, {"chunk": True}, {"chunk": True}]
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

    if family == "e4m3":
        return native_fp8_quant(x.contiguous())
    if family == "e2m1":
        from tessera.kernel_a4 import a4_quantize_activation
        return a4_quantize_activation(x.contiguous(), torch.tensor([gs], device=x.device, dtype=torch.float32))
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
              "finite_codes": bool(torch.isfinite(codes.float()).all()) if family != "e2m1" else True,
              "saturated_rows": saturation.nonzero().flatten().cpu().tolist(),
              "zero_rows": (x.float().abs().amax(1) == 0).nonzero().flatten().cpu().tolist(),
              "smallest_normal_unsaturated_j": min(candidates) if candidates else None}
    if not repeat:
        raise ValueError("quantizer calls gave different bytes")
    return result


def d_facts(gf, d):
    import numpy as np
    from experiments.numerics_rows.reference import bf16_array

    nonnegative, negative = gf >= 0, gf < 0
    signs = True
    for uf in (np.float32(1), np.float32(-1)):
        act = bf16_array((d[nonnegative] * uf).astype(np.float32))
        signs &= bool(np.all((act == 0) | (np.signbit(act) == np.signbit(uf))))
    return {"d_nondecreasing_0_L": bool(np.all(d[nonnegative][1:] >= d[nonnegative][:-1])),
            "d_nonpositive_negative_gf": bool(np.all(d[negative] <= 0)),
            "d_nonnegative_nonnegative_gf": bool(np.all(d[nonnegative] >= 0)),
            "nonzero_act_has_uf_sign_positive_gf": signs}


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
    return {"key": library["key"], "L_bits": struct.unpack("<I", struct.pack("<f", limit))[0],
            "dom_size": len(gf), "dom_path": str(source), "dom_sha256": sha256(source),
            "d_table_path": str(target), "d_table_sha256": sha256(target),
            **d_facts(gf, d),
            "library_so_sha256": library["library"]["sha256"], "probe_sha256": library["probe"]["sha256"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--image", choices=("G", "N"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--build", type=Path, required=True)
    p.add_argument("--artifact", type=Path, required=True)
    args = p.parse_args()
    check_source(args.source)
    check_runtime(args.image)
    sys.path.insert(0, str(args.source / "src"))
    import tessera
    import torch

    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    registration = register_native_ops(args.image)
    write_json(args.out / "native_ops.json", registration)
    built = json.loads(args.build.read_text())
    limit = artifact_limit(args.artifact)
    scales = artifact_scales(args.artifact) if args.image == "G" else {}
    records, boundaries = [], []
    for family in (("e4m3", "e2m1") if args.image == "G" else ("value",)):
        for gs in sorted(set(scales.values())) if family == "e2m1" else (1.0,):
            records.append(record_quantizer(family, gs, args.out, boundaries))
    write_json(args.out / "q1.json", {"native_ops": registration, "quantizers": records, "artifact_scales": scales,
                                      "tables": [probe_table(lib, args.out, limit) for lib in built["libraries"]],
                                      "boundaries": boundaries})


if __name__ == "__main__":
    main()
