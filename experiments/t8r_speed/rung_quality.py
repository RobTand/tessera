"""CPU weight-space screen of actual sampled GLM experts; not served KL."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from safetensors import safe_open
from tessera.control import grid_for_name, unit_wire_bits
from tessera.export import encode_linear
from tessera.unit_artifact import read_unit_artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cases", default=",".join(str(q) for q in range(768, 1153)))
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    model = Path(args.model)
    mapping = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    samples = []
    # A fixed actual expert tile for each projection. All rungs share these exact
    # BF16 source bytes and codec defaults; no random Gaussian quality surrogate.
    for role in ("gate", "up", "down"):
        name = f"model.language_model.layers.3.mlp.experts.0.{role}_proj.weight"
        with safe_open(str(model / mapping[name]), framework="pt", device="cpu") as h:
            source = h.get_slice(name)
            shape = source.get_shape()
            weight = source[:32, :256].contiguous()
        expected = (4096, 2048) if role == "down" else (2048, 4096)
        assert tuple(shape) == expected, (name, shape, expected)
        source_sha = hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest()
        samples.append((weight, {"tensor": name, "file": mapping[name], "source_shape": shape,
                                "slice": [[0,32],[0,256]], "source_sha256": source_sha,
                                "source_squared_norm": float(weight.double().square().sum())}))
    if args.preflight:
        print(json.dumps({"status":"passed", "samples":[m for _,m in samples]}), flush=True)
        return
    grid = grid_for_name("E4M3")
    result = {"schema":"tessera.rung_quality.v1", "source_kind":"actual_sampled_expert_weights",
              "device":"cpu", "model":str(model), "sample_selection":"layer 3, expert 0; first 32 rows and 256 columns of gate/up/down",
              "objective":"unweighted weight-space relative SSE; not served KL or promotion",
              "codec":"tessera.export.encode_linear defaults; read_unit_artifact; exact accountant identity",
              "rungs":{}, "start_unix":time.time()}
    path = Path(args.out)
    path.parent.mkdir(parents=True,exist_ok=True)
    for q in (int(s.removeprefix("q")) for s in args.cases.split(",")):
        start = time.time()
        row = {"measurement_status":"measured", "source_kind":result["source_kind"], "device":"cpu", "samples":[], "anomaly_flags":[]}
        try:
            for weight, meta in samples:
                unit = encode_linear(weight,grid=grid,q256=q)
                decoded = read_unit_artifact(unit.blob).double()
                assert bool(torch.isfinite(decoded).all()), "nonfinite decoded weights"
                bits = unit_wire_bits(grid,q,32,256)
                assert bits == unit.exact_bytes * 8, (q,bits,unit.exact_bytes)
                sse = float((decoded-weight.double()).square().sum())
                row["samples"].append({**meta,"relative_sse":sse/meta["source_squared_norm"],"squared_error":sse,
                                       "exact_bytes":unit.exact_bytes,"accounted_bits":{"numerator":bits.numerator,"denominator":bits.denominator}})
        except Exception as exc:
            row["measurement_status"]="failed"
            row["error"]=repr(exc)
            row["anomaly_flags"]=["quality_screen_failed"]
        result["rungs"][str(q)]=row
        row["seconds"]=time.time()-start
        temporary=path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result,indent=1,allow_nan=False))
        temporary.replace(path)
        print(json.dumps({"q256":q,"status":row["measurement_status"],"seconds":row["seconds"]}),flush=True)
    result["end_unix"]=time.time()
    # Raw adjacent-rung ratios are evidence, not an arbitrary anomaly threshold.
    for q in sorted(int(k) for k in result["rungs"]):
        row=result["rungs"][str(q)]
        high=result["rungs"].get(str(q+1))
        if row["measurement_status"]=="measured" and high and high["measurement_status"]=="measured":
            row["adjacent_higher_raw_error_ratios"]=[h["relative_sse"]/l["relative_sse"] if l["relative_sse"] else None for l,h in zip(row["samples"],high["samples"])]
    path.write_text(json.dumps(result,indent=1,allow_nan=False))


if __name__=="__main__":
    main()
