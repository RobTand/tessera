"""Decode-once E4M3 dense prefill against BF16 and today's T-8 lane (tessera#931).

Per module (GLM-5.3-Flash TP2 rank shapes) and M, three lanes over the same
encoded wire, each timed as a CUDA-graph replay (median of ``--iters``):

* ``bf16``: ``F.linear`` on the source weight (what the serve runs today);
* ``t8``: the prepared module's served lane (``module.apply``; fused unless the
  predicate keeps Triton) on the prequantised activation;
* ``dec``: ``e4m3_prefill.prefill_apply`` over the module's decode-once copy.

``quant`` times vLLM's native per-token quantiser alone; both T-8 lanes need
it, so add it to either for the end-to-end cost.  The decode itself is timed
once per module (load-time cost) and its bytes recorded.  A numerics line per
(module, M) gives the decoded lane's worst ratio to the derived bound of the
fp64 definition and to twice it against the served lane.

Usage: bench_e4m3_prefill.py --out DIR [--modules kda_in,o_proj,q_b_proj,mla_o_proj]
       [--q256 1024] [--ms 16,64,256,512,1024,2048,4096,8192]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "tests"))
from bench_dense_module import MODULES, encode_module, graph_time  # noqa: E402

SHAPES = dict(MODULES)
SHAPES["mla_o_proj"] = ([("o_proj", 4096)], 8192)


def med(samples):
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--modules", default="kda_in,o_proj,q_b_proj,mla_o_proj")
    ap.add_argument("--q256", default="1024")
    ap.add_argument("--ms", default="16,64,256,512,1024,2048,4096,8192")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--artifact", default=None, help="accepted for bench_t8r.sh; unused")
    args = ap.parse_args()
    import fused_bound as fb
    from tessera.serving.e4m3_prefill import decode_e4m3, prefill_apply
    from tessera.serving.native_ops import native_fp8_quant

    os.makedirs(args.out, exist_ok=True)
    meta = {"host": os.environ.get("HOST_NAME"), "head": os.environ.get("TESSERA_HEAD"),
            "state": os.environ.get("TESSERA_STATE"), "image": os.environ.get("ORACLE_IMAGE"),
            "pb_action": os.environ.get("PB_ACTION_KEY"), "torch": torch.__version__,
            "device": torch.cuda.get_device_name(), "weights": "seeded Gaussian",
            "statistic": "median CUDA-graph replay", "start_unix": time.time()}
    print(json.dumps({"meta": meta}), flush=True)
    rows_out = []
    ms = [int(v) for v in args.ms.split(",")]
    for name in args.modules.split(","):
        roles, cols = SHAPES[name]
        for q in (int(v) for v in args.q256.split(",")):
            prepare, wire_bytes, enc_s, w_src = encode_module(roles, cols, q, seed=0)
            module = prepare(True)
            torch.cuda.synchronize()
            t0 = time.time()
            dec = decode_e4m3(module)
            torch.cuda.synchronize()
            head = {"module": name, "q256": q, "rows": int(module.rows), "cols": cols,
                    "lane": module.lane, "wire_bytes": wire_bytes, "decoded_bytes": dec.nbytes,
                    "decode_s": time.time() - t0, "encode_s": enc_s}
            print(json.dumps(head), flush=True)
            ref_w64 = dec.weight.double() * dec.scale.double()[:, None]
            for m in ms:
                g = torch.Generator(device="cuda").manual_seed(m)
                x = torch.randn(m, cols, device="cuda", generator=g).bfloat16()
                xq, a = native_fp8_quant(x)
                xq, a = xq.contiguous(), a.reshape(-1).contiguous().float()
                row = dict(head, m=m)
                row["bf16_ms"] = med(graph_time(lambda: torch.nn.functional.linear(x, w_src), args.warmup, args.iters))
                row["t8_ms"] = med(graph_time(lambda: module.apply(xq, a), args.warmup, args.iters))
                row["dec_ms"] = med(graph_time(lambda: prefill_apply(dec, xq, a), args.warmup, args.iters))
                row["quant_ms"] = med(graph_time(lambda: native_fp8_quant(x), args.warmup, args.iters))
                got, served = prefill_apply(dec, xq, a), module.apply(xq, a)
                if m <= 2048:
                    r, bound = fb.dense_bound("e4m3", xq.double() * a.double()[:, None], ref_w64, cols, s=cols)
                    d = (got.double() - r).abs() / bound
                    row["dec_vs_bound"] = float(d.max())
                    row["dec_vs_served_vs_2bound"] = float(((got.double() - served.double()).abs() / (2 * bound)).max())
                row["bitwise_equal_frac"] = float((got == served).float().mean())
                row["speedup_vs_bf16"] = row["bf16_ms"] / row["dec_ms"]
                row["speedup_vs_t8"] = row["t8_ms"] / row["dec_ms"]
                rows_out.append(row)
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                                  if k not in head or k == "module"}), flush=True)
            del module, dec, w_src
            torch.cuda.empty_cache()
    meta["end_unix"] = time.time()
    with open(os.path.join(args.out, "e4m3_prefill.json"), "w") as f:
        json.dump({"meta": meta, "rows": rows_out}, f, indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
