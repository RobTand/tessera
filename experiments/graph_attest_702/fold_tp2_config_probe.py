#!/usr/bin/env python3
"""Validate a vLLM engine config for the fold arm serve without launching.

Builds EngineArgs mirroring the profile-serve flags and runs
create_engine_config only: distributed init, weight load and GPU use
never happen. Used to isolate serve-flag triggers (MTP, graphs) from
artifact incompatibility. Runs inside the pinned image with tessera
installed from the snapshot.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback


def build(variant: str, artifact: str):
    from vllm.engine.arg_utils import EngineArgs
    kw = dict(model=artifact, trust_remote_code=True, max_model_len=8448,
              max_num_batched_tokens=2048, max_num_seqs=1,
              enable_chunked_prefill=True, enable_prefix_caching=False,
              kv_cache_dtype="fp8_ds_mla", kv_cache_memory_bytes=2147483648,
              gpu_memory_utilization=0.5, tensor_parallel_size=2,
              distributed_executor_backend="mp", dtype="bfloat16",
              language_model_only=True, max_logprobs=20,
              compilation_config={"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"},
              enforce_eager=False)
    if variant == "mtp":
        kw["speculative_config"] = {"method": "mtp", "num_speculative_tokens": 1,
                                    "draft_tensor_parallel_size": 2,
                                    "moe_backend": "triton"}
    return EngineArgs(**kw)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--variants", nargs="+", default=["mtp", "nospec"])
    args = ap.parse_args()
    rc = 0
    for variant in args.variants:
        try:
            build(variant, args.artifact).create_engine_config()
            print(f"variant {variant}: CONFIG OK", flush=True)
        except Exception as exc:
            rc = 1
            print(f"variant {variant}: REFUSED {type(exc).__name__}: {str(exc)[:300]}",
                  flush=True)
            traceback.print_exc(limit=3)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
