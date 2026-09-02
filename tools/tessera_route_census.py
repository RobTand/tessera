#!/usr/bin/env python3
"""Census of the dispatch routes a served Tessera checkpoint executes.

The ``lane_eligibility`` cells in ``tessera/serving/runtime_contract.json``
state which route the plugin *executes*.  This script is the serve-side
observation behind such a cell: it loads a Tessera checkpoint through
``vllm.LLM`` in one process, runs a prefill-shaped forward and a decode-shaped
forward, then reads the route record every Tessera module wrote
(``tessera.serving.telemetry.read_route``) from inside the worker.  No log line
is parsed: the record is the same scalars the route tests assert on, read from
the same objects the serve dispatched through.

It exits non-zero unless every Tessera module, in both shapes, reports
``state == "served"``, a ``<family>:<mode>`` policy equal to the family the
checkpoint declares for that module, that route's activation contract, the
expected GEMM symbol and the native decoder -- so the JSON it writes is a
receipt only when the run also passed.

usage::

    tessera_route_census.py <checkpoint-dir> <out.json> \
        [--expect-modules N] [--prompt-tokens 64] [--gpu-memory-utilization 0.3]

Run it inside the serving image with the plugin installed (the same container
the KL dumps ran in); ``TESSERA_SERVE_MODE`` selects the residency exactly as
it does for ``vllm serve``.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import platform
import subprocess
import sys
import time


def census(model):
    """Runs inside the worker: every module carrying a route record."""
    from tessera.serving.telemetry import read_route
    out = {}
    for name, mod in model.named_modules():
        rec = read_route(mod)
        if rec is not None:
            out[name] = rec
    return out


def _git_head(path):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance is best effort, the check is not
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--expect-modules", type=int, default=None,
                    help="number of Tessera modules the checkpoint declares")
    ap.add_argument("--prompt-tokens", type=int, default=64)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--compiled", action="store_true",
                    help="load with enforce_eager=False (vLLM's default compiled forward + CUDA "
                         "graphs) instead of eager; the route records then carry M='*' because "
                         "the record is written from the trace, and a route that cannot be traced "
                         "fails here with its own traceback instead of an engine-start refusal")
    ap.add_argument("--allow-fallback-decoder", action="store_true",
                    help="accept a module decoded by the pure-torch fallback instead of the "
                         "native span-2 kernel; without it a fallback serve REFUSES, because a "
                         "receipt must not claim the native route for bytes another decoder made")
    ap.add_argument("--tessera-commit", default=None,
                    help="the host's `git rev-parse HEAD` for the Tessera checkout under test; "
                         "inside a container a worktree's .git pointer resolves nowhere and the "
                         "receipt would carry None")
    args = ap.parse_args()

    # The census function must run in the process that holds the model.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch
    import vllm
    from vllm import LLM, SamplingParams

    import tessera
    import tessera.serving as serving
    from tessera.serving import fp8_route, nvfp4_route
    from tessera.serving.lane import TESSERA_MODE_ENV
    from tessera.serving.scheme import TESSERA_FAMILIES, TESSERA_FP8, TESSERA_NVFP4
    from tessera.serving.telemetry import DECODER_NATIVE_SPAN2, DECODER_TORCH_WINDOW

    # The executed A-side contract each route stamps on its layers: the value a
    # cell publishes, compared here against what the serve recorded.
    contract_for = {TESSERA_NVFP4: nvfp4_route.ACTIVATION_CONTRACT,
                    TESSERA_FP8: fp8_route.ACTIVATION_CONTRACT}
    # The decoder each route must have used.  The FP8 route's decoder IS pure
    # torch (the packed-window reader); the NVFP4 route's must be the native
    # span-2 kernel unless the operator explicitly accepted the fallback.
    decoder_for = {TESSERA_NVFP4: DECODER_NATIVE_SPAN2, TESSERA_FP8: DECODER_TORCH_WINDOW}

    with open(os.path.join(args.model, "config.json")) as fh:
        cfg = json.load(fh)
    qc = cfg.get("quantization_config", {})
    groups = qc.get("config_groups", {})
    tessera_groups = {k: g for k, g in groups.items()
                      if g.get("scheme", {}).get("family") in TESSERA_FAMILIES}
    # Which family the checkpoint declares for each module: the route the serve
    # must have taken, module by module (a mixed checkpoint has both).
    declared = {t: g["scheme"]["family"] for g in tessera_groups.values() for t in g.get("targets", [])}

    t0 = time.time()
    llm = LLM(model=args.model, enforce_eager=not args.compiled, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
    tok = llm.get_tokenizer()
    text = ("The receipt names the route the serve took, for every module, "
            "in both the prefill and the decode shape. ") * 20
    ids = tok.encode(text, add_special_tokens=False)[: args.prompt_tokens]
    prompt = {"prompt_token_ids": ids}

    phases = {}
    # One forward over M = len(ids) rows, sample one token, stop.
    outs = llm.generate([prompt], SamplingParams(max_tokens=1, temperature=0.0))
    phases["prefill"] = llm.apply_model(census)[0]
    # Decode steps follow; the last forward is one row wide.
    outs = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))
    phases["decode"] = llm.apply_model(census)[0]
    generated = outs[0].outputs[0].text

    mode = os.environ.get(TESSERA_MODE_ENV, "")
    prefixes = tuple(f"{family}:" for family in TESSERA_FAMILIES)
    problems = []
    histogram = {}
    for phase, recs in phases.items():
        tess = {n: r for n, r in recs.items() if str(r.get("policy", "")).startswith(prefixes)}
        other = {n: r for n, r in recs.items() if n not in tess}
        h = collections.Counter(
            (r["policy"], r["symbol"], r["contract"], r["state"], r.get("kind"), r.get("decoder"))
            for r in tess.values())
        by_family = collections.Counter(str(r["policy"]).split(":")[0] for r in tess.values())
        histogram[phase] = {
            "tessera_modules": len(tess),
            "tessera_modules_by_family": dict(sorted(by_family.items())),
            "other_route_modules": len(other),
            "routes": [dict(policy=k[0], symbol=k[1], contract=k[2], state=k[3], kind=k[4],
                            decoder=k[5], modules=v)
                       for k, v in sorted(h.items(), key=lambda kv: tuple(map(str, kv[0])))],
            "shapes": sorted({str(r.get("shape")) for r in tess.values()}),
        }
        if not tess:
            problems.append(f"{phase}: no module reports a Tessera route")
        for name, r in tess.items():
            family = declared.get(name)
            if family is None:
                problems.append(
                    f"{phase}: {name} took a Tessera route but the checkpoint declares none for it")
                continue
            if r["state"] != "served":
                problems.append(f"{phase}: {name} state={r['state']!r} reason={r.get('reason')!r}")
            if r["contract"] != contract_for[family]:
                problems.append(f"{phase}: {name} contract={r['contract']!r} != {contract_for[family]!r}")
            if r["symbol"] != "torch._scaled_mm":
                problems.append(f"{phase}: {name} symbol={r['symbol']!r}")
            if r["policy"] != f"{family}:{mode}":
                problems.append(f"{phase}: {name} policy={r['policy']!r} != declared {family}:{mode}")
            if r.get("decoder") != decoder_for[family] and not args.allow_fallback_decoder:
                problems.append(
                    f"{phase}: {name} decoder={r.get('decoder')!r} != {decoder_for[family]!r}; the "
                    "native route was not taken (pass --allow-fallback-decoder to record that "
                    "deliberately)")
        missing = sorted(set(declared) - set(tess))
        if missing:
            problems.append(
                f"{phase}: {len(missing)} declared Tessera modules report no route, e.g. {missing[:3]}")
        if args.expect_modules is not None and len(tess) != args.expect_modules:
            problems.append(
                f"{phase}: {len(tess)} Tessera modules, the checkpoint declares {args.expect_modules}")
    if phases["prefill"] and phases["decode"]:
        shapes = [(phases["prefill"][n].get("shape", ""), phases["decode"][n].get("shape", ""))
                  for n in phases["prefill"] if n in phases["decode"]]
        if args.compiled:
            # A compiled forward writes the record from the trace, where the
            # token dimension is symbolic: ``route_shape`` spells it ``M*`` and
            # the record cannot tell the regimes apart.  What the census can
            # attest here is dispatch (every module reached its route in both
            # steps) and the polymorphic form itself; a concrete M in a compiled
            # record would mean the route specialised the batch.
            bad = [p for p, d in shapes if not (p.startswith("M*:") and d.startswith("M*:"))]
            if bad:
                problems.append(f"compiled records must be shape-polymorphic (M*); got {bad[:3]}")
        elif all(p == d for p, d in shapes):
            problems.append("prefill and decode records carry the same shape; only one shape was exercised")

    receipt = {
        "schema": "tessera.serving.route_census/1",
        "checkpoint": os.path.abspath(args.model),
        "quant_method": qc.get("quant_method"),
        "compiled": bool(args.compiled),
        "tessera_config_groups": len(tessera_groups),
        "prompt_tokens": len(ids),
        "generated_text": generated,
        "declared_families": dict(sorted(collections.Counter(declared.values()).items())),
        "env": {TESSERA_MODE_ENV: mode or None,
                "VLLM_DISABLED_KERNELS": os.environ.get("VLLM_DISABLED_KERNELS")},
        "versions": {"vllm": vllm.__version__, "torch": torch.__version__,
                     "tessera": getattr(tessera, "__version__", None),
                     "tessera_serving": getattr(serving, "__version__", None),
                     "tessera_commit": args.tessera_commit or _git_head(
                         os.path.dirname(os.path.dirname(os.path.dirname(
                             os.path.abspath(tessera.__file__))))),
                     "python": platform.python_version()},
        "device": {"name": torch.cuda.get_device_name(0),
                   "capability": list(torch.cuda.get_device_capability(0))},
        "elapsed_s": round(time.time() - t0, 1),
        "histogram": histogram,
        "records": phases,
        "problems": problems,
        "verdict": "served" if not problems else "REFUSED",
    }
    with open(args.out, "w") as fh:
        json.dump(receipt, fh, indent=1, sort_keys=True)
    print(json.dumps({k: receipt[k] for k in ("verdict", "histogram", "env", "device", "elapsed_s")},
                     indent=1))
    for p in problems:
        print("PROBLEM:", p)
    print(f"-> {args.out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
