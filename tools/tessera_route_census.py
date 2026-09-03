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
checkpoint declares for that module, that route's activation contract, and a
``(symbol, decoder)`` pair the route owns for the driven regime (the streamed
FP8 route reports the window-GEMV pair in the decode regime and the
materialised pair in batch, and the streamed BF16 route the same shape over
``torch.mm``) -- so the JSON it writes is a receipt only when
the run also passed.

usage::

    tessera_route_census.py <checkpoint-dir> <out.json> \
        [--expect-modules N] [--prompt-tokens 64] [--gpu-memory-utilization 0.3]

The two forwards it drives are the two regimes ``lane_eligibility`` declares.
The census calls them by the shape it drove (``prefill``, ``decode``) because
its receipt is keyed by those names and served receipts quote them; the
contract calls them ``batch`` and ``decode``.  One table maps the pair --
``tessera.serving.contract.CENSUS_PHASE_REGIMES`` -- the census resolves its
phase names through it and stamps the contract's word into every histogram
entry, and ``load_serving_contract`` refuses a contract whose declared regimes
are not exactly that table's values.  So a per-(family, regime) expectation can
join the two sides, and a rename or a third regime fails before the first model
load rather than at a per-module ``KeyError`` after two.

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


#: The regimes this tool can actually drive, in the contract's vocabulary: one
#: many-row forward and one one-row forward.  It is a statement about this
#: tool's two ``llm.generate`` calls, not about the contract -- which is why a
#: regime declared there and absent here is a refusal below rather than a
#: silently unobserved cell.
DRIVEN_REGIMES = ("batch", "decode")


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
    from tessera.serving import bf16_route, fp8_gemv, fp8_route, nvfp4_route
    from tessera.serving.contract import CENSUS_PHASE_REGIMES, load_serving_contract
    from tessera.serving.lane import TESSERA_MODE_ENV
    from tessera.serving.scheme import (
        ROUTES, TESSERA_BF16, TESSERA_FAMILIES, TESSERA_FP8, TESSERA_NVFP4)
    from tessera.serving.telemetry import DECODER_NATIVE_SPAN2, DECODER_TORCH_WINDOW

    # The executed A-side contract each route stamps on its layers: the value a
    # cell publishes, compared here against what the serve recorded.
    contract_for = {TESSERA_NVFP4: nvfp4_route.ACTIVATION_CONTRACT,
                    TESSERA_FP8: fp8_route.ACTIVATION_CONTRACT,
                    TESSERA_BF16: bf16_route.ACTIVATION_CONTRACT}
    # The decoder each route must have used.  The NVFP4 route's must be
    # the native span-2 kernel unless the operator explicitly accepted the
    # fallback.
    decoder_for = {TESSERA_NVFP4: DECODER_NATIVE_SPAN2, TESSERA_FP8: DECODER_TORCH_WINDOW,
                   TESSERA_BF16: DECODER_TORCH_WINDOW}
    # The GEMM each route invokes, off the route table rather than a literal
    # here: the two 4/8-bit routes call ``torch._scaled_mm`` and the 16-bit one
    # calls ``torch.mm`` (there is no scale to hand a scaled GEMM -- the row
    # scale is an epilogue), and a hardcoded symbol read that as a refusal on
    # every module of a route it had simply never been told about.
    symbol_for = {family: ROUTES[family]["gemm_symbol"] for family in TESSERA_FAMILIES}
    # The streamed FP8 route serves two launches: the window GEMV in the
    # decode regime, the materialised tile under ``_scaled_mm`` in batch (and
    # wherever the GEMV lane did not prepare).  The streamed BF16 route is the
    # same shape over ``torch.mm``: the window GEMV in the decode regime where
    # the lane prepared, the kernel-decoded tile above it, the torch decode
    # where it did not.  The pairs each regime may report live where the
    # dispatch lives (``fp8_gemv.census_expected``, ``bf16_route.
    # census_expected``), not in a second spelling here; every other family
    # reports one pair.
    fp8_expected = fp8_gemv.census_expected(compiled=args.compiled)
    bf16_expected = bf16_route.census_expected(compiled=args.compiled)

    def _expected(family, regime):
        if family == TESSERA_FP8:
            return fp8_expected[regime]
        if family == TESSERA_BF16:
            return bf16_expected[regime]
        return {(symbol_for[family], decoder_for[family])}
    missing = sorted(set(TESSERA_FAMILIES) - (set(contract_for) & set(decoder_for)))
    if missing:
        raise SystemExit(
            f"this census has no expectation for {missing}; a family the plugin serves and "
            "the census does not know would be counted as a mismatch on every module. Add "
            "its contract and decoder above rather than widening the comparison.")

    # ONE regime vocabulary (issue #61).  ``load_serving_contract`` refuses a
    # contract whose ``lane_eligibility.regimes`` are not exactly this table's
    # values, and the phase names below are resolved THROUGH the table rather
    # than written a second time here.  Both checks run before the first model
    # load (85-160 s each in tessera-bf16-route-served-2026-09-02.md), because
    # the one outcome a receipt tool must not have is failing at a per-module
    # lookup with two loaded models behind it.
    load_serving_contract()
    undrivable = sorted(set(CENSUS_PHASE_REGIMES.values()) - set(DRIVEN_REGIMES))
    if undrivable:
        raise SystemExit(
            f"the contract declares the regime(s) {undrivable}, which this census does not "
            f"drive (it drives {list(DRIVEN_REGIMES)}). A declared regime no forward exercises "
            "is a cell nothing observes; add the forward here rather than widening the table.")
    phase_of = {regime: phase for phase, regime in CENSUS_PHASE_REGIMES.items()}
    if len(phase_of) != len(CENSUS_PHASE_REGIMES):
        raise SystemExit(
            f"CENSUS_PHASE_REGIMES maps two phases onto one regime ({dict(CENSUS_PHASE_REGIMES)}); "
            "the join is then ambiguous in the direction this tool reads it.")
    batch_phase, decode_phase = phase_of["batch"], phase_of["decode"]

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
    phases[batch_phase] = llm.apply_model(census)[0]
    # Decode steps follow; the last forward is one row wide.
    outs = llm.generate([prompt], SamplingParams(max_tokens=8, temperature=0.0))
    phases[decode_phase] = llm.apply_model(census)[0]
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
            # The contract's word for the shape this phase drove, so a
            # per-(family, regime) expectation joins the receipt to a cell
            # without either side guessing the other's vocabulary.
            "regime": CENSUS_PHASE_REGIMES[phase],
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
            if r["policy"] != f"{family}:{mode}":
                problems.append(f"{phase}: {name} policy={r['policy']!r} != declared {family}:{mode}")
            # The (symbol, decoder) pair, not each half alone: the streamed FP8
            # route reports the GEMV pair in the decode regime and the
            # materialised pair in batch (``fp8_gemv.census_expected`` owns the
            # sets), and a half-wise comparison would read either half as a
            # refusal on every module of the other regime.
            want = _expected(family, CENSUS_PHASE_REGIMES[phase])
            if ((r["symbol"], r.get("decoder")) not in want
                    and not (args.allow_fallback_decoder and r["symbol"] == symbol_for[family])):
                problems.append(
                    f"{phase}: {name} (symbol, decoder)={(r['symbol'], r.get('decoder'))!r} "
                    f"not in {sorted(want)!r}; without --allow-fallback-decoder a serve must "
                    "report a pair its route owns")
        missing = sorted(set(declared) - set(tess))
        if missing:
            problems.append(
                f"{phase}: {len(missing)} declared Tessera modules report no route, e.g. {missing[:3]}")
        if args.expect_modules is not None and len(tess) != args.expect_modules:
            problems.append(
                f"{phase}: {len(tess)} Tessera modules, the checkpoint declares {args.expect_modules}")
    if phases[batch_phase] and phases[decode_phase]:
        shapes = [(phases[batch_phase][n].get("shape", ""), phases[decode_phase][n].get("shape", ""))
                  for n in phases[batch_phase] if n in phases[decode_phase]]
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
            problems.append(
                f"{batch_phase} and {decode_phase} records carry the same shape; only one "
                "shape was exercised")

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
