#!/usr/bin/env python3
"""Derive the TR3 scorer argv and headless peer argv as one engine.

Parses the scorer argv with the scorer's OWN argparse (measure replaced
by a capture) and derives the peer argv from PQ's own gold-engine
options with its own round-trip check. Refuses when the peer argv does
not reproduce the scorer kwargs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def derive(pq_root: str, scorer_argv: list[str]) -> dict:
    sys.path.insert(0, pq_root)
    sys.path.insert(0, str(Path(pq_root) / "tools"))
    import experiments.measure_glm_tr3_vllm as scorer
    from tools import gold_engine_options as geo

    captured: dict = {}
    real_measure, real_argv = scorer.measure, sys.argv
    scorer.measure = lambda args: captured.setdefault("args", args)
    sys.argv = ["measure_glm_tr3_vllm.py", *scorer_argv]
    try:
        scorer.main()
    finally:
        scorer.measure, sys.argv = real_measure, real_argv
    args = captured["args"]
    topology = geo.gold_engine_kwargs(args)
    kwargs = scorer.scorer_engine_kwargs(args, model=Path(args.model).resolve(), topology=topology)
    kwargs["model"] = str(kwargs["model"])
    peer = geo.headless_peer_argv(kwargs, node_rank=1)
    model, node_rank, back = geo.parse_headless_peer_argv(peer)
    if model != kwargs["model"] or node_rank != 1:
        raise ValueError("peer argv names a different model or node rank")
    local = geo._PEER_POSITIONAL_OR_LOCAL
    booleans = geo._PEER_BOOLEAN_SPELLING
    mismatches = []
    for key in sorted((set(kwargs) - local) | set(back)):
        want = kwargs.get(key)
        if key not in back:
            if not (want is None or (want is False and key in booleans and booleans[key][1] is None)):
                mismatches.append(key)
            continue
        got = back[key]
        if isinstance(want, bool):
            ok = got is want
        elif isinstance(want, dict):
            ok = got == want
        else:
            ok = want is not None and str(want) == got
        if not ok:
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"peer argv does not reproduce the scorer kwargs: {mismatches}")
    return {"scorer_argv": scorer_argv, "engine_kwargs": kwargs, "peer_argv": peer,
            "peer_roundtrip_equal": True, "output": args.output}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pq", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--argv-json", type=Path, default=None)
    ap.add_argument("scorer_argv", nargs=argparse.REMAINDER)
    ns = ap.parse_args()
    if ns.argv_json is not None:
        argv = json.loads(ns.argv_json.read_bytes())
    else:
        argv = ns.scorer_argv[1:] if ns.scorer_argv[:1] == ["--"] else ns.scorer_argv
    result = derive(ns.pq, argv)
    ns.out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: (v if k != "engine_kwargs" else "...") for k, v in result.items()},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
