#!/usr/bin/env python3
"""The prefill regime restricted to the decode regime's positions (issue #102).

A decode-regime KL differs from a prefill-regime KL for two reasons at once:
the scored forward is M=1 instead of 512 rows, and the position set is a
stride subsample rather than all 511 positions of each chunk. Quoting the two
regime numbers side by side confounds them.

This script removes the second reason. The decode dump records the prefix
length L of every position it scored; the distribution at prefix L is
conditioned on exactly the same tokens as prefill position L, so it has a
counterpart row in a prefill dump of the same corpus. Restricting the prefill
pair to those rows leaves ONE difference between the two numbers: which
forward produced them.

The KL itself comes from ``kl_estimator.lumped_kl`` -- the same estimator
``kl_tool compare`` uses, imported rather than re-implemented, because a
second copy of an estimator is a second answer.

usage::

    decode_regime_subset.py --teacher-prefill T.npz --student-prefill S.npz \
        --decode-student D.npz [--seqlen 512] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

KL_TOOL_DIR = Path("/home/rob/dq-runs")
sys.path.insert(0, str(KL_TOOL_DIR))
from kl_estimator import DEFAULT_STUDENT_PROB_FLOOR, lumped_kl  # noqa: E402
import kl_tool  # noqa: E402


def prefill_rows(decode_meta, *, seqlen):
    """Prefill row indices for the positions a decode dump scored.

    A prefill dump drops position 0 of every chunk (it predicts nothing from
    inside the chunk), so chunk ``c``'s position ``p`` -- 1-based, predicting
    the token at index ``p`` -- is row ``c * (seqlen - 1) + (p - 1)``. A decode
    position of prefix length ``L`` predicts the token at index ``L``, so
    ``p = L``.
    """
    regime = decode_meta["regime"]
    rows = []
    for c, prefixes in enumerate(regime["prefix_lengths"]):
        for L in prefixes:
            rows.append(c * (seqlen - 1) + (L - 1))
    return np.array(rows, dtype=np.int64)


def kl_over(t_ids, t_lps, s_ids, s_lps, rows, floor):
    per = []
    agree = 0
    for i in rows:
        tmap = {int(t): float(v) for t, v in zip(t_ids[i], t_lps[i])
                if int(t) != kl_tool.PAD_ID and np.isfinite(v)}
        smap = {int(t): float(v) for t, v in zip(s_ids[i], s_lps[i])
                if int(t) != kl_tool.PAD_ID and np.isfinite(v)}
        if not tmap or not smap:
            continue
        shared = [t for t in tmap if t in smap]
        stats = lumped_kl(np.exp(np.array([tmap[t] for t in shared])),
                          np.exp(np.array([smap[t] for t in shared])),
                          student_prob_floor=floor, n_tail_tokens=None)
        per.append(stats["kl_lower"])
        if max(tmap, key=tmap.get) == max(smap, key=smap.get):
            agree += 1
    per = np.array(per, dtype=np.float64)
    return {"positions": int(per.size), "kl_lower_mean": float(per.mean()),
            "kl_lower_max": float(per.max()),
            "top1_agree_pct": 100.0 * agree / max(per.size, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-prefill", required=True)
    ap.add_argument("--student-prefill", required=True)
    ap.add_argument("--decode-student", required=True)
    ap.add_argument("--seqlen", type=int, default=512)
    ap.add_argument("--floor", type=float, default=DEFAULT_STUDENT_PROB_FLOOR)
    ap.add_argument("--json")
    args = ap.parse_args(argv)

    d_meta, _d_ids, _d_lps = kl_tool.read_payload(args.decode_student)
    if kl_tool.payload_regime(d_meta) != "decode":
        raise SystemExit("--decode-student is not a decode-regime payload")
    t_meta, t_ids, t_lps = kl_tool.read_payload(args.teacher_prefill)
    s_meta, s_ids, s_lps = kl_tool.read_payload(args.student_prefill)
    for name, meta in (("teacher", t_meta), ("student", s_meta)):
        if kl_tool.payload_regime(meta) != "prefill":
            raise SystemExit(f"--{name}-prefill is not a prefill-regime payload")

    rows = prefill_rows(d_meta, seqlen=args.seqlen)
    if rows.max() >= t_ids.shape[0]:
        raise SystemExit(
            f"row {int(rows.max())} is past the prefill payload's "
            f"{t_ids.shape[0]} positions; --seqlen {args.seqlen} does not "
            "describe this corpus")
    result = {
        "schema": "tessera.decode_regime_subset/1",
        "decode_payload": str(Path(args.decode_student).resolve()),
        "teacher_prefill": str(Path(args.teacher_prefill).resolve()),
        "student_prefill": str(Path(args.student_prefill).resolve()),
        "note": ("the prefill regime restricted to the decode regime's "
                 "positions: the only remaining difference between this and "
                 "the decode-regime compare is which forward ran"),
        "positions_requested": int(rows.size),
        "all_prefill_positions": int(t_ids.shape[0]),
        "prefill_on_decode_positions": kl_over(t_ids, t_lps, s_ids, s_lps,
                                               rows, args.floor),
        "prefill_on_all_positions": kl_over(t_ids, t_lps, s_ids, s_lps,
                                            np.arange(t_ids.shape[0]),
                                            args.floor),
    }
    print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
