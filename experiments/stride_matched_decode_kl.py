#!/usr/bin/env python3
"""Exact KL on the served stride rows, from the full-vocabulary payloads.

The served decode dump scores one row per 16-token stride prefix per chunk
(256 rows); the exact decode payloads hold all 4088 rows. This tool reads the
served row addresses out of the served payload's own regime record, scores
exactly those rows from both full-vocabulary payloads with the same blockwise
math as ``full_vocab_kl_dump compare``, and prints one JSON object. It runs
CPU-only through PrismaBuild; it serves nothing and trains nothing.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_dump():
    here = Path(__file__).resolve().parent / "full_vocab_kl_dump.py"
    spec = importlib.util.spec_from_file_location("_fv_kl_stride", here)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--served-meta", required=True,
                    help="served decode .meta.json carrying prefix_lengths")
    ap.add_argument("--teacher", required=True,
                    help="exact decode teacher .meta.json")
    ap.add_argument("--student", required=True,
                    help="exact decode student .meta.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    fv = _load_dump()
    served = json.loads(Path(args.served_meta).read_text())
    teacher_meta, t_arr = fv.read_full_vocab_payload(args.teacher)
    student_meta, s_arr = fv.read_full_vocab_payload(args.student)
    rows = fv.decode_subset_rows(served, teacher_meta, student_meta)
    kl = np.empty(rows.size, dtype=np.float64)
    agree = np.empty(rows.size, dtype=bool)
    step = 256
    for lo in range(0, rows.size, step):
        idx = rows[lo:lo + step]
        reduced, tb, sb, _ = fv.kl_rows(t_arr[idx], s_arr[idx])
        m = tb.shape[0]
        kl[lo:lo + m] = reduced
        agree[lo:lo + m] = tb.argmax(axis=1) == sb.argmax(axis=1)
    result = {
        "schema": "tessera.stride_matched_exact_kl/1",
        "rows": int(rows.size),
        "row_index": [int(r) for r in rows],
        "teacher_payload": str(Path(args.teacher).resolve()),
        "student_payload": str(Path(args.student).resolve()),
        "served_meta": str(Path(args.served_meta).resolve()),
        "kl_mean": float(kl.mean()),
        "kl_p99": fv.kl_p99(kl),
        "kl_max": float(kl.max()),
        "top1_agree_pct": float(100.0 * agree.mean()),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
