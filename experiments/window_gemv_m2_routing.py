"""Does routing M=2 to the MT=4 kernel pay? (#59)

`docs/measurements/tessera-window-gemv-2026-09-02.md` §11/§12 record "routing
M=2 (and M=3) to the MT=4 kernel by padding is a free ~8%", from the contended
per-token table in §5.  The same document's quiet-box addendum re-took M = 1,
2, 4, 8 on an idle box, and its raw JSON says the opposite in the aggregate.

This script re-derives both readings from that JSON so the claim is checkable
rather than quoted.  It reads only the numbers the receipt already cites; it
runs no GPU work.

The arithmetic step, stated: padding M=2 to MT=4 runs the launch M=4 already
runs -- same ``mt`` from ``_m_tile``, same ``items_for(mt)``, same ``rpl = 8``,
two of the four ``x`` rows zero -- so its cost is the M=4 column.  That is not
measured here; what is measured is the M=4 column.

    python experiments/window_gemv_m2_routing.py [path-to-json]
"""
import json
import sys

#: The addendum's own raw file, named in the receipt.
DEFAULT = ("/home/rob/tessera-runs/gemv/quiet_sparky.json/"
           "bench_gemv_quietbox_20260902-185032.json")

#: The contended §5 per-token table, `fused_kernel` us, for the same models.
#: Transcribed from the receipt so the two readings sit side by side.
CONTENDED = {"Qwen3-4B": {1: 9570, 2: 12663, 4: 11633, 8: 16838},
             "Qwen3-4B-fused": {1: 9033, 2: 18293, 4: 17633, 8: 22845},
             "Qwen3-0.6B": {1: 1979, 2: 2946, 4: 2517, 8: 4192}}


def main(path: str) -> int:
    d = json.load(open(path))
    procs = {m["fused_kernel"]["cuda_procs_max"]
             for e in d["gemv"].values() for m in e["M"].values()}
    print(f"{path}\n  commit {d['commit']}  plan {d['args']['plan']!r}  "
          f"cuda_procs_max {sorted(procs)}")
    if procs != {1}:
        print("  WARNING: not a quiet box; every ratio below is contaminated")

    print("\nPer token (us, fused_kernel). 'rule' = padding M=2 to MT=4, i.e. the M=4 cost.")
    print(f"{'model':18s} {'M=1':>8s} {'M=2':>8s} {'M=4':>8s} {'M=8':>8s}  {'rule':>8s}  contended")
    bad = []
    for model, per_m in d["totals"].items():
        us = {int(k): v["fused_kernel"] for k, v in per_m.items()}
        rule = us[4] / us[2] - 1.0
        c = CONTENDED.get(model)
        cs = f"{c[4] / c[2] - 1.0:+7.1%}" if c else "  n/a  "
        print(f"{model:18s} {us[1]:8.0f} {us[2]:8.0f} {us[4]:8.0f} {us[8]:8.0f}  "
              f"{rule:+7.1%}  {cs}")
        if rule > 0:
            bad.append(model)

    print("\nPer shape (M=4 / M=2 kernel us; < 1 means the MT=4 build wins)")
    print(f"{'shape':14s} {'rows':>6s} {'M=2':>8s} {'M=4':>8s} {'M4/M2':>7s}")
    for shape, e in sorted(d["gemv"].items(), key=lambda kv: kv[1]["rows"]):
        us = {int(k): v["fused_kernel"]["us"] for k, v in e["M"].items()}
        if 2 not in us or 4 not in us:
            continue
        print(f"{shape:14s} {e['rows']:6d} {us[2]:8.1f} {us[4]:8.1f} {us[4] / us[2]:7.3f}")

    print(f"\nThe blanket M-only rule is a per-token REGRESSION on: {bad or 'none'}")
    print("Per shape it is not uniform, so a per-shape variant stays open -- with a")
    print("threshold derived from occupancy and re-measured, not read off this table.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT))
