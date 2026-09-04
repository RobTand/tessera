"""Read the #60 bracket's served compares and state the verdict zone.

Pure reader: it takes whichever ``kl_*.json`` the driver has written, prints
each arm's ``all.kl_lower_mean`` at full precision with its teacher, and
resolves the zones pre-registered in
``docs/measurements/tessera-dense4-gap-2026-09-03.md``.  It states the zone; it
does not choose one, and it refuses to name a verdict while a control is
missing, because a candidate without its own session's control is not a pair.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

COMPARATOR = "/mnt/shared/tessera-runs/ldlq-block/ts12_kl_nvfp4prod_vs_lina.json"
FALLBACK = "/home/rob/tessera-runs/stock/kl_nvfp4-prod.json"


def read(p: str) -> "dict | None":
    try:
        d = json.load(open(p))
    except Exception:
        return None
    return {"path": p, "kl": d["all"]["kl_lower_mean"],
            "confident": d["confident"]["kl_lower_mean"],
            "top1": d["all"]["top1_agree_pct"],
            "n": d["positions"],
            "teacher": Path(d["teacher_payload"]).name,
            "student": Path(d["student_payload"]).name}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/mnt/shared/tessera-runs/ldlq-block-serve")
    args = ap.parse_args()

    arms, other = {}, {}
    for p in sorted(glob.glob(f"{args.dir}/kl_*.json")):
        r = read(p)
        if not r:
            continue
        name = Path(p).stem[3:]
        # Only ``b<digits>`` names are bracket arms.  Anything else in this
        # directory is a different experiment that happens to live beside them
        # -- notably ``repro_ldlqH1``, which re-serves the *2026-09-02*
        # checkpoint and is not one of the three blocks.  Counting it as a
        # candidate would compare an arm against a control it never shared a
        # session or an encoder with.
        (arms if re.fullmatch(r"b32[ab]_\S+|b\d+", name) else other)[name] = r

    comp = read(COMPARATOR) or read(FALLBACK)
    if comp:
        print(f"comparator  {comp['kl']!r}  teacher={comp['teacher']} "
              f"n={comp['n']}  ({comp['path']})")
    for name, r in arms.items():
        print(f"{name:<24} {r['kl']!r}  conf={r['confident']!r} "
              f"top1={r['top1']!r}  teacher={r['teacher']}")
    for name, r in other.items():
        print(f"(not a bracket arm) {name}: {r['kl']!r} "
              f"student={r['student']}")

    # ``ldlq_block_serve_ab.sh`` writes the control twice per bracket, as
    # ``b32a_<cands>`` and ``b32b_<cands>``, and the candidates as bare
    # ``b8``/``b4``.  The two controls are the session's own error bar.
    controls = [r["kl"] for n, r in arms.items() if n.startswith("b32")]
    cands = {n: r["kl"] for n, r in arms.items() if not n.startswith("b32")}
    if not controls:
        print("\nNo control serve in this directory yet -- no verdict. "
              "A candidate without its own session's control is not a pair.")
        return
    print(f"\ncontrols: {controls}  spread="
          f"{max(controls) - min(controls)!r}")
    for n, k in cands.items():
        line = [f"{n}: {k!r}", f"vs control {k / min(controls):.4f}x"]
        if comp:
            line.append(f"vs comparator {k / comp['kl']:.4f}x")
            zone = ("A closed (<= comparator)" if k <= comp["kl"] else
                    "B narrowed, still open" if k < min(controls) else
                    "C closed the other way (>= control)")
            line.append(f"zone {zone}")
        print("  " + "  ".join(line))


if __name__ == "__main__":
    main()
