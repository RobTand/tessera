"""Tables from t4_code_compare JSONs: per-class geometric means at matched bytes.

    python3 experiments/t4_code/summarize.py DIR [DIR ...] > table.md

Pairs each TCQ rung with the window rung at the same bytes (q256 + 64) and
reports window/TCQ ratios per leg; plain Python, no torch.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

PAIRS = [(128, 192), (256, 320), (384, 448), (512, 576), (640, 704), (768, 832), (896, 960)]
LEGS = ("wt", "out", "a4s", "a4d", "a4s_ref")


def klass(tag: str) -> str:
    if ".e0." in tag:
        return "routed expert"
    if "attn" in tag:
        return "attention"
    if "shared" in tag:
        return "shared expert"
    return "dense MLP"


def gmean(v):
    v = [x for x in v if x is not None and x > 0]
    return math.exp(sum(math.log(x) for x in v) / len(v)) if v else None


def main():
    tensors = {}
    meta = []
    for d in sys.argv[1:]:
        for p in sorted(Path(d).rglob("t4_code_*.json")):
            j = json.load(open(p))
            meta.append((str(p), j.get("tessera_head"), j.get("encoder_digest"), j.get("host")))
            tensors.update(j["tensors"])
    classes = {}
    for tag, r in tensors.items():
        classes.setdefault(klass(tag), []).append((tag, r))
    print("sources: " + "; ".join(f"{p} (head {h[:10] if h else '?'}, encoder {e}, {host})"
                                   for p, h, e, host in meta))
    print()
    # Per-class, per matched pair: bpp and each leg's geomean for both bodies and the ratio.
    for cname, rows in list(classes.items()) + [("ALL", [x for v in classes.values() for x in v])]:
        print(f"### {cname} ({len(rows)} tensors: {', '.join(t for t, _ in rows)})")
        print()
        print("| bpp | TCQ q | win12 q | wt TCQ | wt win | win/TCQ wt | out TCQ | out win | win/TCQ out "
              "| a4s TCQ | a4s win | win/TCQ a4s |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for qt, qw in PAIRS:
            vals = {}
            for body, q in (("tcq", qt), ("win12", qw)):
                arm = f"{body} q{q}"
                got = [r["arms"].get(arm) for _, r in rows]
                got = [g for g in got if g and "error" not in g]
                vals[body] = {leg: gmean([g.get(leg) for g in got]) for leg in LEGS}
                vals[body]["bpp"] = sum(g["bpp"] for g in got) / len(got) if got else None
                vals[body]["n"] = len(got)

            def f(x, n=4):
                return "-" if x is None else f"{x:.{n}f}"

            def ratio(leg):
                a, b = vals["win12"][leg], vals["tcq"][leg]
                return "-" if a is None or b is None else f"{a / b:.3f}"
            print(f"| {f(vals['tcq']['bpp'], 3)}/{f(vals['win12']['bpp'], 3)} | {qt} | {qw} | "
                  f"{f(vals['tcq']['wt'])} | {f(vals['win12']['wt'])} | {ratio('wt')} | "
                  f"{f(vals['tcq']['out'])} | {f(vals['win12']['out'])} | {ratio('out')} | "
                  f"{f(vals['tcq']['a4s'])} | {f(vals['win12']['a4s'])} | {ratio('a4s')} |")
        extra = []
        for arm in ("win12 q1024", "win14 q576", "win14 q960", "nvfp4 rtn"):
            got = [r["arms"].get(arm) for _, r in rows]
            got = [g for g in got if g and "error" not in g]
            if got:
                extra.append(f"{arm}: bpp {sum(g['bpp'] for g in got) / len(got):.3f}, wt "
                             f"{gmean([g['wt'] for g in got]):.4f}"
                             + (f", out {gmean([g.get('out') for g in got]):.4f}, a4s "
                                f"{gmean([g.get('a4s') for g in got]):.4f}"
                                if any(g.get('out') for g in got) else ""))
        print()
        print("; ".join(extra))
        print()
    # A-side: static vs dynamic global, per tensor.
    print("### A-side global: static (fit-row calibration) vs per-token dynamic")
    print()
    print("| tensor | act err static (vLLM) | static (ref) | dynamic (ref) | ref==vLLM codes | tokens > fit amax "
          "| sat blocks static | a4s_ref/a4d at win12 q704 |")
    print("|---|---|---|---|---|---|---|---|")
    for tag, r in tensors.items():
        a = r.get("a_side")
        if not a:
            continue
        arm = r["arms"].get("win12 q704", {})
        rr = (arm.get("a4s_ref") / arm.get("a4d")) if arm.get("a4d") else None
        print(f"| {tag} | {a['act_err_static']:.5f} | {a['act_err_static_ref']:.5f} | "
              f"{a['act_err_dynamic']:.5f} | {1 - a['ref_vs_vllm_static']['code_mismatch_frac']:.5f} | "
              f"{a['tokens_over_fit_amax']} | {a['events_static']['saturated_blocks']} | "
              f"{'-' if rr is None else f'{rr:.4f}'} |")
    errs = [(t, arm, v["error"][:120]) for t, r in tensors.items() for arm, v in r["arms"].items()
            if "error" in v]
    if errs:
        print()
        print("errors: " + json.dumps(errs))


if __name__ == "__main__":
    main()
