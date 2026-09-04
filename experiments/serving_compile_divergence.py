#!/usr/bin/env python3
"""Characterise the served eager-vs-compiled gap, from dumps already on disk.

WHY THIS EXISTS.  Two receipts quote a number for "the same weights served two
ways" and neither says what kind of number it is.  The plugin receipt
(``docs/measurements/tessera-serving-plugin-2026-09-02.md``) reports 0.0176 for
``k2-resident-graph``; that is a *plugin-vs-Gridbook* mutual KL on the NVFP4
route, not an eager-vs-compiled gap.  The allocated receipt
(``docs/measurements/tessera-allocated-served-2026-09-02.md``) then compared its
own 0.0288 / 0.0200 eager-vs-compiled numbers against that 0.0176 as if the
three measured one quantity, and issue #16 inherited the table.  They do not.

Every position dump those receipts were computed from is still on disk under
``/mnt/shared/tessera-kl`` with its artifact path, corpus contract and
tokenizer digest attached, so the whole question is answerable **offline**: no
serve, no GPU, no new bytes.  This script recomputes the comparisons that
matter, from the same tool (``kl_tool.py compare``) the receipts used, in three
families:

* ``eager_vs_compiled`` -- one artifact, one residency mode, eager dump against
  compiled dump.  Repeated across two independent lanes (the Gridbook lane and
  the Tessera plugin) on the same three checkpoints, which is the replication
  test: if the gap is a deterministic property of a route it comes back the
  same to six digits from two serves that never shared a process.
* ``build_vs_build`` -- one artifact, one mode, one regime, two *compiled
  builds* of it, plus the replay of a single build.  This is the axis issue #16
  calls "compiled builds are not reproducible".
* ``lane_vs_lane`` -- the plugin against the Gridbook lane on identical bytes,
  which is where the 0.0176 actually comes from.

Guards, because a comparison of the wrong pair would look like a result: an
``eager_vs_compiled`` or ``build_vs_build`` row asserts both dumps name the
same ``artifact_path`` and the same corpus contract sha, and every row asserts
the tokenizer digest matches.  A missing dump is reported, not skipped.

Since #30 every row also carries the **build identity** of its two arms, read
from the ``<dump>.build.json`` sidecars the serve wrappers stamp
(``tessera.serving.build_identity``): what compiled artifact each arm actually
served, digested from the contents of the compile-cache slot rather than from
the AOT key, which does not distinguish two builds.  Two rows *claim* something
about the build -- one is a rebuild, one is a replay -- and those two are
checked and refused on mismatch, because reading them the wrong way round would
report 0.000000 as evidence that rebuilds agree.  Every dump taken before the
stamp existed reads ``unstamped``: reported in the row and in ``problems``, and
still compared, since an archive that cannot be read is not a safer archive.

Usage::

    TMPDIR=/home/rob/tmp CUDA_VISIBLE_DEVICES="" python experiments/serving_compile_divergence.py \
        --out experiments/results/serving_compile_divergence.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.serving import build_identity as bid  # noqa: E402

DUMPS = Path("/mnt/shared/tessera-kl")
KL_TOOL = Path("/home/rob/dq-runs/kl_tool.py")

# (family, label, teacher dump, student dump, note)
COMPARISONS: list[tuple[str, str, str, str, str]] = [
    # ---- eager vs compiled, Tessera plugin (2026-09-02 chain) ----
    ("eager_vs_compiled", "plugin E4M3 resident",
     "qwen_tessera_e8-resident", "qwen_tessera_e8-resident-graph", "FP8 route, q1024 uniform"),
    ("eager_vs_compiled", "plugin E4M3 streamed",
     "qwen_tessera_e8-streamed", "qwen_tessera_e8-streamed-graph", "FP8 route, q1024 uniform"),
    ("eager_vs_compiled", "plugin K2 resident",
     "qwen_tessera_k2-resident", "qwen_tessera_k2-resident-graph", "NVFP4 route, q896"),
    ("eager_vs_compiled", "plugin K2 resident (fresh build)",
     "qwen_tessera_k2-resident", "qwen_tessera_k2-resident-graph-fresh", "NVFP4 route, second build"),
    ("eager_vs_compiled", "plugin K2 streamed",
     "qwen_tessera_k2-streamed", "qwen_tessera_k2-streamed-graph", "NVFP4 route, q896"),
    ("eager_vs_compiled", "plugin mixed resident",
     "qwen_tessera_mixed-resident", "qwen_tessera_mixed-resident-graph", "both routes in one body"),
    ("eager_vs_compiled", "plugin mixed streamed",
     "qwen_tessera_mixed-streamed", "qwen_tessera_mixed-streamed-graph", "both routes in one body"),
    # ---- eager vs compiled, Gridbook lane (the pre-move runtime) ----
    ("eager_vs_compiled", "gridbook E4M3 resident",
     "qwen_gridbook_e8-resident", "qwen_gridbook_e8-resident-graph", "same bytes, other lane"),
    ("eager_vs_compiled", "gridbook E4M3 streamed",
     "qwen_gridbook_e8-streamed", "qwen_gridbook_e8-streamed-graph", "same bytes, other lane"),
    ("eager_vs_compiled", "gridbook K2 resident",
     "qwen_gridbook_k2-resident-v14", "qwen_gridbook_k2-resident-graph", "same bytes, other lane"),
    ("eager_vs_compiled", "gridbook K2 streamed",
     "qwen_gridbook_k2-streamed", "qwen_gridbook_k2-streamed-graph", "same bytes, other lane"),
    ("eager_vs_compiled", "gridbook mixed resident",
     "qwen_gridbook_mixed-resident", "qwen_gridbook_mixed-resident-graph", "same bytes, other lane"),
    ("eager_vs_compiled", "gridbook mixed streamed",
     "qwen_gridbook_mixed-streamed", "qwen_gridbook_mixed-streamed-graph", "same bytes, other lane"),
    # ---- eager vs compiled, the allocated-serve campaign ----
    ("eager_vs_compiled", "allocated 4.0 resident",
     "qwen_tessera_alloc4-resident", "qwen_tessera_alloc4-resident-graph", "FP8 route, four rungs"),
    ("eager_vs_compiled", "uniform R1006 resident",
     "qwen_tessera_unif1006-resident", "qwen_tessera_unif1006-resident-graph", "FP8 route, one rung"),
    # ---- eager vs compiled, the stock lane (vanilla vLLM, no plugin) ----
    ("eager_vs_compiled", "stock twin K2 (vanilla vLLM NVFP4)",
     "qwen_stock_tessera-k2", "qwen_stock_tessera-k2-graph", "materialised NVFP4 bytes, no plugin"),
    # ---- build vs build, and the replay of one build ----
    ("build_vs_build", "plugin K2 resident compiled: chain build vs fresh build",
     "qwen_tessera_k2-resident-graph", "qwen_tessera_k2-resident-graph-fresh", "two builds of one graph"),
    ("build_vs_build", "plugin K2 resident compiled: build replayed by a second serve",
     "qwen_tessera_k2-resident-graph", "qwen_tessera_k2-resident-graph-rep", "one build, served twice"),
    # ---- lane vs lane (where 0.0176 actually comes from) ----
    ("lane_vs_lane", "K2 resident compiled: plugin vs gridbook",
     "qwen_gridbook_k2-resident-graph", "qwen_tessera_k2-resident-graph", "the receipt's 0.0176"),
    ("lane_vs_lane", "K2 resident compiled (fresh build): plugin vs gridbook",
     "qwen_gridbook_k2-resident-graph", "qwen_tessera_k2-resident-graph-fresh", "the rebuild"),
    ("lane_vs_lane", "K2 resident eager: plugin vs gridbook",
     "qwen_gridbook_k2-resident-v14", "qwen_tessera_k2-resident", "the eager control"),
    ("lane_vs_lane", "E4M3 resident compiled: plugin vs gridbook",
     "qwen_gridbook_e8-resident-graph", "qwen_tessera_e8-resident-graph", "the FP8 route's compiled arm"),
]


# Which rows make a claim about the compiled BUILD, and therefore have to be
# checked against the build identities the serve wrappers now stamp (issue
# #30).  Only two rows do: the whole point of the build_vs_build family is that
# one of its rows is a rebuild and the other is a replay, and reading them the
# wrong way round would report 0.000000 as evidence that rebuilds agree.  Every
# other row compares two different things (two regimes, two lanes, two
# artifacts) where the builds are allowed to differ, so it is annotated, not
# constrained.  Rows dumped before the stamp existed report "unstamped" and
# still compute -- an archive that cannot be read is not a safer archive.
BUILD_EXPECTATION: dict[str, str] = {
    "plugin K2 resident compiled: chain build vs fresh build": "distinct",
    "plugin K2 resident compiled: build replayed by a second serve": "same",
}


def _meta(name: str) -> dict:
    return json.loads((DUMPS / f"{name}.meta.json").read_text())


def _build(name: str) -> dict | None:
    path = DUMPS / f"{name}.build.json"
    return bid.load(path) if path.is_file() else None


def _build_check(label: str, teacher: str, student: str,
                 problems: list[str]) -> dict:
    """Read both arms' build identities; report, and refuse what was declared."""
    a, b = _build(teacher), _build(student)
    expect = BUILD_EXPECTATION.get(label)
    missing = [n for n, r in ((teacher, a), (student, b)) if r is None]
    if missing:
        note = f"{label}: build identity not stamped for {', '.join(missing)}"
        if expect:
            problems.append(
                note + f" -- this row claims the two arms are a '{expect}' build pair "
                "and nothing on disk supports that")
        return {"status": "unstamped", "expected": expect,
                "same_dispatch": None, "dispatch_known": False,
                "fingerprints": {teacher: a and a["build_fingerprint"],
                                 student: b and b["build_fingerprint"]}}
    verdict = bid.compare(a, b)
    status = "same_build" if verdict["same_build"] else "different_build"
    if expect:
        require = (bid.require_same_build if expect == "same"
                   else bid.require_distinct_build)
        try:
            require(a, b, why=label)
        except bid.BuildIdentityError as exc:
            problems.append(str(exc))
            status = "REFUSED"
    return {
        "status": status,
        "expected": expect,
        # Which implementations each arm ran (#16): on this runtime an eager
        # arm and a compiled arm do not run the same program unless the
        # dispatch was pinned, so a row that crosses regimes says so here
        # rather than leaving the reader to infer it from the family name.
        "same_dispatch": verdict["same_dispatch"],
        "dispatch_known": verdict["dispatch_known"],
        "differs": verdict["differs"],
        "incomplete": verdict["incomplete"],
        "fingerprints": {teacher: a["build_fingerprint"],
                         student: b["build_fingerprint"]},
    }


def _dispatch_mark(build: dict) -> str:
    """One column for both questions: which build, and which implementations."""
    if not build.get("dispatch_known"):
        return build["status"] + " ?"
    return build["status"] + (" =ops" if build["same_dispatch"] else " !ops")


def _identity(meta: dict) -> dict:
    tok = meta.get("tokenizer", {})
    files = tok.get("files", {})
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()[:12]
    return {
        "artifact_path": meta.get("model", {}).get("artifact_path"),
        "corpus_sha256": meta.get("corpus", {}).get("contract_sha256"),
        "tokenizer": digest,
        "produced_at_utc": meta.get("produced_at_utc"),
    }


def _compare(teacher: str, student: str, label: str, workdir: Path, python: str) -> dict:
    stem = label.replace(" ", "_").replace("/", "_").replace(":", "")
    out = workdir / f"{stem}.json"
    if not out.exists():
        cmd = [
            python, str(KL_TOOL), "compare",
            str(DUMPS / f"{teacher}.json.npz"), str(DUMPS / f"{student}.json.npz"),
            "--teacher-label-override", teacher,
            "--out", str(out),
        ]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", TMPDIR="/home/rob/tmp")
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    return json.loads(out.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/serving_compile_divergence.json")
    ap.add_argument("--workdir", default="/home/rob/tmp/serving-compile")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    problems: list[str] = []
    for family, label, teacher, student, note in COMPARISONS:
        missing = [n for n in (teacher, student)
                   if not (DUMPS / f"{n}.json.npz").exists()]
        if missing:
            problems.append(f"{label}: missing dump(s) {missing}")
            continue
        tid, sid = _identity(_meta(teacher)), _identity(_meta(student))
        if tid["tokenizer"] != sid["tokenizer"]:
            problems.append(f"{label}: tokenizer digests differ")
            continue
        if tid["corpus_sha256"] != sid["corpus_sha256"]:
            problems.append(f"{label}: corpus contracts differ")
            continue
        if family in ("eager_vs_compiled", "build_vs_build"):
            if tid["artifact_path"] != sid["artifact_path"]:
                problems.append(
                    f"{label}: {family} across two artifacts "
                    f"({tid['artifact_path']} vs {sid['artifact_path']})")
                continue
        build = _build_check(label, teacher, student, problems)
        res = _compare(teacher, student, label, workdir, args.python)
        rows.append({
            "family": family,
            "label": label,
            "note": note,
            "build": build,
            "teacher": teacher,
            "student": student,
            "artifact_path": tid["artifact_path"],
            "artifact_path_student": sid["artifact_path"],
            "positions": res["positions"],
            "kl_lower_mean": res["all"]["kl_lower_mean"],
            "kl_lower_p99": res["all"]["kl_lower_p99"],
            "top1_agree_pct": res["all"]["top1_agree_pct"],
            "kl_confident": res["confident"]["kl_lower_mean"],
        })

    payload = {
        "schema": "tessera.serving_compile_divergence/2",
        "dumps": str(DUMPS),
        "rows": rows,
        "problems": problems,
    }
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, indent=1) + "\n")

    width = max((len(r["label"]) for r in rows), default=10)
    for family in ("eager_vs_compiled", "build_vs_build", "lane_vs_lane"):
        sel = [r for r in rows if r["family"] == family]
        if not sel:
            continue
        print(f"\n== {family} ==")
        print(f"  {'comparison':<{width}}  {'KL >=':>9}  {'top-1':>7}  {'p99':>8}  "
              f"{'build/dispatch':<15}  note")
        for r in sel:
            print(f"  {r['label']:<{width}}  {r['kl_lower_mean']:9.6f}  "
                  f"{r['top1_agree_pct']:6.2f}%  {r['kl_lower_p99']:8.4f}  "
                  f"{_dispatch_mark(r['build']):<15}  {r['note']}")
    if problems:
        print("\n== problems ==")
        for p in problems:
            print("  " + p)
    print(f"\n-> {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
