#!/usr/bin/env python3
"""What actually differs between two vLLM compile-cache builds of one key.

WHY THIS EXISTS.  ``docs/measurements/tessera-serving-plugin-2026-09-02.md`` §3
records one arm whose compiled forward disagrees with the reference lane
(0.017591 / 95.65%), shows that the identical sources recompiled from an empty
cache land on 0.000000 instead, and names the *likely* mechanism -- inductor's
compile-time benchmarking under box load -- while saying in as many words that
it "is not proven here".  Both builds are still on disk.  This script proves or
refutes it by reading them, at three levels:

1. ``cache_key_factors.json`` -- what vLLM hashed into the key.  If these are
   identical, the key did not distinguish the two builds, which is the whole
   reproducibility problem in one file.
2. ``computation_graph.py`` -- the split graph vLLM dumps after compiling.  A
   difference here is upstream of codegen: a different graph, not a different
   schedule.  Censused by ``torch.ops.vllm_ir.<op>.<overload>`` counts, because
   that is where the two builds in hand differ.
3. ``torch_aot_compile/<key>/inductor_cache/**.best_config`` -- inductor's
   per-kernel autotune choice, with its measured ``time_taken_ms``.  A record
   that differs ONLY in the time is the same choice benchmarked twice; a record
   whose ``XBLOCK``/``num_warps``/``R0_BLOCK`` differ is a different kernel, and
   a different kernel is a different reduction order and therefore different
   arithmetic.  Counting those two apart is what turns "likely" into a number.

Read-only.  Nothing here writes into or removes a compile cache.

usage::

    python experiments/compile_build_forensics.py \\
        --a /home/rob/tessera-runs/tsplugin/vllm-cache \\
        --b /home/rob/tessera-runs/tsplugin/vllm-cache-fresh \\
        --key 36d07e6697 --aot-key 15957ad9e7a7...
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

IR_OP = re.compile(r"torch\.ops\.vllm_ir\.([a-z_0-9]+)\.([a-z_0-9]+)\(")
# Keys that record the measurement rather than the choice.
TIMING_KEYS = {"time_taken_ms", "triton_cache_hash"}


def normalize_graph_overloads(text: str) -> str:
    """Strip the overload suffix off ``torch.ops.vllm_ir.<op>.<overload>(``.

    Issue #29: the dumped ``computation_graph.py`` records pass progress at
    write time -- the pinned functionalization pass rewrites every
    ``maybe_inplace`` node to ``default``, yet dumps carry 0 to 56 of them --
    so two dumps of one key can differ in suffixes alone while naming the same
    call sites.  The substitution is derived from ``IR_OP``, the same pattern
    the census counts with, so the census and the normalizer cannot disagree;
    non-``vllm_ir`` overloads (``aten.add.Tensor`` and kin) are another
    dispatcher's semantics and are left alone.
    """
    return IR_OP.sub(lambda m: f"torch.ops.vllm_ir.{m.group(1)}(", text)


def compare_graphs(a: Path, b: Path) -> dict | None:
    """Raw and overload-normalized comparison of two dumped graphs.

    Returns ``None`` when either side is missing.  ``identical_raw`` is the
    byte comparison the tool always reported; ``identical_modulo_overload``
    re-compares after :func:`normalize_graph_overloads`.  Raw-different but
    normalized-identical is an overload relabeling of the same call sites --
    a dump artefact, not a second source of build-to-build variation -- and
    ``a_ops_total``/``b_ops_total`` (per-op call counts without the suffix)
    show it.  Normalized-different is still a different graph.
    """
    if not a.is_file() or not b.is_file():
        return None
    raw_a, raw_b = a.read_bytes(), b.read_bytes()
    norm_a, norm_b = normalize_graph_overloads(
        raw_a.decode(errors="replace")), normalize_graph_overloads(
        raw_b.decode(errors="replace"))
    totals_a: dict[str, int] = dict(sorted(
        collections.Counter(
            op for op, _ in IR_OP.findall(raw_a.decode(errors="replace"))).items()))
    totals_b: dict[str, int] = dict(sorted(
        collections.Counter(
            op for op, _ in IR_OP.findall(raw_b.decode(errors="replace"))).items()))
    return {
        "identical_raw": raw_a == raw_b,
        "identical_modulo_overload": norm_a == norm_b,
        "a_sha256_16": hashlib.sha256(raw_a).hexdigest()[:16],
        "b_sha256_16": hashlib.sha256(raw_b).hexdigest()[:16],
        "a_norm_sha256_16": hashlib.sha256(norm_a.encode()).hexdigest()[:16],
        "b_norm_sha256_16": hashlib.sha256(norm_b.encode()).hexdigest()[:16],
        "a_ops_total": totals_a,
        "b_ops_total": totals_b,
    }


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _graph_census(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    counts: collections.Counter = collections.Counter()
    for line in path.read_text(errors="replace").splitlines():
        for op, overload in IR_OP.findall(line):
            counts[f"{op}.{overload}"] += 1
    return dict(sorted(counts.items()))


def _best_configs(root: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in root.rglob("*.best_config"):
        try:
            out[str(p.relative_to(root))] = json.loads(p.read_text())
        except Exception:  # noqa: BLE001  - a truncated record is data, not a crash
            out[str(p.relative_to(root))] = {"__unparsed__": True}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first vLLM cache root (VLLM_CACHE)")
    ap.add_argument("--b", required=True, help="second vLLM cache root")
    ap.add_argument("--key", required=True, help="torch_compile_cache/<key> directory name")
    ap.add_argument("--prefix", default="rank_0_0/backbone")
    ap.add_argument("--aot-key", default=None, help="torch_aot_compile/<key>; skipped if absent")
    ap.add_argument(
        "--census", action="store_true",
        help="also census every backbone graph under both roots: how many keys in a "
             "campaign's cache carry which op-overload split")
    ap.add_argument("--out", default="experiments/results/compile_build_forensics.json")
    args = ap.parse_args()

    a_root, b_root = Path(args.a), Path(args.b)
    a_dir = a_root / "torch_compile_cache" / args.key / args.prefix
    b_dir = b_root / "torch_compile_cache" / args.key / args.prefix

    report: dict = {
        "schema": "tessera.compile_build_forensics/1",
        "a": str(a_dir),
        "b": str(b_dir),
        "cache_key_factors": {
            "a_sha256_16": _sha(a_dir / "cache_key_factors.json"),
            "b_sha256_16": _sha(b_dir / "cache_key_factors.json"),
        },
        "computation_graph": {
            "a_sha256_16": _sha(a_dir / "computation_graph.py"),
            "b_sha256_16": _sha(b_dir / "computation_graph.py"),
            "a_vllm_ir_ops": _graph_census(a_dir / "computation_graph.py"),
            "b_vllm_ir_ops": _graph_census(b_dir / "computation_graph.py"),
        },
    }
    kf = report["cache_key_factors"]
    report["cache_key_factors"]["identical"] = (
        kf["a_sha256_16"] is not None and kf["a_sha256_16"] == kf["b_sha256_16"]
    )
    cg = report["computation_graph"]
    cg["identical"] = cg["a_sha256_16"] is not None and cg["a_sha256_16"] == cg["b_sha256_16"]
    verdict = compare_graphs(a_dir / "computation_graph.py", b_dir / "computation_graph.py")
    if verdict is not None:
        cg["identical_modulo_vllm_ir_overload"] = verdict["identical_modulo_overload"]
        cg["a_vllm_ir_ops_total"] = verdict["a_ops_total"]
        cg["b_vllm_ir_ops_total"] = verdict["b_ops_total"]

    if args.aot_key:
        a_aot = a_root / "torch_compile_cache" / "torch_aot_compile" / args.aot_key
        b_aot = b_root / "torch_compile_cache" / "torch_aot_compile" / args.aot_key
        a_cfg, b_cfg = _best_configs(a_aot), _best_configs(b_aot)
        shared = sorted(set(a_cfg) & set(b_cfg))
        identical = timing_only = retuned = 0
        examples: list[dict] = []
        for rel in shared:
            ja, jb = a_cfg[rel], b_cfg[rel]
            ka = {k: v for k, v in ja.items() if k not in TIMING_KEYS}
            kb = {k: v for k, v in jb.items() if k not in TIMING_KEYS}
            if ka != kb:
                retuned += 1
                if len(examples) < 8:
                    examples.append({
                        "record": rel,
                        "changed": {
                            k: [ka.get(k), kb.get(k)]
                            for k in sorted(set(ka) | set(kb))
                            if ka.get(k) != kb.get(k)
                        },
                    })
            elif ja != jb:
                timing_only += 1
            else:
                identical += 1
        report["autotune"] = {
            "a": str(a_aot),
            "b": str(b_aot),
            "records_in_both": len(shared),
            "only_in_a": len(set(a_cfg) - set(b_cfg)),
            "only_in_b": len(set(b_cfg) - set(a_cfg)),
            "byte_identical": identical,
            "same_choice_different_measured_time": timing_only,
            "different_tuning_choice": retuned,
            "examples": examples,
        }

    if args.census:
        census = []
        for root in (a_root, b_root):
            base = root / "torch_compile_cache"
            for graph in sorted(base.glob(f"*/{args.prefix}/computation_graph.py")):
                counts = _graph_census(graph) or {}
                census.append({
                    "root": str(root),
                    "key": graph.relative_to(base).parts[0],
                    "mtime": graph.stat().st_mtime,
                    "vllm_ir_ops": counts,
                })
        report["campaign_census"] = census

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")

    print(f"cache_key_factors identical : {report['cache_key_factors']['identical']}")
    print(f"computation_graph identical : {cg['identical']}")
    if not cg["identical"] and cg["a_vllm_ir_ops"] and cg["b_vllm_ir_ops"]:
        keys = sorted(set(cg["a_vllm_ir_ops"]) | set(cg["b_vllm_ir_ops"]))
        print("  vllm_ir op census (a -> b):")
        for k in keys:
            va, vb = cg["a_vllm_ir_ops"].get(k, 0), cg["b_vllm_ir_ops"].get(k, 0)
            mark = "  <-- differs" if va != vb else ""
            print(f"    {k:<48s} {va:5d} -> {vb:5d}{mark}")
        if verdict is not None:
            print(f"  identical modulo vllm_ir overload : {verdict['identical_modulo_overload']}")
            if verdict["identical_modulo_overload"]:
                print("  -> same call sites, overload suffixes only: a dump artefact "
                      "(issue #29), not a second source of build-to-build variation")
    if "autotune" in report:
        at = report["autotune"]
        print(f"autotune records in both    : {at['records_in_both']} "
              f"(only in a {at['only_in_a']}, only in b {at['only_in_b']})")
        print(f"  byte-identical            : {at['byte_identical']}")
        print(f"  same choice, new timing   : {at['same_choice_different_measured_time']}")
        print(f"  DIFFERENT TUNING CHOICE   : {at['different_tuning_choice']}")
    if args.census:
        print(f"\ncampaign census: {len(report['campaign_census'])} backbone graphs")
        splits: collections.Counter = collections.Counter()
        for row in report["campaign_census"]:
            ops = row["vllm_ir_ops"]
            splits[(ops.get("fused_add_rms_norm.maybe_inplace", 0),
                    ops.get("fused_add_rms_norm.default", 0))] += 1
        print("  fused_add_rms_norm (maybe_inplace, default) -> how many graphs:")
        for split, n in sorted(splits.items()):
            print(f"    {split}: {n}")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
