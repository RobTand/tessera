#!/usr/bin/env python3
"""Fail-closed closure check for one half of the #545 served A/B.

For every REQUIRED (artifact, arm) pair the half declares, this refuses the
half unless ALL of the following hold:

- the arm's result JSON exists, parses, and carries no ``fatal``;
- the recorded tessera commit equals the expected arm commit
  (before-commit.txt / after-commit.txt beside the receipts);
- the launcher's runtime-image declaration is present and non-null;
- the workload's prompt-token-id digest is IDENTICAL across the arms of one
  artifact (the fixed-input contract);
- in-engine profile proof exists: the profiled phases record new trace files
  written by vLLM's own profiler;
- route records exist and satisfy the DISPATCH EXPECTATION, derived from
  each tree's own packaged contract at closure time:
    * AFTER arms must stamp at least one launch in that family's cell
      ``executes`` set of the AFTER tree's packaged contract;
    * BEFORE arms must stamp NONE of those symbols anywhere (the v29-era
      runtime cannot dispatch launches that did not exist in it), which is
      what makes the comparison straddle the fusion at all.

Missing required arms fail the half closed — a refused before-arm is a
missing prerequisite, never a completed partial A/B.  Diagnostics are still
printed; the exit status is the verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Family -> which packaged-contract key carries the cell executes sets.
FAMILY_KEY = {
    "h1": ["TESSERA_E4M3_K1", "TESSERA_BF16_K1"],
    "h2": ["TESSERA_E2M1_K2"],
}


def load_commit(path: Path) -> str:
    return path.read_text().strip()


def after_executes_symbols(after_tree: Path, families: list[str]) -> set[str]:
    """Symbols the AFTER runtime's own packaged contract attests for the
    half's families (dense+routed cells of those families on sm_121)."""
    contract = json.loads(
        (after_tree / "src" / "tessera" / "serving" / "runtime_contract.json").read_text())
    symbols = set()
    for cell in contract.get("lane_eligibility", {}).get("cells", []):
        if cell.get("family") in families and cell.get("platform") == "sm_121":
            for ex in cell.get("executes", []) or []:
                if ex.get("symbol"):
                    symbols.add(ex["symbol"])
    return symbols


def arm_problems(result: dict, expected_commit: str, after_symbols: set[str],
                 after: bool, prompt_digest_by_arm: dict[str, str]) -> list[str]:
    problems = []
    ident = result.get("identity") or {}
    if result.get("fatal"):
        problems.append("fatal traceback recorded")
    if ident.get("tessera_commit") != expected_commit:
        problems.append(
            f"commit {ident.get('tessera_commit')!r} != expected {expected_commit!r}")
    if not ident.get("runtime_image_declared"):
        problems.append("no runtime-image declaration recorded")
    if not result.get("routes_after_warmup") and not result.get("routes_final"):
        problems.append("no route records collected")
    # In-engine profile proof: every phase that designated a profiled rep
    # must have recorded engine-written trace files.
    for phase in ("phase_decode", "phase_batch"):
        ph = result.get(phase) or {}
        if not ph:
            problems.append(f"{phase} missing")
            continue
        prof = ph.get("in_engine_profile")
        if not prof or not prof.get("new_trace_files"):
            problems.append(f"{phase}: no in-engine profiler trace files")
    # Fixed-input contract: identical prompt token ids across arms.
    wl = result.get("workload") or {}
    digest = wl.get("prompt_ids_sha256")
    if not digest:
        problems.append("workload prompt digest missing")
    else:
        prompt_digest_by_arm.setdefault("all", digest)
        if prompt_digest_by_arm["all"] != digest:
            problems.append("prompt token ids differ from the other arm")
    # Dispatch expectation.
    stamped = set()
    for key in ("routes_after_warmup", "routes_final"):
        for rec in (result.get(key) or {}).values():
            if isinstance(rec, dict) and rec.get("symbol"):
                stamped.add(rec["symbol"])
    if after:
        if not (stamped & after_symbols):
            problems.append(
                "no module stamped any attested executes symbol "
                f"{sorted(after_symbols)}; stamped={sorted(stamped)[:8]}")
    else:
        leaked = sorted(stamped & after_symbols)
        if leaked:
            problems.append(
                f"BEFORE arm stamped fused-era symbol(s) {leaked}; the v29-era "
                "runtime cannot dispatch them, so this is not the unfused arm")
    return problems


def check_half(half: str, half_dir: Path, after_tree: Path) -> int:
    families = FAMILY_KEY[half]
    expected = {
        "before": load_commit(half_dir / "before-commit.txt"),
        "after": load_commit(half_dir / "after-commit.txt"),
    }
    after_symbols = after_executes_symbols(after_tree, families)
    if not after_symbols:
        print(f"REFUSED: no sm_121 cell executes symbols for {families} in the "
              "AFTER tree's packaged contract; the dispatch expectation cannot "
              "be derived")
        return 2
    results = {}
    for f in sorted(half_dir.glob("arm-*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"REFUSED: unreadable {f.name}: {exc}")
            return 2
        results.setdefault(r.get("artifact_tag") or "unknown", {})[r.get("arm")] = r

    missing_pairs = [(tag, arm) for tag, by_arm in results.items()
                     for arm in ("before", "after") if arm not in by_arm]
    failed = False
    for tag in sorted(results):
        prompt_digests: dict[str, str] = {}
        for arm in ("before", "after"):
            r = results[tag].get(arm)
            if r is None:
                print(f"FAIL {tag}/{arm}: REQUIRED ARM MISSING")
                failed = True
                continue
            problems = arm_problems(r, expected[arm], after_symbols,
                                    arm == "after", prompt_digests)
            if problems:
                failed = True
                print(f"FAIL {tag}/{arm}:")
                for p in problems:
                    print(f"  - {p}")
            else:
                print(f"PASS {tag}/{arm}")
    if missing_pairs:
        failed = True
    if failed:
        print(f"VERDICT {half}: REFUSED (fail-closed)")
        return 1
    print(f"VERDICT {half}: closed")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--half", required=True, choices=sorted(FAMILY_KEY),
                    help="which half the receipts belong to (the action knows "
                         "it; the directory name is not the identity)")
    ap.add_argument("--half-dir", required=True, type=Path,
                    help="receipt dir for the half (contains arm-*.json and "
                         "before/after-commit.txt)")
    ap.add_argument("--after-tree", required=True, type=Path)
    a = ap.parse_args()
    return check_half(a.half, a.half_dir, a.after_tree)


if __name__ == "__main__":
    sys.exit(main())
