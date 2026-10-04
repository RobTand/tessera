"""CPU-only control tests for the #545 served A/B source/control paths.

Parent review 2026-10-04 findings, pinned here so they cannot return:

- finding 1: kernel-duration claims come only from the in-engine profiler;
  the driver records the timing locus and engine-core facts explicitly;
- finding 2: the phase median is the real median (even-count convention),
  and the profiled rep is excluded from latency/energy summaries;
- finding 3: the action/analyzer fail closed on any missing required arm,
  identity, input digest, profile proof, or dispatch expectation;
- finding 4: the GLM env args exist only when the flag is 1 (the ${VAR:+}
  nonempty trap must not leak TESSERA_RESEARCH_GLM53_NOPE into h1);
- finding 5: every comparison record is labelled a matched-runtime
  comparison, never an isolated epilogue-only delta.

No torch, no GPU, no network: the tests exercise the shared summary module,
the env helper, and the closure checker against synthetic receipts.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
from served_fused_ab_summary import phase_summary  # noqa: E402

ENV_HELPER = EXPERIMENTS / "served_fused_ab_env.sh"
CLOSURE = EXPERIMENTS / "served_fused_ab_closure.py"

BEFORE_COMMIT = "4c384e6049dca3eeaf503bb2c9cd1cd2778978d1"
AFTER_COMMIT = "b40c93cb73745097e57a1ba4cf5b9eee166c759a"


def _rows(n, ms=100.0, profiled=None):
    out = []
    for r in range(1, n + 1):
        out.append({"rep": r, "engine_call_ms": ms + r, "wall_s": 0.1 + r / 1000,
                    "gen_tokens": 64, "prompt_tokens": 512,
                    "profiled": r == profiled})
    return out


class TestSummaryConvention:
    def test_even_count_real_median(self):
        # 24 reps: upper-order-statistic sorted(x)[len//2] would answer 113.0;
        # the real median of 101..124 is the mean of 112 and 113.
        s = phase_summary(_rows(24), None)
        assert s["engine_call_ms_median"] == pytest.approx((112.0 + 113.0) / 2)

    def test_odd_count_median(self):
        s = phase_summary(_rows(13), None)
        assert s["engine_call_ms_median"] == pytest.approx(107.0)

    def test_profiled_rep_excluded(self):
        s = phase_summary(_rows(12, profiled=12), 12)
        assert s["excluded_profiled_rep"] == 12
        assert s["n_included"] == 11
        assert 12 not in s["included_reps"]
        # the profiled rep's inflated time cannot move the summary
        rows = _rows(12, profiled=12)
        rows[-1]["engine_call_ms"] = 10_000.0
        rows[-1]["wall_s"] = 900.0
        s2 = phase_summary(rows, 12)
        assert s2["engine_call_ms_median"] == s["engine_call_ms_median"]
        assert s2["gen_tokens_included"] == 11 * 64

    def test_raw_population_kept_by_caller(self):
        rows = _rows(12, profiled=12)
        phase_summary(rows, 12)
        assert len(rows) == 12  # the summary never replaces the population

    def test_all_profiled_refuses(self):
        rows = _rows(4, profiled=3)
        with pytest.raises(ValueError):
            phase_summary(rows, 3) if False else phase_summary(
                [r for r in rows if r["rep"] == 3], 3)


class TestGlmEnvArgs:
    def _run(self, flag):
        script = (
            f'source "{ENV_HELPER}"\n'
            f'served_fused_ab_glm_env {flag}\n'
        )
        return subprocess.run(["bash", "-c", script], capture_output=True,
                              text=True, check=True).stdout

    def test_off_is_empty(self):
        # finding 4: GLM=0 must produce NO env args (the ${VAR:+} trap
        # exported the flag for the value "0").
        assert self._run("0") == ""

    def test_empty_is_empty(self):
        assert self._run('""') == ""

    def test_on_sets_nope_only(self):
        assert self._run("1") == "-e TESSERA_RESEARCH_GLM53_NOPE=1\n"

    def test_profiler_env(self):
        script = (
            f'source "{ENV_HELPER}"\n'
            f'served_fused_ab_profiler_env /out/profiles\n'
        )
        out = subprocess.run(["bash", "-c", script], capture_output=True,
                             text=True, check=True).stdout
        assert out == "-e VLLM_TORCH_PROFILER_DIR=/out/profiles\n"


def _write_contract(tree: Path, symbols: list[str]):
    d = tree / "src" / "tessera" / "serving"
    # test_expectation_derived_from_contract_not_magic rewrites the contract
    # on the fixture's tree, so the directory may already exist
    d.mkdir(parents=True, exist_ok=True)
    cells = [{"family": fam, "platform": "sm_121",
              "executes": [{"symbol": s, "decoder": "d"} for s in symbols]}
             for fam in ("TESSERA_E4M3_K1", "TESSERA_BF16_K1")]
    (d / "runtime_contract.json").write_text(
        json.dumps({"lane_eligibility": {"cells": cells}}))


def _write_arm(half: Path, tag: str, arm: str, *, commit: str = AFTER_COMMIT,
               fatal: bool = False, traces: bool = True, symbol: str = "FUSED_SYM",
               prompt: str = "p", image: str | None = "img@sha256:aa",
               routes: bool = True):
    ph = {"reps": [{"rep": 1, "engine_call_ms": 5.0, "wall_s": 0.01,
                    "gen_tokens": 8, "prompt_tokens": 8, "profiled": True}],
          "profiled_rep": 1}
    if traces:
        ph["in_engine_profile"] = {"new_trace_files": ["trace-rank0.json"]}
    rec = {
        "arm": arm, "artifact_tag": tag,
        "identity": {"tessera_commit": commit, "runtime_image_declared": image},
        "workload": {"prompt_ids_sha256": prompt},
        "routes_after_warmup": {"m": {"symbol": symbol, "decoder": "d"}} if routes else {},
        "routes_final": {},
        "phase_decode": dict(ph), "phase_batch": dict(ph),
    }
    if fatal:
        rec["fatal"] = "Traceback ..."
    (half / f"arm-{arm}-{tag}.json").write_text(json.dumps(rec))


def _run_closure(half: Path, after_tree: Path):
    return subprocess.run(
        [sys.executable, str(CLOSURE), "--half", "h1", "--half-dir", str(half),
         "--after-tree", str(after_tree)],
        capture_output=True, text=True)


@pytest.fixture()
def tree(tmp_path):
    _write_contract(tmp_path / "after", ["FUSED_SYM"])
    (tmp_path / "half").mkdir()
    (tmp_path / "half" / "before-commit.txt").write_text(BEFORE_COMMIT + "\n")
    (tmp_path / "half" / "after-commit.txt").write_text(AFTER_COMMIT + "\n")
    return tmp_path / "half", tmp_path / "after"


class TestClosureFailClosed:
    def test_complete_pair_passes(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM")
        _write_arm(half, "art", "after", symbol="FUSED_SYM")
        rc = _run_closure(half, after)
        assert rc.returncode == 0, rc.stdout

    def test_missing_arm_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "after", symbol="FUSED_SYM")
        rc = _run_closure(half, after)
        assert rc.returncode == 1
        assert "REQUIRED ARM MISSING" in rc.stdout

    def test_fatal_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM")
        _write_arm(half, "art", "after", symbol="FUSED_SYM", fatal=True)
        assert _run_closure(half, after).returncode == 1

    def test_missing_profile_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM")
        _write_arm(half, "art", "after", symbol="FUSED_SYM", traces=False)
        rc = _run_closure(half, after)
        assert rc.returncode == 1
        assert "no in-engine profiler trace files" in rc.stdout

    def test_commit_mismatch_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit="deadbeef", symbol="OLD_SYM")
        _write_arm(half, "art", "after", symbol="FUSED_SYM")
        assert _run_closure(half, after).returncode == 1

    def test_prompt_digest_mismatch_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM",
                   prompt="p1")
        _write_arm(half, "art", "after", symbol="FUSED_SYM", prompt="p2")
        rc = _run_closure(half, after)
        assert rc.returncode == 1
        assert "differ from the other arm" in rc.stdout

    def test_missing_image_declaration_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM",
                   image=None)
        _write_arm(half, "art", "after", symbol="FUSED_SYM")
        rc = _run_closure(half, after)
        assert rc.returncode == 1
        assert "no runtime-image declaration" in rc.stdout

    def test_before_stamping_fused_symbol_refuses(self, tree):
        # the v29-era runtime cannot dispatch the fused launches; if the
        # before arm stamps one, the comparison does not straddle the fusion
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="FUSED_SYM")
        _write_arm(half, "art", "after", symbol="FUSED_SYM")
        rc = _run_closure(half, after)
        assert rc.returncode == 1
        assert "fused-era symbol" in rc.stdout

    def test_after_without_attested_symbol_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM")
        _write_arm(half, "art", "after", symbol="SOMETHING_ELSE")
        rc = _run_closure(half, after)
        assert rc.returncode == 1
        assert "no module stamped any attested executes symbol" in rc.stdout

    def test_no_route_records_refuses(self, tree):
        half, after = tree
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM",
                   routes=False)
        _write_arm(half, "art", "after", symbol="FUSED_SYM")
        rc = _run_closure(half, after)
        assert rc.returncode == 1
        assert "no route records collected" in rc.stdout

    def test_expectation_derived_from_contract_not_magic(self, tree):
        # change the AFTER contract's attested symbol and the closure must
        # follow the contract, not a baked-in table
        half, after = tree
        _write_contract(after.parent / "after", ["OTHER_FUSED_SYM"])
        _write_arm(half, "art", "before", commit=BEFORE_COMMIT, symbol="OLD_SYM")
        _write_arm(half, "art", "after", symbol="OTHER_FUSED_SYM")
        assert _run_closure(half, after).returncode == 0


class TestScopeLabels:
    def test_driver_records_matched_runtime_scope(self):
        src = (EXPERIMENTS / "served_fused_ab_profile.py").read_text()
        assert "matched runtime comparison" in src
        assert "comparison_scope" in src

    def test_analyzer_labels_and_fails_closed(self):
        src = (EXPERIMENTS / "analyze_545_served_ab.py").read_text()
        assert "matched runtime comparison" in src
        assert "not an isolated epilogue-only delta" in src
        assert src.count("any_refused") >= 2  # refused pairs flip the exit

    def test_action_fail_closed(self):
        src = (EXPERIMENTS / "served_fused_ab_20261004.sh").read_text()
        assert "served_fused_ab_closure.py" in src
        assert 'exit "$closure"' in src  # the closure verdict IS the exit
