"""CPU controls for the executable shared-fold qualification decisions."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/t8r_speed/shared_fold_check.py"
spec = importlib.util.spec_from_file_location("shared_fold_check", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
ADDED = "_Z23token_sum_shared_kernelv"
DEMANGLED = "token_sum_shared_kernel()"


@pytest.fixture
def cases():
    return [dict(library=lib, tokens=tokens, data=data, bitwise=True,
                 targeted_failures=[],
                 runs=[dict(order=i, form=form, sha256="a" * 64, differing=0,
                            targeted_failures=[])
                       for i, form in enumerate(("stock", "fold", "stock", "fold"))])
            for lib in check.LIBRARIES for tokens in check.TOKENS
            for data in ("served_scale", "adversarial")]


@pytest.fixture
def sass():
    return dict(dump_before_rc=0, dump_after_rc=0, compare_rc=0,
                demangled={ADDED: DEMANGLED},
                report={lib: dict(identical=["original_kernel"], multiset=[], differs=[],
                                  only_before=[], only_after=[ADDED])
                        for lib in check.SASS_LIBRARIES})


def test_complete_numeric_coverage(cases):
    assert check.qualify_numeric(cases)["passed"]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "library", "tokens", "data", "error", "empty"])
def test_numeric_coverage_refuses_incomplete_or_unexpected_cases(cases, defect):
    if defect == "missing":
        cases.pop()
    elif defect == "duplicate":
        cases.append(copy.deepcopy(cases[0]))
    elif defect in ("library", "data"):
        cases[0][defect] = "outside_scope"
    elif defect == "tokens":
        cases[0][defect] = 9
    elif defect == "error":
        cases[0]["error"] = "launch failed"
    else:
        cases.clear()
    assert not check.qualify_numeric(cases)["passed"]


@pytest.mark.parametrize("run", range(4))
@pytest.mark.parametrize("defect", ["bits", "digest", "targeted", "order", "form", "error"])
def test_numeric_refuses_defects_in_every_execution(cases, run, defect):
    case = next(c for c in cases if c["data"] == "adversarial")
    row = case["runs"][run]
    field, value = {"bits": ("differing", 1), "digest": ("sha256", "b" * 64),
                    "targeted": ("targeted_failures", ["wrong answer"]),
                    "order": ("order", 99), "form": ("form", "other"),
                    "error": ("error", "execution failed")}[defect]
    row[field] = value
    assert not check.qualify_numeric(cases)["passed"]


@pytest.mark.parametrize("field", ["runs", "bitwise", "targeted_failures"])
def test_numeric_refuses_omitted_evidence(cases, field):
    del cases[0][field]
    assert not check.qualify_numeric(cases)["passed"]


def test_complete_identical_sass(sass):
    assert check.qualify_sass(sass)["passed"]


@pytest.mark.parametrize("defect", ["multiset", "differs", "only_before", "missing_library",
                                    "empty", "extra_addition", "no_addition", "missing_field",
                                    "bad_rows", "bad_report", "duplicate", "error"])
def test_sass_refuses_incomplete_or_changed_kernels(sass, defect):
    rows = sass["report"][check.SASS_LIBRARIES[0]]
    if defect in ("multiset", "differs", "only_before"):
        rows["identical"] = []
        rows[defect] = [dict(kernel="original_kernel")] if defect == "differs" else ["original_kernel"]
    elif defect == "missing_library":
        del sass["report"][check.SASS_LIBRARIES[-1]]
    elif defect == "empty":
        rows["identical"] = []
    elif defect == "extra_addition":
        rows["only_after"].append("other_kernel")
    elif defect == "no_addition":
        rows["only_after"] = []
    elif defect == "missing_field":
        del rows["identical"]
    elif defect == "bad_rows":
        rows["identical"] = "not a list"
    elif defect == "bad_report":
        sass["report"] = []
    elif defect == "duplicate":
        rows["identical"] *= 2
    else:
        sass["error"] = "command failed"
    assert not check.qualify_sass(sass)["passed"]


@pytest.mark.parametrize("name", ["prefix_token_sum_shared_kernel()", "token_sum_shared_kernel_suffix()",
                                  "token_sum_shared_kernel<int>()", "token_sum_shared_kernel",
                                  "wrapper(token_sum_shared_kernel)", "_Z23token_sum_shared_kernelv",
                                  "token_sum_shared_kernel() [clone .isra.0]"])
def test_sass_refuses_added_functions_outside_scope(sass, name):
    sass["demangled"][ADDED] = name
    assert not check.qualify_sass(sass)["passed"]


def test_sass_accepts_exact_unqualified_function_in_namespace(sass):
    sass["demangled"][ADDED] = "(anonymous namespace)::token_sum_shared_kernel(unsigned short const*, int)"
    assert check.qualify_sass(sass)["passed"]


@pytest.mark.parametrize("command", ["dump_before_rc", "dump_after_rc", "compare_rc"])
@pytest.mark.parametrize("rc", [None, 1])
def test_sass_refuses_failed_or_missing_commands(sass, command, rc):
    if rc is None:
        del sass[command]
    else:
        sass[command] = rc
    assert not check.qualify_sass(sass)["passed"]


def test_overall_requires_both_numeric_and_sass_evidence(cases, sass):
    sass.update(check.qualify_sass(sass))
    assert check.qualify(dict(cases=cases, sass=sass))["passed"]
    assert not check.qualify(dict(cases=cases))["passed"]
    assert not check.qualify(dict(sass=sass))["passed"]
    sass["report"][check.SASS_LIBRARIES[0]]["multiset"] = ["other"]
    assert not check.qualify(dict(cases=cases, sass=sass))["passed"]


@pytest.mark.parametrize("baseline", ["missing", "nonexistent", "directory"])
def test_entry_point_names_baseline_refusal(tmp_path, baseline):
    argv = [sys.executable, str(SCRIPT), "--out", str(tmp_path / "out")]
    if baseline != "missing":
        argv += ["--base-source", str(tmp_path if baseline == "directory" else tmp_path / "absent.cu")]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    result = subprocess.run(argv, capture_output=True, text=True, env=env)
    assert result.returncode != 0
    assert "--base-source" in result.stdout + result.stderr
    receipt = json.loads((tmp_path / "out/shared_fold_check.json").read_text())
    assert not receipt["qualification"]["passed"]
    assert receipt["qualification"]["failure_reasons"]


@pytest.fixture
def baseline(tmp_path):
    source = tmp_path / "base.cu"
    source.write_text("pre-fold source")
    return source


def fake_sass_commands(monkeypatch, sass, *, failure=None, malformed=False, kernels_by_tag=None):
    def run(argv, **kwargs):
        rc, stdout = 0, ""
        if "dump" in argv:
            destination = Path(argv[argv.index("--out") + 1])
            destination.mkdir(parents=True, exist_ok=True)
            kernels = {"original_kernel": ["EXIT"]}
            if destination.name == "sass-after":
                kernels[ADDED] = ["EXIT"]
            if kernels_by_tag is not None:
                kernels = kernels_by_tag[destination.name]
            for lib in check.SASS_LIBRARIES:
                (destination / f"{lib}.json").write_text(json.dumps(dict(kernels=kernels)))
            assert argv[argv.index("--libraries") + 1].split(",") == list(check.SASS_LIBRARIES)
            rc = 1 if failure == destination.name else 0
        elif "compare" in argv:
            Path(argv[-1]).write_text("{" if malformed else json.dumps(sass["report"]))
            rc = 1 if failure == "compare" else 0
        else:
            assert argv[0] == "c++filt"
            stdout = DEMANGLED + "\n"
            rc = 1 if failure == "demangle" else 0
        return subprocess.CompletedProcess(argv, rc, stdout, "")
    monkeypatch.setattr(check.subprocess, "run", run)


def test_sass_execution_retains_artifacts_and_source_digests(monkeypatch, tmp_path, baseline, sass):
    fake_sass_commands(monkeypatch, sass)
    result = check.sass_check(str(baseline), str(tmp_path))
    assert result["passed"]
    assert result["base_sha256"] == hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert result["new_sha256"] == hashlib.sha256(Path(check.SOURCE).read_bytes()).hexdigest()
    assert (tmp_path / "sass-compare.json").is_file()
    for lib in check.SASS_LIBRARIES:
        assert (tmp_path / f"sass-before/{lib}.json").is_file()
        assert (tmp_path / f"sass-after/{lib}.json").is_file()


def test_sass_execution_normalizes_names_but_demangles_original_symbol(monkeypatch, tmp_path, baseline, sass):
    def symbol(hash_value, function):
        namespace = f"_GLOBAL__N__{hash_value}_1_routed_fused_window_cu_abcd"
        return f"_ZN{len(namespace)}{namespace}{len(function)}{function}Ev"

    old = symbol("abcd", "original_kernel")
    new = symbol("beef", "original_kernel")
    added = symbol("beef", "token_sum_shared_kernel")
    normal = check.sass_dump._kernel_name(added)
    for rows in sass["report"].values():
        rows["identical"] = [check.sass_dump._kernel_name(old)]
        rows["only_after"] = [normal]
    fake_sass_commands(monkeypatch, sass,
                       kernels_by_tag={"sass-before": {old: ["EXIT"]},
                                       "sass-after": {new: ["EXIT"], added: ["EXIT"]}})
    run = check.subprocess.run

    def demangle_original(argv, **kwargs):
        if argv[0] == "c++filt":
            assert argv[1] == added
        return run(argv, **kwargs)

    monkeypatch.setattr(check.subprocess, "run", demangle_original)
    result = check.sass_check(str(baseline), str(tmp_path))
    assert result["passed"]
    assert result["demangled"][normal] == DEMANGLED


@pytest.mark.parametrize("failure", ["sass-before", "sass-after", "compare", "demangle", "malformed"])
def test_sass_execution_refuses_command_or_report_failure(monkeypatch, tmp_path, baseline, sass, failure):
    fake_sass_commands(monkeypatch, sass, failure=failure, malformed=failure == "malformed")
    result = check.sass_check(str(baseline), str(tmp_path))
    assert not result["passed"]
    assert result["failure_reasons"]


@pytest.mark.parametrize("sass_passes", [False, True])
def test_written_verdict_agrees_with_exit_status(monkeypatch, tmp_path, baseline, cases, sass, sass_passes):
    from tessera import routed_fused as rf

    if not sass_passes:
        sass["report"][check.SASS_LIBRARIES[0]]["multiset"] = ["changed_order"]
    sass.update(check.qualify_sass(sass))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--out", str(tmp_path), "--base-source", str(baseline)])
    monkeypatch.setattr(check.torch.cuda, "get_device_name", lambda: "CPU test double")
    monkeypatch.setattr(check.torch.cuda, "get_device_capability", lambda: (12, 1))
    monkeypatch.setattr(rf, "_ext", lambda lib: SimpleNamespace())
    by_key = {(c["library"], c["data"], c["tokens"]): c for c in cases}
    monkeypatch.setattr(check, "check_case", lambda lib, ext, data, tokens, seed: by_key[lib, data, tokens])
    monkeypatch.setattr(check, "sass_check", lambda *args: sass)
    monkeypatch.setattr(check, "screen", lambda *args: {})
    rc = check.main()
    receipt = json.loads((tmp_path / "shared_fold_check.json").read_text())
    assert receipt["all_bitwise"]
    assert receipt["qualification"]["passed"] == sass_passes
    assert rc == (0 if sass_passes else 1)
    assert bool(receipt["qualification"]["failure_reasons"]) != sass_passes
