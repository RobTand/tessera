"""The determinism knob, measured rather than named (issue #30, second change).

WHY THIS FILE EXISTS.  ``docs/measurements/serving-compile-divergence-
2026-09-02.md`` §3b counted the mechanism behind the one non-reproducible
compiled arm: 120 of 196 autotuned Triton kernels chose a different
``XBLOCK``/``num_warps`` on a rebuild, because inductor picks those by timing
candidates on the device at compile time.  The pinned torch names a knob for
exactly that mechanism (``TORCHINDUCTOR_DETERMINISTIC=1``), wired here as
``TESSERA_SERVE_DETERMINISTIC=1`` -- and ``tests/test_serve_build_identity.py``
already proves the wiring (the env var flips the installed torch's config, it
enters the fingerprint, the warm-cache no-op is refused).  What was never
measured is what the flag *does to a build*.  ``experiments/
inductor_determinism_probe.py`` is the instrument for that half, and until now
it was an unverified wip snapshot with no test: this file runs it.

SCOPE, read before quoting this test.  It runs on the host torch (2.11.0+cu130
here, not the serving image's 2.13.0+cu130) with the GPU hidden
(``CUDA_VISIBLE_DEVICES=""`` -- a GPU job is out of scope for this change), on
a handful of reduction kernels (not the served backbone that autotuned 196).
On CPU there is no Triton autotuning at all, so both arms record zero
``.best_config`` records and zero triton kernels -- this run cannot answer
whether the flag suppresses device benchmarking.  What it does answer, and
what is pinned here:

* the env var reaches inductor at import in the compiling process
  (``on.at_import`` all True) and does not without it;
* reading ``torch._inductor.config.deterministic`` back after a compile ran
  reports False either way -- so the attribute is not evidence the flag was in
  force, and the report carries that finding machine-checked
  (``knob.summary``), not eyeball-only;
* two fresh-cache builds per arm agree bitwise on this torch;
* the report records the parameters it was compiled under (``params``), and
  every child build carries the same dict, so two reports from different
  settings cannot be mistaken for a build disagreement.

The receipt this test re-measures on every run is what keeps the campaign
decision honest: the flag stays OFF (the GPU-serve measurement -- two K2
resident builds from empty caches with the flag, compared at 0.000000 -- still
needs two live serves and is not claimed here), and the practice remains the
pinned cache root plus the content-digest stamp.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _probe_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "inductor_determinism_probe",
        ROOT / "experiments" / "inductor_determinism_probe.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_knob_summary_names_the_post_compile_reset():
    """``knob.summary`` derives the finding the probe's docstring claims.

    The module attribute cannot serve as evidence the flag was in force,
    because on the torch measured here it reads False after a compile ran
    whether or not the flag was set.  That has to be a derived field in the
    report, not a shape the reader must notice in ``knob.reads``.
    """
    probe = _probe_module()
    measured = {
        "off": {"at_import": [False, False],
                "after_first_compile": [False, False],
                "after_second_compile": [False, False]},
        "on": {"at_import": [True, True],
               "after_first_compile": [False, False],
               "after_second_compile": [False, False]},
    }
    assert probe._knob_summary(measured) == {
        "off": {"live_at_import": False, "resets_after_compile": False},
        "on": {"live_at_import": True, "resets_after_compile": True},
    }
    # The rule, not today's roster: a torch that kept reporting True would not
    # be called reset, and one where the flag never arrived would not be live.
    honest = {
        "off": {"at_import": [False], "after_first_compile": [False],
                "after_second_compile": [False]},
        "on": {"at_import": [True], "after_first_compile": [True],
               "after_second_compile": [True]},
    }
    assert probe._knob_summary(honest) == {
        "off": {"live_at_import": False, "resets_after_compile": False},
        "on": {"live_at_import": True, "resets_after_compile": False},
    }


def test_two_fresh_cache_cpu_builds_per_arm_and_what_the_knob_did(tmp_path, monkeypatch):
    """Run the probe: the receipt, re-measured, on the host torch's CPU path.

    Matched pair -- two builds per arm from a fresh inductor cache each time,
    the only difference between the arms the environment variable.  Small
    hidden width keeps the suite fast; the graph shape (row reduction plus
    pointwise tail) is the record class that differed 120 of 196 times.
    """
    torch = pytest.importorskip("torch")
    probe = _probe_module()
    out = tmp_path / "probe.json"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("TESSERA_PROBE_HIDDEN", "256")
    monkeypatch.delenv("TORCHINDUCTOR_DETERMINISTIC", raising=False)
    monkeypatch.setattr(sys, "argv",
                        ["probe", "--work", str(tmp_path / "w"), "--out", str(out)])
    assert probe.main() == 0
    report = json.loads(out.read_text())

    assert report["schema"] == "tessera.inductor_determinism_probe/1"
    assert report["torch"] == torch.__version__
    # No GPU job ran here: with the device hidden the children compile for cpu.
    assert report["device"] == "cpu"

    # The receipt records its own parameters, identically in every child build,
    # so a settings difference can never read as a build difference.
    assert report["params"] == {"hidden": 256, "second_compile_hidden": 128,
                                "shapes": [7, 256, 2048], "compile": "dynamic",
                                "device": "cpu"}

    reads = report["knob"]["reads"]
    assert reads["off"]["at_import"] == [False, False]
    assert reads["on"]["at_import"] == [True, True]
    assert report["knob"]["summary"] == {
        "off": {"live_at_import": False, "resets_after_compile": False},
        "on": {"live_at_import": True, "resets_after_compile": True},
    }

    for arm in ("off", "on"):
        assert report["arms"][arm]["outputs_bitwise_equal"] is True
    # Deliberately unasserted: autotune/kernel counts.  On the CPU path both
    # arms record zero ``.best_config`` records and zero triton kernels (this
    # run: 0/0 on both sides), so this run cannot say whether the flag
    # suppresses device benchmarking -- that needs the CUDA path, i.e. the two
    # live serves #30 asks for.  Asserting the zeros would pin a property of
    # CPU codegen, not of the knob.
