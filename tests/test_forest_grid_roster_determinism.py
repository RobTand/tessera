"""Behavioral acceptance for the roster import probe, not a speed benchmark.

A child-startup deadline can expire even when the roster never fits a grid.
Inject that outcome immediately rather than depending on a busy test host.
The real roster must still execute, and an eager-fit negative control must
still be rejected.  These checks deliberately do not run Lloyd's algorithm.
"""
from __future__ import annotations

import inspect
import subprocess

import pytest

import test_forest_grid_roster as roster
from tessera import alphabet


def _invoke_probe(request):
    # Permit the corrected test to use ordinary pytest-managed fixtures such
    # as monkeypatch, while also invoking the current zero-argument test.
    probe = roster.test_importing_the_grid_roster_does_not_fit_lloyd_grids
    fixtures = {name: request.getfixturevalue(name)
                for name in inspect.signature(probe).parameters}
    probe(**fixtures)


@pytest.fixture
def child_startup_unavailable(monkeypatch):
    def expired(*args, **kwargs):
        raise subprocess.TimeoutExpired("roster child startup (injected)", 60)

    def no_spawn(*args, **kwargs):
        raise AssertionError("roster import semantics must not require a child process")

    monkeypatch.setattr(subprocess, "run", expired)
    monkeypatch.setattr(subprocess, "Popen", no_spawn)


def test_roster_probe_executes_fresh_module_without_child_process(
    request, monkeypatch, child_startup_unavailable,
):
    original_fit = alphabet.lloyd_max_grid
    original_parametrize = pytest.mark.parametrize
    declarations = []

    def observe_declaration(*args, **kwargs):
        declarations.append(args[0])
        return original_parametrize(*args, **kwargs)

    monkeypatch.setattr(pytest.mark, "parametrize", observe_declaration)
    _invoke_probe(request)

    # A cached import or deleted probe would not execute this declaration.
    assert declarations == ["grid"], "the real roster module was not freshly executed"
    assert alphabet.lloyd_max_grid is original_fit, "the import guard leaked into later tests"


def test_roster_probe_rejects_eager_fitting_and_restores_guard(
    request, monkeypatch, tmp_path, child_startup_unavailable,
):
    eager_roster = tmp_path / "eager_roster.py"
    eager_roster.write_text(
        "from tessera.alphabet import lloyd_max_grid\n"
        "lloyd_max_grid(16)\n",
        encoding="utf-8",
    )

    def unguarded_fit(*args, **kwargs):
        # Never perform a real fit even if an implementation loses its guard.
        raise RuntimeError("eager fitting reached the unguarded implementation")

    monkeypatch.setattr(alphabet, "lloyd_max_grid", unguarded_fit)
    monkeypatch.setattr(roster, "__file__", str(eager_roster))
    with pytest.raises(AssertionError, match="[Ll]loyd"):
        _invoke_probe(request)
    assert alphabet.lloyd_max_grid is unguarded_fit, "the failing import leaked its guard"
