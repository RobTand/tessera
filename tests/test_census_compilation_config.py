"""The census can hand vLLM a compilation_config, and says what compiled attests.

THE DEFECT THIS PINS (#704).  ``--compiled`` built ``LLM(enforce_eager=False)``
with no ``compilation_config``, so vLLM defaulted to VLLM_COMPILE with
FULL_AND_PIECEWISE graphs, which ``glm53_nope`` refuses for any graph mode but
FULL_DECODE_ONLY: a compiled GLM-5.3 census stopped at engine start.  The
receipt also said nothing about the limit of a compiled observation: routes
that ran, with shapes unattested.

THE FAIL-BEFORE.  On the pre-change tree ``compilation_kwargs`` does not exist
and ``--compilation-config`` is not an argument, so these tests fail at
attribute lookup and at parse.  No GPU or vLLM is needed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving.runtime_image import container_env, resolve  # noqa: E402

IMAGE = "example/runtime@sha256:" + "1" * 64
CONFIG = '{"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}'
GRAPH_ONLY = '{"mode": "NONE", "cudagraph_mode": "FULL_DECODE_ONLY"}'
GRAPH_ONLY_INT = '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'


def _tool():
    spec = importlib.util.spec_from_file_location(
        "census_compilation_under_test", ROOT / "tools" / "tessera_route_census.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(*argv):
    env = container_env(resolve(IMAGE, inspector=lambda _ref: {
        "present": True, "local_id": "sha256:" + "ab" * 32,
        "repo_digests": [IMAGE]}))
    return _tool().parse_args(
        ["/model", "/out.json", "--runtime-image", IMAGE,
         "--gpu-memory-utilization", "0.2", *argv], env=env)


def test_default_command_line_passes_no_compilation_config():
    assert _tool().compilation_kwargs(_args("--compiled")) == {}


def test_compilation_config_reaches_the_engine_unchanged():
    args = _args("--compiled", "--compilation-config", CONFIG)
    assert _tool().compilation_kwargs(args) == {
        "compilation_config": {"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"}}


def test_receipt_records_the_config_and_the_route_only_limit():
    source = (ROOT / "tools" / "tessera_route_census.py").read_text()
    assert '"compilation_config": args.compilation_config,' in source
    assert "routes_only_shapes_unattested" in source
    assert "**compilation_kwargs(args)" in source


@pytest.mark.parametrize("bad", ["not json", "[1, 2]", '"str"'])
def test_a_non_object_is_a_usage_error_before_any_load(bad, capsys):
    with pytest.raises(SystemExit) as excinfo:
        _args("--compiled", "--compilation-config", bad)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "not valid JSON" in err or "must be a JSON object" in err


def test_eager_with_a_compilation_config_is_refused(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _args("--compilation-config", CONFIG)
    assert excinfo.value.code == 2
    assert "requires --compiled" in capsys.readouterr().err


def test_graph_only_compilation_config_is_refused_as_compiled(capsys):
    """Mode NONE keeps CUDA graphs but runs no Torch trace (issue #1062)."""
    with pytest.raises(SystemExit) as excinfo:
        _args("--compiled", "--compilation-config", GRAPH_ONLY)
    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert "mode NONE" in err
    assert "no Python" in err


def test_graph_only_int_mode_is_refused_as_compiled(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _args("--compiled", "--compilation-config", GRAPH_ONLY_INT)
    assert excinfo.value.code == 2
    assert "mode NONE" in capsys.readouterr().err


def test_torch_compile_detector_names_only_disabled_mode():
    tool = _tool()
    assert tool.torch_compile_disabled_by_config({"mode": "NONE"}) is True
    assert tool.torch_compile_disabled_by_config({"mode": "none"}) is True
    assert tool.torch_compile_disabled_by_config({"mode": 0}) is True
    assert tool.torch_compile_disabled_by_config({"mode": 3}) is False
    assert tool.torch_compile_disabled_by_config({}) is False
    assert tool.torch_compile_disabled_by_config(None) is False
