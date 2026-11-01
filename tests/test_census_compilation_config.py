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
import json
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


def test_graph_only_compilation_config_parses_as_compiled_engine_mode():
    """Mode NONE keeps CUDA graphs but runs no Torch trace (issue #1062)."""
    args = _args("--compiled", "--compilation-config", GRAPH_ONLY)
    assert args.compiled is True
    assert args.execution_mode == "compiled"
    assert tool_torch_trace(args) is False


def test_graph_only_int_mode_parses_without_a_torch_trace():
    args = _args("--compiled", "--compilation-config", GRAPH_ONLY_INT)
    assert args.execution_mode == "compiled"
    assert tool_torch_trace(args) is False


def test_real_torch_trace_stays_active_without_a_disabling_config():
    assert tool_torch_trace(_args("--compiled")) is True
    assert tool_torch_trace(_args("--compiled", "--compilation-config", CONFIG)) is True


def tool_torch_trace(args):
    """The trace flag the census checks read, derived once per run."""
    return _tool().torch_trace_active(
        compiled=args.compiled, compilation_config=args.compilation_config)

def test_torch_compile_detector_names_only_disabled_mode():
    tool = _tool()
    assert tool.torch_compile_disabled_by_config({"mode": "NONE"}) is True
    assert tool.torch_compile_disabled_by_config({"mode": "none"}) is True
    assert tool.torch_compile_disabled_by_config({"mode": 0}) is True
    assert tool.torch_compile_disabled_by_config({"mode": 3}) is False
    assert tool.torch_compile_disabled_by_config({}) is False
    assert tool.torch_compile_disabled_by_config(None) is False


# --- retained R768 receipts: the CPU regression proof (issue #1062) ---
#
# The two existing GPU actions stay successful census executions with a refused
# graph verdict. They establish no admission. These tests replay the retained
# receipts' own records through the fixed checks. No GPU or vLLM is needed.


def _retained(name):
    """The retained served-window receipt, or a skip naming its root."""
    import box_artifacts

    return box_artifacts.skip_now(
        "measurements", "tp2-served-census-sol-20261007", "served-window", name)


def _replay(receipt, *, compilation_config, shapes=None):
    """The fixed validator's problems for a retained receipt's own records."""
    from tessera.serving.contract import CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE
    from tessera.serving.scheme import TESSERA_FAMILIES

    tool = _tool()
    ranks = receipt.get("ranks") or []
    world = (receipt.get("topology") or {}).get("observed_world_size", 1) or 1
    phases = sorted(receipt["records"])
    if ranks:
        assert len(ranks) == world
        phases_by_rank = {phase: [rank["records"][phase] for rank in ranks]
                          for phase in phases}
        refusals = [rank.get("lane_refusals", {}) for rank in ranks]
        identities = [{"rank": rank.get("rank", i)} for i, rank in enumerate(ranks)]
    else:
        phases_by_rank = {phase: [receipt["records"][phase]] for phase in phases}
        refusals = [receipt.get("lane_refusals", {})]
        identities = [{"rank": 0}]
    if shapes is not None:
        phases_by_rank = {
            phase: [{name: {**record, "shape": shapes[phase]}
                     for name, record in per_rank.items()}
                    for per_rank in per_ranks]
            for phase, per_ranks in phases_by_rank.items()}
    owners = receipt.get("record_owner", {})
    targets = sorted({target for mapping in owners.values()
                      for target in mapping.values()})
    (family,) = list(receipt.get("declared_families", {}))
    declared = {target: family for target in targets}
    first = next(record for per_ranks in phases_by_rank.values()
                 for per_rank in per_ranks for record in per_rank.values())
    pairs = {(record["symbol"], record.get("decoder"))
             for per_ranks in phases_by_rank.values()
             for per_rank in per_ranks for record in per_rank.values()}
    checked = tool.validate_census_observations(
        phases_by_rank=phases_by_rank, identities=identities,
        refusals_by_rank=refusals, declared=declared, declared_rungs={},
        phase_plan=tool.census_phase_plan(None),
        mode=receipt.get("env", {}).get("TESSERA_SERVE_MODE", ""),
        platform=receipt["device"]["platform_token"],
        runtime_image=receipt["runtime"]["image"],
        execution_mode=receipt["runtime"]["execution_mode"],
        compiled=receipt["compiled"], compilation_config=compilation_config,
        cells=[],
        contract_for={family: first["contract"]},
        expected=lambda fam, regime, kind: set(pairs),
        symbol_for={family: first["symbol"]},
        symbol_base=lambda symbol: symbol,
        families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
        policy_prefixes=tuple(f"{name}:" for name in TESSERA_FAMILIES))
    return checked["problems"]


def test_retained_eager_receipt_stays_clean():
    """The retained eager control still validates with no problems."""
    receipt = json.loads(_retained("tp2-eager-served.json").read_text())
    assert receipt["problems"] == [] and receipt["verdict"] == "served"
    assert _replay(receipt, compilation_config=receipt.get("compilation_config")) == []


def test_retained_graph_receipt_refuses_without_m_star():
    """The retained capture is refused for its regime, not for M star."""
    receipt = json.loads(_retained("tp2-graph-served.json").read_text())
    assert "shape-polymorphic" in " ".join(receipt["problems"])
    problems = _replay(receipt, compilation_config=receipt.get("compilation_config"))
    joined = " ".join(problems)
    assert "shape-polymorphic" not in joined
    assert "does not exercise the declared regime" in joined
    assert "capture time" in joined and "no Python" in joined
    assert problems


def test_equal_capture_shapes_grant_no_logical_row_credit():
    """Even eager-matching concrete shapes state capture time only."""
    receipt = json.loads(_retained("tp2-graph-served.json").read_text())
    problems = _replay(
        receipt, compilation_config=receipt.get("compilation_config"),
        shapes={"prefill": "M64:N2048:K4096", "decode": "M1:N2048:K4096"})
    joined = " ".join(problems)
    assert "shape-polymorphic" not in joined
    assert "capture time" in joined and "no Python" in joined
    assert problems


def test_torch_compiled_receipt_still_requires_m_star():
    """The same concrete records under a real trace still need M star."""
    receipt = json.loads(_retained("tp2-graph-served.json").read_text())
    problems = _replay(
        receipt,
        compilation_config={"mode": 3, "cudagraph_mode": "FULL_DECODE_ONLY"})
    assert "shape-polymorphic" in " ".join(problems)
