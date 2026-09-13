"""The load refuses what the platform does not execute, and the record says where.

THE DEFECT THIS PINS (RobTand/tessera#457).  Contract v23 (#456/#464) gave
``lane_eligibility`` a platform axis, so the document can now say that
``TESSERA_E4M3_K1`` and ``TESSERA_E2M1_K2`` execute NOTHING on gfx1151 and
gfx1201.  Nothing read it at load.  An E4M3 artifact on an AMD box would have
walked all the way into ``fp8_route``, imported vLLM's quantizer namespace and
failed at an ABI probe -- a message about a missing operator, which is a true
statement about the wrong thing.  The absence is ATTESTED: the pinned runtime
publishes no native route for those bytes on that device, and that is what the
refusal has to say, before any HIP kernel is touched.

The second half is the observation.  A route record carried the family, the
symbol, the decoder, the contract and the shape -- every coordinate a
``lane_eligibility`` cell is keyed by EXCEPT the platform.  A gfx1151 serve and
an sm_121 serve of the same artifact wrote byte-identical records, so a census
could join either to the sm_121 cell and report agreement.  ``ROUTE_FIELDS``
gains ``platform``; ``emit_route`` stamps it; the cell join reads it.

No AMD hardware is needed and none is claimed: the platform is a stub, the
contract is the packaged one, and every assertion here is about what the code
decides, never about what a kernel computes.
"""
from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The autouse ``_unlatched_platform`` fixture below imports ``tessera.serving.
# telemetry`` for every test, and that module imports ``torch`` at load (as it
# has since 1c9b128f, on master too). So this whole file needs torch in the
# process; declare it the way every other torch-needing test module does
# (e.g. test_route_trace.py) so the bytes-only CI lane SKIPS it instead of
# erroring at fixture setup. The gate itself is exercised with torch present on
# the PrismaBuild suite and on a real gfx1201.
pytest.importorskip("torch")

from tessera.serving import backend as backend_module  # noqa: E402
from tessera.serving import census as census_module  # noqa: E402
from tessera.serving import contract as contract_module  # noqa: E402
from tessera.serving.ext import NativeKernelUnavailableError  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src" / "tessera"

#: What a HIP box's ``executes`` table says in v23, and what an NVIDIA box's
#: does.  Read off the packaged contract rather than restated, so a contract
#: that changes its mind fails here instead of passing against a copy.
AMD_PLATFORMS = ("gfx1151", "gfx1201")
QUANTIZED = ("TESSERA_E4M3_K1", "TESSERA_E2M1_K2")


@pytest.fixture(autouse=True)
def _unlatched_platform():
    """The telemetry stamp is a PROCESS constant, so a stub must not outlive
    its test: every load seam latches it, and a cached ``gfx1201`` would
    follow the stub into whatever ran next."""
    from tessera.serving import telemetry

    telemetry.reset_platform_for_tests()
    yield
    telemetry.reset_platform_for_tests()


def _stub_platform(monkeypatch, token, hip=True):
    """Make every reader of "which platform is this" answer ``token``.

    Patched at ``probed_platform_token``, which is the one function the gate,
    the telemetry stamp and the census tool all reach -- and deliberately NOT
    at ``platform_token``, whose answer ``TESSERA_PLATFORM_TOKEN`` can move.
    A build override must never decide what a serve is allowed to load
    (#452's ``test_the_override_never_moves_the_probed_token`` is the other
    half of that rule).
    """
    monkeypatch.setattr(backend_module, "probed_platform_token",
                        lambda device=0, torch=None: token)
    monkeypatch.setattr(backend_module, "backend",
                        lambda torch=None: "hip" if hip else "cuda")


# --------------------------------------------------------------------------
# the gate itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize("platform", AMD_PLATFORMS)
@pytest.mark.parametrize("family", QUANTIZED)
def test_a_quantized_family_is_refused_on_a_platform_that_executes_none(
        monkeypatch, platform, family):
    """The three things the message must carry: the contract's word, the
    platform token and the payload family.

    The FAMILY and not the route: a reader holding a checkpoint sees
    ``TESSERA_E4M3_K1`` in its config groups, and a message about
    ``TESSERA_FP8`` would leave them to map a route name back to the bytes.
    """
    _stub_platform(monkeypatch, platform)
    with pytest.raises(NativeKernelUnavailableError) as excinfo:
        backend_module.require_platform_backs(family, "tessera target 'model.layers.0.q'")
    message = str(excinfo.value)
    assert "unbacked" in message, message
    assert platform in message, message
    assert family in message, message
    assert "tessera target 'model.layers.0.q'" in message, message


@pytest.mark.parametrize("platform", AMD_PLATFORMS)
def test_the_sixteen_bit_family_loads_on_the_same_platform(monkeypatch, platform):
    """The AMD lane IS ``TESSERA_BF16_K1``: W16A16, no activation quantizer,
    no ``native_ops``.  A gate that refused it would refuse the only family
    the platform axis says these platforms execute."""
    _stub_platform(monkeypatch, platform)
    assert backend_module.require_platform_backs("TESSERA_BF16_K1", "ctx") is None


@pytest.mark.parametrize("family", QUANTIZED + ("TESSERA_BF16_K1",))
def test_nothing_changes_on_the_platform_the_contract_was_written_for(monkeypatch, family):
    _stub_platform(monkeypatch, "sm_121", hip=False)
    assert backend_module.require_platform_backs(family, "ctx") is None


@pytest.mark.parametrize("family", QUANTIZED + ("TESSERA_BF16_K1",))
def test_a_platform_the_table_does_not_name_refuses_nothing(monkeypatch, family):
    """``unstated`` is not a refusal.

    A silence in the document is not a claim about a runtime nobody read --
    principle 14 in the direction it is usually read backwards.  sm_90 is a
    real platform this contract has attested nothing about.
    """
    _stub_platform(monkeypatch, "sm_90", hip=False)
    assert backend_module.require_platform_backs(family, "ctx") is None


@pytest.mark.parametrize("family", QUANTIZED)
def test_a_contract_written_before_the_platform_axis_refuses_nothing(monkeypatch, family):
    """Every ``contract_version`` 22 document, on every platform.

    Without this the gate would refuse every serve the day it landed: v22
    carries no ``platforms`` table at all, so the honest state is ``unstated``
    and the honest behaviour is to change nothing.
    """
    _stub_platform(monkeypatch, "gfx1151")
    v22 = {"lane_eligibility": {"cells": []}}
    assert backend_module.require_platform_backs(family, "ctx", contract=v22) is None


@pytest.mark.parametrize("family", QUANTIZED)
def test_a_box_with_no_device_refuses_nothing(monkeypatch, family):
    """A CPU box names no platform, so it attests nothing and refuses nothing.

    This is what keeps every import-only and CPU test in the suite -- and a
    producer reading the contract without a GPU -- unchanged by the gate.
    """
    def no_device(device=0, torch=None):
        raise backend_module.PlatformTokenError("no device")

    monkeypatch.setattr(backend_module, "probed_platform_token", no_device)
    assert backend_module.platform_of_this_process() is None
    assert backend_module.require_platform_backs(family, "ctx") is None


def test_the_gate_reads_the_probed_token_and_not_the_build_override(monkeypatch):
    """``TESSERA_PLATFORM_TOKEN`` is a COMPILER's fact.

    It exists so a gfx1201 box can build a gfx1151 object; letting it reach
    this gate would let an environment variable decide which artifacts a serve
    accepts.  The probe says sm_121, the override says gfx1151, and the gate
    must follow the probe.
    """
    monkeypatch.setenv(backend_module.PLATFORM_TOKEN_ENV, "gfx1151")
    monkeypatch.setattr(backend_module, "probed_platform_token",
                        lambda device=0, torch=None: "sm_121")
    assert backend_module.require_platform_backs("TESSERA_E4M3_K1", "ctx") is None


# --------------------------------------------------------------------------
# where the gate is asked: the load seam, before the route module is imported
# --------------------------------------------------------------------------

def _dense_scheme(family, rows=256, columns=256):
    return {"family": family, "structure": "dense", "rows": rows, "columns": columns,
            "q256": 1024, "wire_bytes": 1}


@pytest.mark.parametrize("platform", AMD_PLATFORMS)
@pytest.mark.parametrize("route,family", [("TESSERA_FP8", "TESSERA_E4M3_K1"),
                                          ("TESSERA_NVFP4", "TESSERA_E2M1_K2")])
def test_the_dense_load_refuses_before_it_imports_the_route(monkeypatch, platform,
                                                            route, family):
    """``lane.build_tessera_method`` is what ``TesseraConfig.get_quant_method``
    calls, and the refusal happens before ``import_module``.

    Proved rather than asserted: ``importlib.import_module`` is replaced by a
    function that fails the test if it is reached.  A refusal that arrived
    after the import would have pulled in the route module, its extension
    loader and vLLM's quantizer namespace -- which on a HIP box is exactly the
    "crash three layers down" this gate replaces.
    """
    import importlib

    from tessera.serving import lane

    _stub_platform(monkeypatch, platform)

    def never(name, *args, **kwargs):
        raise AssertionError(f"the route module {name!r} was imported before the refusal")

    monkeypatch.setattr(importlib, "import_module", never)
    with pytest.raises(NativeKernelUnavailableError) as excinfo:
        lane.build_tessera_method(_dense_scheme(route), "model.layers.0.q_proj",
                                  mode="resident")
    assert "unbacked" in str(excinfo.value)
    assert family in str(excinfo.value)


@pytest.mark.parametrize("platform", AMD_PLATFORMS)
def test_the_sixteen_bit_load_reaches_its_route_on_the_same_platform(monkeypatch, platform):
    """The complement, and the half that says the gate is not a ban on AMD.

    The route module IS imported for ``TESSERA_BF16``; what happens after that
    is the BF16 route's business and needs vLLM, so this stops at the import
    -- which is the only thing the gate decides.
    """
    import importlib

    from tessera.serving import lane

    _stub_platform(monkeypatch, platform)
    reached = []

    def record(name, *args, **kwargs):
        reached.append(name)
        raise RuntimeError("stop here: the gate let this through")

    monkeypatch.setattr(importlib, "import_module", record)
    with pytest.raises(RuntimeError, match="stop here"):
        lane.build_tessera_method(_dense_scheme("TESSERA_BF16"), "model.layers.0.q_proj",
                                  mode="resident")
    assert reached == ["tessera.serving.bf16_route"]


def test_the_expert_route_asks_the_same_question_before_it_touches_vllm():
    """The MoE seam, read off the source because driving it needs vLLM.

    ``build_tessera_moe_method`` refuses in two directions and both are named:
    a 16-bit expert stack has no builder on ANY platform
    (``scheme.MOE_BUILDERS`` names only ``TESSERA_FP8``, and
    ``refuse_a_family_with_no_expert_route`` says so), and an FP8 expert stack
    on a platform that executes no E4M3 route is refused by the same gate the
    dense routes use.  What this test pins is the ORDER: both refusals precede
    the first ``from vllm...`` import in the function, which is what "before
    any HIP kernel is touched" means on this path.
    """
    tree = ast.parse((SRC / "serving" / "moe_route.py").read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "build_tessera_moe_method")
    gate = [node.lineno for node in ast.walk(function)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None)
            in ("require_platform_backs", "refuse_a_family_with_no_expert_route")]
    vllm = [node.lineno for node in ast.walk(function)
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm")]
    assert len(gate) == 2, "both expert-route refusals must be in this function"
    assert vllm, "the walk found no vLLM import to be earlier than"
    assert max(gate) < min(vllm), (gate, vllm)


# --------------------------------------------------------------------------
# the telemetry stamp
# --------------------------------------------------------------------------

def test_the_record_carries_the_platform_it_ran_on(monkeypatch):
    from tessera.serving import telemetry

    assert "platform" in telemetry.ROUTE_FIELDS
    monkeypatch.setattr(backend_module, "probed_platform_token",
                        lambda device=0, torch=None: "gfx1201")
    telemetry.reset_platform_for_tests()
    try:
        layer = types.SimpleNamespace()
        telemetry.emit_route(layer, kind="dense", policy="TESSERA_BF16:streamed",
                             symbol="torch.mm", shape="M1:N256:K256",
                             contract="bf16_unquantized", decoder="window_gemv")
        assert telemetry.read_route(layer)["platform"] == "gfx1201"
    finally:
        telemetry.reset_platform_for_tests()


def test_the_stamp_is_a_process_constant_and_never_a_probe_per_call(monkeypatch):
    """``emit_route`` runs on the forward, inside the traced body under
    ``torch.compile``.  A device probe there would be a CUDA/HIP call per
    module per forward on a path whose whole premise is that it touches no
    tensor -- and this surface has broken a serve before (#113).  One process
    serves one device, so the token is read once."""
    from tessera.serving import telemetry

    calls = []

    def probe(device=0, torch=None):
        calls.append(1)
        return "sm_121"

    monkeypatch.setattr(backend_module, "probed_platform_token", probe)
    telemetry.reset_platform_for_tests()
    try:
        layer = types.SimpleNamespace()
        for _ in range(5):
            telemetry.emit_route(layer, kind="dense", policy="TESSERA_BF16:resident",
                                 symbol="torch.mm", contract="bf16_unquantized")
        assert len(calls) == 1, calls
    finally:
        telemetry.reset_platform_for_tests()


def test_a_box_that_cannot_name_a_platform_stamps_nothing_and_still_serves(monkeypatch):
    """``""``, not a crash and not a guess: telemetry that can break a serve
    is not telemetry, and a record that invented a platform would be worse
    than one that says it does not know."""
    from tessera.serving import telemetry

    def boom(device=0, torch=None):
        raise RuntimeError("no device")

    monkeypatch.setattr(backend_module, "probed_platform_token", boom)
    telemetry.reset_platform_for_tests()
    try:
        layer = types.SimpleNamespace()
        telemetry.emit_route(layer, kind="dense", policy="TESSERA_BF16:resident",
                             symbol="torch.mm", contract="bf16_unquantized")
        assert telemetry.read_route(layer)["platform"] == ""
    finally:
        telemetry.reset_platform_for_tests()


# --------------------------------------------------------------------------
# the census join, per (platform, family)
# --------------------------------------------------------------------------

#: A digest-shaped image, because ``cell_runtime_scope`` requires one.  The
#: cell/record pair below is ``test_census_runtime_scope._case``'s shape with
#: one field added, so what is being varied here is the platform and nothing
#: else.
IMAGE = "example/runtime@sha256:" + "1" * 64


def _record(platform):
    record = {"kind": "dense", "policy": "TESSERA_FP8:resident",
              "symbol": "torch._scaled_mm", "shape": "M1:N4096:K4096",
              "decoder": "torch_materialize_stock"}
    if platform is not None:
        record["platform"] = platform
    return record


def _agreement(record_platform, census_platform="sm_121"):
    cell = {"id": "synthetic", "platform": census_platform, "structure": "dense",
            "family": "E4M3", "regime": "decode", "rungs_q256": [1024],
            "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
            "runtime": {"image": IMAGE, "execution_modes": ["eager"]},
            "executes": [{"symbol": "torch._scaled_mm",
                          "decoder": "torch_materialize_stock"}]}
    return census_module.cell_launch_agreement(
        {"decode": {"model.proj": _record(record_platform)}},
        cells=[cell], phase_regimes={"decode": "decode"}, platform=census_platform,
        structure="dense", rungs_by_module={"model.proj": 1024},
        families_by_route={"TESSERA_FP8": "E4M3"},
        runtime_image=IMAGE, execution_mode="eager")


def test_a_record_that_matches_the_censused_platform_is_checked_as_before():
    block, problems = _agreement("sm_121")
    assert problems == []
    assert block["agrees"] is True


def test_a_record_from_another_platform_is_a_disagreement_not_a_match():
    """The join this field exists to prevent.

    Without ``platform`` on the record, a gfx1151 serve wrote exactly the
    record an sm_121 serve writes, and the only statement of which platform
    served was the tool's argument -- read off the box the tool ran on.  A
    cell is keyed by platform; joining across that key attests a launch on
    hardware no cell in the contract covers.
    """
    block, problems = _agreement("gfx1151")
    assert len(problems) == 1, problems
    assert "gfx1151" in problems[0] and "sm_121" in problems[0]
    assert block["agrees"] is False
    assert block["phases"]["decode"]["covered_by_cell"] == 0
    assert block["phases"]["decode"]["unattested"] == 1


def test_a_record_written_before_the_stamp_is_joined_as_it_always_was():
    """Replayed receipts and every record older than #457 carry no platform.
    Absence is not disagreement; the field says nothing and the join is the
    one it was before."""
    block, problems = _agreement(None)
    assert problems == []
    assert block["agrees"] is True


# --------------------------------------------------------------------------
# the expectation, per (platform, family)
# --------------------------------------------------------------------------

def test_the_expectation_is_unchanged_when_no_platform_is_given():
    from tessera.serving import bf16_route, fp8_gemv

    for module in (bf16_route, fp8_gemv):
        assert module.census_expected(compiled=False, platform=None) == \
            module.census_expected(compiled=False)


@pytest.mark.parametrize("platform", AMD_PLATFORMS)
def test_a_platform_that_executes_no_e4m3_route_expects_no_e4m3_launch(platform):
    """Nothing of that family can load, so every regime's expectation is the
    empty set -- and a record that turns up anyway is a disagreement rather
    than something silently matched against the sm_121 pairs."""
    from tessera.serving import fp8_gemv, moe_route

    for module in (fp8_gemv, moe_route):
        expected = module.census_expected(compiled=False, platform=platform)
        assert expected, "the regimes themselves must survive"
        assert all(pairs == set() for pairs in expected.values()), expected


@pytest.mark.parametrize("platform", AMD_PLATFORMS)
def test_the_sixteen_bit_expectation_survives_on_the_same_platform(platform):
    """The asymmetry is the whole point: ``TESSERA_BF16_K1`` executes on both
    AMD platforms, so its expectation is untouched there."""
    from tessera.serving import bf16_route

    assert bf16_route.census_expected(compiled=False, platform=platform) == \
        bf16_route.census_expected(compiled=False)


def test_an_unstated_platform_narrows_nothing():
    from tessera.serving import fp8_gemv

    assert fp8_gemv.census_expected(compiled=False, platform="sm_90") == \
        fp8_gemv.census_expected(compiled=False)


# --------------------------------------------------------------------------
# one reader of the platform table
# --------------------------------------------------------------------------

def test_the_gate_and_the_abi_probe_raise_one_refusal(monkeypatch):
    """``native_ops.require_native_fp8_quant`` asks the same question the load
    seam asks, through the same function, so the two cannot word it
    differently -- and the ABI probe below it is never reached on a platform
    the contract has already refused."""
    from tessera.serving import native_ops

    _stub_platform(monkeypatch, "gfx1201")
    monkeypatch.setattr(native_ops, "_load_native_ops",
                        lambda context: pytest.fail("the ABI was probed after an "
                                                    "attested platform refusal"))
    with pytest.raises(NativeKernelUnavailableError, match="unbacked"):
        native_ops.require_native_fp8_quant("ctx")
    with pytest.raises(NativeKernelUnavailableError, match="unbacked"):
        native_ops.require_native_fp4_quant("ctx")


def test_the_packaged_contract_is_what_all_of_this_reads():
    """No fixture stands in for the document on the AMD half: the shipped
    ``runtime_contract.json`` is what the gate reads, and this is the
    statement it makes."""
    shipped = contract_module.cached_serving_contract()
    for platform in AMD_PLATFORMS:
        for family in QUANTIZED:
            assert contract_module.platform_execution_contract(family, platform, shipped) \
                == (contract_module.PLATFORM_UNBACKED, None)
        assert contract_module.platform_execution_contract(
            "TESSERA_BF16_K1", platform, shipped) == (
                contract_module.PLATFORM_BACKED, "bf16_unquantized")
