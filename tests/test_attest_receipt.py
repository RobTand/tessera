"""The certification receipt: its schema, and the claim it must refuse.

``tools/tessera_attest.py`` exists so that a run on somebody else's Strix Halo
can become a ``lane_eligibility`` cell -- and so that a run on a box that is
NOT a Strix Halo cannot.  Everything checked here is the second half: the
scope a token selects, the qualification a partial run withholds, and the
refusal to mint ``device_qualified`` for a platform the harness did not run
on.  None of it needs a GPU, which is the point: the rule that decides what a
receipt may claim must be exercisable on a box with no device at all.

The device is a stub throughout.  Real identity comes from torch's
``gcnArchName`` and from nothing else -- under WSL2 ``amdsmi`` cannot
initialise, ``rocm-smi --json`` prints nothing, and the vLLM ROCm platform
helpers are backed by amdsmi -- so the read is a parameterised function and
the test passes it an object shaped like ``torch``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "tessera_attest.py"
PROTOCOL = ROOT / "docs" / "strix-halo-tester-protocol.md"


def _module():
    spec = importlib.util.spec_from_file_location("_tessera_attest", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


attest = _module()


class _Props:
    """What ``torch.cuda.get_device_properties(0)`` answers on a ROCm build."""

    def __init__(self, gcn="gfx1151:sramecc+:xnack-", **rest):
        self.gcnArchName = gcn
        self.name = "AMD Radeon Graphics"
        self.warp_size = 32
        self.multi_processor_count = 40
        self.total_memory = 128 * 1024 ** 3
        self.shared_memory_per_block = 65536
        for key, value in rest.items():
            setattr(self, key, value)


class _Torch:
    """A stand-in for the module, not for a device: only the read is pinned."""

    def __init__(self, *, hip="7.2.4", cuda=None, available=True, props=None):
        self.__version__ = "2.11.0+rocm7.2.4" if hip else "2.13.0+cu130"
        self.version = type("v", (), {"hip": hip, "cuda": cuda})()
        outer = self

        class _Cuda:
            @staticmethod
            def is_available():
                return available

            @staticmethod
            def device_count():
                return 1 if available else 0

            @staticmethod
            def get_device_properties(index):
                return outer._props

        self._props = props if props is not None else _Props()
        self.cuda = _Cuda()


def _ran(*names):
    return {name: attest.step(attest.STATUS_RAN) for name in names}


def _all_steps_ran():
    return _ran(*(attest.REQUIRED_CODE_PATH_STEPS + attest.REQUIRED_DEVICE_PERF_STEPS
                  + (attest.STEP_DEVICE_IDENTITY, attest.STEP_PACKAGE_POWER)))


# --- identity -------------------------------------------------------------

def test_the_platform_token_is_the_arch_name_without_its_build_features():
    """``gfx1201:sramecc+:xnack-`` names one part; the suffixes are the build.

    A receipt keyed on the full string joins to no contract platform entry,
    and the suffixes differ between two boxes holding the same silicon.
    """
    assert attest.platform_token_of("gfx1201:sramecc+:xnack-") == "gfx1201"
    assert attest.platform_token_of("gfx1151") == "gfx1151"
    assert attest.cuda_platform_token(12, 1) == "sm_121"


def test_an_arch_name_that_is_not_a_gfx_token_is_an_error_not_a_truncation():
    with pytest.raises(ValueError, match="gcnArchName"):
        attest.platform_token_of("NVIDIA GB10")


def test_identity_is_read_from_gcn_arch_name_and_says_so():
    identity = attest.probe_identity(_Torch())
    assert identity["backend"] == "hip"
    assert identity["platform"] == "gfx1151"
    assert identity["gcn_arch_name"] == "gfx1151:sramecc+:xnack-"
    assert identity["source"] == "torch.cuda.get_device_properties(0).gcnArchName"
    assert identity["warp_size"] == 32 and identity["shared_memory_per_block"] == 65536


def test_a_torch_with_no_device_reports_unavailable_rather_than_raising():
    identity = attest.probe_identity(_Torch(available=False))
    assert identity["available"] is False and identity["platform"] is None
    assert identity["reason"]


# --- the §8.1 rows --------------------------------------------------------

def test_the_scope_row_is_derived_from_the_token_and_gfx1201_admits_no_perf():
    """The row is what separates a code-path receipt from a hardware claim."""
    assert attest.scope_for("gfx1151") == attest.SCOPE_DEVICE_STRIX_HALO
    assert attest.scope_for("gfx1201") == attest.SCOPE_EXECUTE_GFX12
    assert attest.scope_for("gfx1151", executed=False) == attest.SCOPE_COMPILE_GFX1151
    assert attest.SCOPES[attest.SCOPE_EXECUTE_GFX12]["perf_claim"] is False
    assert attest.SCOPES[attest.SCOPE_COMPILE_GFX1151]["perf_claim"] is False
    assert attest.SCOPES[attest.SCOPE_DEVICE_STRIX_HALO]["perf_claim"] is True


def test_a_cuda_platform_has_no_row_in_this_harness():
    with pytest.raises(ValueError, match="AMD certification harness"):
        attest.scope_for("sm_121")


# --- what may be claimed --------------------------------------------------

def test_the_qualification_vocabulary_is_the_contracts_own():
    """A word ``validate_serving_contract`` would refuse is a word a cell
    cannot carry, so the harness reads the set rather than restating it."""
    from tessera.serving import contract

    assert {attest.QUALIFICATION_DEVICE, attest.QUALIFICATION_COMPILE_ONLY} == set(
        contract._QUALIFICATIONS)


def test_a_full_run_on_the_platform_it_claims_is_device_qualified():
    qualification, problems = attest.qualification_for(
        claim_platform="gfx1151", measured_platform="gfx1151",
        scope=attest.SCOPE_DEVICE_STRIX_HALO, steps=_all_steps_ran())
    assert qualification == attest.QUALIFICATION_DEVICE
    assert problems == []


def test_it_refuses_to_mint_device_qualified_for_a_platform_it_did_not_run_on():
    """The failure the whole protocol exists to prevent: a gfx1151 cell minted
    from a gfx1201 run.  Every step ran, and the claim is still withheld."""
    qualification, problems = attest.qualification_for(
        claim_platform="gfx1151", measured_platform="gfx1201",
        scope=attest.SCOPE_DEVICE_STRIX_HALO, steps=_all_steps_ran())
    assert qualification is None
    assert any("gfx1151" in p and "gfx1201" in p for p in problems)


def test_a_missing_step_withholds_the_claim_and_names_it():
    steps = _all_steps_ran()
    steps[attest.STEP_SERVED_KL] = attest.step(attest.STATUS_NOT_RUN, "no --kl-receipt")
    qualification, problems = attest.qualification_for(
        claim_platform="gfx1151", measured_platform="gfx1151",
        scope=attest.SCOPE_DEVICE_STRIX_HALO, steps=steps)
    assert qualification is None
    assert any(attest.STEP_SERVED_KL in p for p in problems)


def test_a_build_with_no_device_is_compile_only():
    qualification, _ = attest.qualification_for(
        claim_platform=None, measured_platform=None,
        scope=attest.SCOPE_COMPILE_GFX1151,
        steps=_ran(attest.STEP_EXTENSION_BUILD))
    assert qualification == attest.QUALIFICATION_COMPILE_ONLY


def test_the_perf_half_is_required_only_where_a_perf_number_is_admissible():
    """On gfx1201 the tok/s steps cannot be quoted, so they cannot be required;
    the code-path half still is."""
    steps = _ran(*(attest.REQUIRED_CODE_PATH_STEPS + (attest.STEP_DEVICE_IDENTITY,)))
    qualification, _ = attest.qualification_for(
        claim_platform="gfx1201", measured_platform="gfx1201",
        scope=attest.SCOPE_EXECUTE_GFX12, steps=steps)
    assert qualification == attest.QUALIFICATION_DEVICE
    qualification, problems = attest.qualification_for(
        claim_platform="gfx1151", measured_platform="gfx1151",
        scope=attest.SCOPE_DEVICE_STRIX_HALO, steps=steps)
    assert qualification is None
    assert any(attest.STEP_DECODE_TOK_S in p for p in problems)


# --- the receipt ----------------------------------------------------------

def _receipt(**overrides):
    identity = overrides.pop("identity", attest.probe_identity(_Torch()))
    kwargs = dict(
        identity=identity,
        steps=_all_steps_ran(),
        power={"tool": "amd-smi", "available": False, "reason": "amd-smi is not on PATH",
               "interval_s": 1.0, "samples": 0, "mean_watts": None, "max_watts": None},
        reference={"family": attest.REFERENCE_FAMILY, "rungs_q256": [896, 1024],
                   "reference_rungs_q256": list(attest.REFERENCE_RUNGS_Q256)},
        versions={"torch": "2.11.0+rocm7.2.4", "vllm": "0.30.0", "hip": "7.2.4",
                  "cuda": None, "tessera": "0.1.0", "python": "3.12.0"},
        image="vllm/vllm-openai@sha256:" + "0" * 64,
    )
    kwargs.update(overrides)
    return attest.build_receipt(**kwargs)


def test_the_receipt_carries_the_header_the_protocol_requires():
    receipt = _receipt()
    assert receipt["schema"] == attest.SCHEMA
    header = receipt["header"]
    assert {"platform", "image", "versions", "qualification", "scope"} <= set(header)
    assert header["platform"] == "gfx1151"
    assert header["measured_platform"] == "gfx1151"
    assert header["scope"] == attest.SCOPE_DEVICE_STRIX_HALO
    assert header["scope_sentence"].startswith("scope: ")
    assert header["qualification"] == attest.QUALIFICATION_DEVICE
    # The versions a cell's reader needs (contract.RUNTIME_VERSION_KEYS).
    from tessera.serving import contract

    assert set(contract.RUNTIME_VERSION_KEYS) <= set(header["versions"])
    # It must round-trip as JSON: a tester mails this file.
    json.loads(json.dumps(receipt))


def test_both_protocol_sections_are_emitted_over_one_step_map():
    """§8.3 item 3 IS §8.2 items 2-4; two copies could disagree about what ran."""
    receipt = _receipt()
    assert set(receipt["sections"]) == {"8.2", "8.3"}
    assert set(receipt["sections"]["8.2"]) == {"1", "2", "3", "4", "5"}
    assert set(receipt["sections"]["8.3"]) == {"1", "2", "3", "4", "5", "6"}
    assert receipt["sections"]["8.3"]["3"]["refers_to"] == [
        attest.STEP_DECODER_BIT_EXACT, attest.STEP_GEMV_TOLERANCE, attest.STEP_ROUTE_CENSUS]
    steps = _all_steps_ran()
    steps[attest.STEP_GEMV_TOLERANCE] = attest.step(attest.STATUS_FAILED, "tolerance")
    assert _receipt(steps=steps)["sections"]["8.3"]["3"]["status"] == attest.STATUS_FAILED


def test_a_receipt_for_another_platform_is_written_with_no_qualification():
    """A refused claim is evidence too: the file is still produced, and it says
    which platform it was measured on."""
    receipt = _receipt(identity=attest.probe_identity(_Torch(props=_Props("gfx1201"))),
                       claim_platform="gfx1151")
    assert receipt["header"]["platform"] == "gfx1151"
    assert receipt["header"]["measured_platform"] == "gfx1201"
    assert receipt["header"]["qualification"] is None
    assert receipt["problems"]


def test_a_gfx1201_receipt_states_that_it_admits_no_performance_number():
    receipt = _receipt(identity=attest.probe_identity(_Torch(props=_Props("gfx1201"))))
    assert receipt["header"]["perf_claim"] is False
    assert any("no performance number" in note for note in receipt["limitations"])


def test_absent_power_is_a_field_and_never_an_exception():
    """A correctness receipt must not be lost because a power meter was missing;
    under WSL2 there is no driver behind ``amd-smi`` at all."""
    power = attest.PackagePower(binary="amd-smi-that-is-not-installed")
    with power:
        pass
    block = power.block()
    assert block["available"] is False and "not on PATH" in block["reason"]
    assert any("package power unavailable" in note for note in _receipt(power=block)["limitations"])


# --- the reference set ----------------------------------------------------

def _manifest(grid="BF16", q256=896):
    return {"modules": {"model.layers.0.mlp.down_proj": {
        "roles": [{"tensor": "w", "grid": grid, "q256": q256, "wire_bytes": 16}]}}}


def test_the_reference_set_is_tessera_16_at_896_and_1024():
    report = attest.reference_set_report(_manifest(q256=1024))
    assert report["family"] == attest.REFERENCE_FAMILY == "TESSERA_BF16_K1"
    assert report["rungs_q256"] == [1024]
    assert report["reference_rungs_q256"] == [896, 1024]


def test_a_rung_outside_the_reference_set_is_refused_not_measured():
    with pytest.raises(SystemExit, match="1792"):
        attest.reference_set_report(_manifest(q256=1792))


def test_a_family_outside_the_lane_is_refused():
    with pytest.raises(SystemExit, match="TESSERA_BF16_K1"):
        attest.reference_set_report(_manifest(grid="E4M3", q256=1024))


# --- the doc and the tool say the same thing ------------------------------

def test_the_tester_protocol_reproduces_the_scope_rule_verbatim():
    """The scope rule goes on every receipt AND in the tester's instructions.
    Two copies of one sentence drift; this is what stops them."""
    text = PROTOCOL.read_text(encoding="utf-8")
    for scope, row in attest.SCOPES.items():
        assert row["receipt"] in text, f"{scope}: the doc does not name this receipt"
        assert row["proves"] in text, f"{scope}: the doc does not state what it proves"
        assert row["does_not_prove"] in text, f"{scope}: the doc omits what it does not prove"


def test_the_head_to_head_is_marked_as_needing_hardware_we_do_not_have():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "requires hardware we do not have" in text
    assert "EXL3" in text


def test_the_placeholder_numbers_are_marked_unmeasured():
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "unmeasured" in text
    for placeholder in ("256 GB/s", "40 CUs"):
        assert placeholder in text
