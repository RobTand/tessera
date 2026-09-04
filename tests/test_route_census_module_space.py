"""``tools/tessera_route_census.py`` joins its records in MODULE space.

The census reads a route record off every entry of ``named_modules()`` and
matches it to the families ``config_groups`` declares.  Those two are written
in different namespaces whenever the model class carries an
``hf_to_vllm_mapper``: the records are in the namespace vLLM built, the targets
in the checkpoint's.  Every census taken before these tests was on Qwen3-0.6B,
whose class declares no mapper -- so the two spaces coincided and the omission
could not show.  On ``Glm5NextForConditionalGeneration`` the mapper is
``{"model.language_model." -> "language_model.model.", ...}`` and nothing would
have joined, which the census would have reported as every served module
lacking a declaration: the opposite of what is true.

These pin the semantics of the census's USE of the mapper.  The mapper here is
a stub; that the replay matches vLLM's own ``WeightsMapper`` is attested
elsewhere (``tests/test_serving_name_mapping.py``, and the census asks the live
model class for its mapper rather than restating a table).
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

TOOL = pathlib.Path(__file__).resolve().parents[1] / "tools" / "tessera_route_census.py"


def _tool():
    spec = importlib.util.spec_from_file_location("tessera_route_census", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)      # its top level imports stdlib only
    return module


class _Unstacked:
    """The one method the census calls on vLLM's unstacked mapper."""

    def __init__(self, prefixes, drop=()):
        self._prefixes = dict(prefixes)
        self._drop = set(drop)

    def apply_list(self, names):
        out = []
        for name in names:
            if name in self._drop:
                continue
            for old, new in self._prefixes.items():
                if name.startswith(old):
                    name = new + name[len(old):]
                    break
            out.append(name)
        return out


class _Mapper:
    def __init__(self, unstacked):
        self._unstacked = unstacked

    def get_unstacked_mapper(self):
        return self._unstacked


class _Model:
    def __init__(self, mapper=None):
        if mapper is not None:
            self.hf_to_vllm_mapper = mapper


def test_a_model_with_no_mapper_reports_no_translation():
    """``None`` means checkpoint space IS module space -- Qwen3-0.6B's case."""
    assert _tool().declared_in_module_space(_Model(), ["model.layers.0.mlp.down_proj"]) is None


def test_a_mapped_architecture_translates_every_target():
    mapper = _Mapper(_Unstacked({"model.language_model.": "language_model.model.",
                                 "model.visual.": "visual."}))
    targets = ["model.language_model.layers.1.mlp.experts",
               "model.language_model.layers.0.mlp.down_proj",
               "model.visual.blocks.3.mlp.down_proj"]
    assert _tool().declared_in_module_space(_Model(mapper), targets) == {
        "model.language_model.layers.1.mlp.experts": "language_model.model.layers.1.mlp.experts",
        "model.language_model.layers.0.mlp.down_proj":
            "language_model.model.layers.0.mlp.down_proj",
        "model.visual.blocks.3.mlp.down_proj": "visual.blocks.3.mlp.down_proj"}


def test_a_target_the_mapper_drops_is_reported_as_no_module():
    """A dropped target is not the identity: the runtime builds nothing for it."""
    dead = "model.dead.layers.0.proj"
    mapper = _Mapper(_Unstacked({"model.": "lm."}, drop=[dead]))
    got = _tool().declared_in_module_space(_Model(mapper), [dead, "model.layers.0.proj"])
    assert got[dead] is None
    assert got["model.layers.0.proj"] == "lm.layers.0.proj"


@pytest.mark.parametrize("target", ["Linear", "re:.*down_proj"])
def test_a_class_name_or_a_regex_target_is_left_alone(target):
    """compressed-tensors' own target shapes: only a dotted path is a path."""
    mapper = _Mapper(_Unstacked({"model.": "lm."}))
    assert _tool().declared_in_module_space(_Model(mapper), [target]) == {target: target}


# --- and the second half of the join: which DECLARATION a record belongs to ---
#
# Module space alone is not enough for an expert stack.  The checkpoint
# declares ``<layer>.mlp.experts``; vLLM builds the quant method for that
# prefix and attaches it to the ``RoutedExperts`` child it constructs
# underneath, so the record is read at ``<layer>.mlp.experts.routed_experts``.
# The first routed-MoE census (2026-09-04, `docs/measurements/
# tessera-moe-served-2026-09-04.md`) refused on six problems that were three
# names: each stack counted once as a route nothing declared and once as a
# declaration nothing served, while its record said `state: "served"`.

DENSE = "language_model.model.layers.0.mlp.down_proj"
STACK = "language_model.model.layers.1.mlp.experts"


def test_a_dense_record_joins_to_its_own_name():
    owner, problems = _tool().join_records_to_declared(
        {DENSE: {"kind": "dense"}}, {DENSE: "TESSERA_FP8"})
    assert owner == {DENSE: DENSE} and problems == []


def test_an_expert_record_joins_to_the_stack_that_declares_it():
    child = f"{STACK}.routed_experts"
    owner, problems = _tool().join_records_to_declared(
        {child: {"kind": "moe"}, DENSE: {"kind": "dense"}},
        {STACK: "TESSERA_FP8", DENSE: "TESSERA_FP8"})
    assert owner == {child: STACK, DENSE: DENSE}
    assert problems == []


def test_the_joined_stack_is_no_longer_missing_and_no_longer_undeclared():
    """The two halves of the defect, stated as the census states them."""
    tool = _tool()
    child = f"{STACK}.routed_experts"
    records, declared = {child: {"kind": "moe"}}, {STACK: "TESSERA_FP8"}
    owner, _ = tool.join_records_to_declared(records, declared)
    assert declared.get(owner.get(child, child)) == "TESSERA_FP8"   # not "declares none"
    assert sorted(set(declared) - set(owner.values())) == []        # not "reports no route"


def test_a_non_expert_record_under_a_declared_target_does_not_join():
    """The rule is narrow on purpose: ``kind`` is the record's own word.

    Anything else nested under a declared module is a route the checkpoint
    genuinely declares nothing for, and saying so is the census's job.
    """
    child = f"{STACK}.something_else"
    owner, problems = _tool().join_records_to_declared(
        {child: {"kind": "dense"}}, {STACK: "TESSERA_FP8"})
    assert owner == {} and problems == []


def test_an_expert_record_under_two_declarations_is_ambiguous_not_guessed():
    inner = f"{STACK}.inner"
    child = f"{inner}.routed_experts"
    owner, problems = _tool().join_records_to_declared(
        {child: {"kind": "moe"}}, {STACK: "TESSERA_FP8", inner: "TESSERA_FP8"})
    assert owner == {}
    assert len(problems) == 1 and "belongs to one declaration" in problems[0]


def test_a_record_that_is_its_own_declaration_wins_over_containment():
    """A stack that carries its own record joins to itself, not to a parent."""
    owner, problems = _tool().join_records_to_declared(
        {STACK: {"kind": "moe"}}, {STACK: "TESSERA_FP8"})
    assert owner == {STACK: STACK} and problems == []


def test_moe_cell_rung_requires_agreement_of_every_group_role():
    tool = _tool()
    scheme = {"structure": "routed_moe", "groups": {
        "w13": {"q256": [1024, 1024]}, "w2": {"q256": 1024}}}
    assert tool.declared_rung(scheme) == 1024
    scheme["groups"]["w2"]["q256"] = 896
    assert tool.declared_rung(scheme) is None
    del scheme["groups"]["w2"]["q256"]
    assert tool.declared_rung(scheme) is None


def _moe_agreement_fixture(symbol="vllm.fused_moe.modular_kernel:TRITON"):
    image = "example/runtime@sha256:" + "1" * 64
    child = f"{STACK}.routed_experts"
    records = {"decode": {child: {"kind": "moe", "policy": "TESSERA_FP8:resident",
        "symbol": symbol, "decoder": "torch_materialize_stock"}}}
    cell = {"id": "synthetic_moe_cell", "platform": "sm_121", "structure": "routed_moe",
        "family": "E4M3", "regime": "decode", "rungs_q256": [1024],
        "runtime": {"image": image, "execution_modes": ["eager"]},
        "requires_serve_flags": ["TESSERA_SERVE_MODE=resident"],
        "executes": [{"symbol": "vllm.fused_moe.modular_kernel",
                      "decoder": "torch_materialize_stock"}]}
    kwargs = {"phase_regimes": {"decode": "decode"}, "platform": "sm_121",
        "declared_rungs": {STACK: 1024}, "record_owners": {"decode": {child: STACK}},
        "families_by_route": {"TESSERA_FP8": "E4M3"},
        "runtime_image": image, "execution_mode": "eager"}
    return records, cell, kwargs


def test_moe_agreement_joins_the_child_to_its_declared_rung_and_structure():
    records, cell, kwargs = _moe_agreement_fixture()
    records["decode"][DENSE] = {"kind": "dense", "policy": "TESSERA_FP8:resident",
        "symbol": "torch._scaled_mm", "decoder": "torch_materialize_stock"}
    dense_cell = dict(cell, id="synthetic_dense_cell", structure="dense",
                      executes=[{"symbol": "torch._scaled_mm",
                                 "decoder": "torch_materialize_stock"}])
    kwargs["declared_rungs"][DENSE] = 1024
    kwargs["record_owners"]["decode"][DENSE] = DENSE
    block, problems = _tool().all_structure_agreement(records, cells=[cell, dense_cell], **kwargs)
    assert problems == []
    assert block["agrees"] is True
    assert block["structures"]["routed_moe"]["phases"]["decode"]["covered_by_cell"] == 1
    assert block["structures"]["dense"]["phases"]["decode"]["covered_by_cell"] == 1


def test_a_moe_cell_launch_disagreement_is_a_failure():
    records, cell, kwargs = _moe_agreement_fixture("wrong.kernel")
    block, problems = _tool().all_structure_agreement(records, cells=[cell], **kwargs)
    assert block["agrees"] is False
    assert len(problems) == 1 and "wrong.kernel" in problems[0]
    records, cell, kwargs = _moe_agreement_fixture()
    cell["executes"][0]["symbol"] = "vllm.fused_moe.modular_kernel:OTHER_BACKEND"
    block, problems = _tool().all_structure_agreement(records, cells=[cell], **kwargs)
    assert block["agrees"] is False and len(problems) == 1


def test_dense_cells_do_not_attest_moe_and_missing_owners_stay_unattested():
    records, cell, kwargs = _moe_agreement_fixture()
    cell["structure"] = "dense"
    block, problems = _tool().all_structure_agreement(records, cells=[cell], **kwargs)
    assert problems == [] and block["agrees"] is None
    assert block["structures"]["routed_moe"]["phases"]["decode"]["unattested"] == 1
    cell["structure"] = "routed_moe"
    kwargs["record_owners"] = {"decode": {}}
    block, problems = _tool().all_structure_agreement(records, cells=[cell], **kwargs)
    assert problems == [] and block["agrees"] is None
