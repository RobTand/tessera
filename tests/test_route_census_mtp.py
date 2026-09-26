"""CPU contract for observing the stock GLM MTP draft, without a model launch."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


TOOL = Path(__file__).resolve().parents[1] / "tools" / "tessera_route_census.py"
spec = importlib.util.spec_from_file_location("mtp_route_census_tool", TOOL)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


class _Mapper:
    def apply_list(self, names):
        return [name.replace("model.language_model.", "model.", 1)
                if name.startswith("model.language_model.") else name for name in names]


class Glm5NextMTP:
    __module__ = "vllm.models.glm5next.nvidia.mtp"
    hf_to_vllm_mapper = _Mapper()

    def __init__(self):
        self.config = SimpleNamespace(num_hidden_layers=45, num_nextn_predict_layers=1)

    def named_modules(self):
        yield "", self
        yield "model.layers.45.mtp_block.mlp.experts", SimpleNamespace()


def test_mtp_declared_mapping_uses_actual_draft_module_and_preserves_source_name():
    source = "model.language_model.layers.45.mlp.experts"
    assert census.draft_declared_in_module_space(Glm5NextMTP(), [source]) == {
        source: "model.layers.45.mtp_block.mlp.experts"}


def test_mtp_mapping_refuses_another_class_or_missing_module():
    class Other(Glm5NextMTP):
        pass

    source = "model.language_model.layers.45.mlp.experts"
    with pytest.raises(ValueError, match="Glm5NextMTP"):
        census.draft_declared_in_module_space(Other(), [source])
    draft = Glm5NextMTP()
    draft.named_modules = lambda: iter([("", draft)])
    with pytest.raises(ValueError, match="no module"):
        census.draft_declared_in_module_space(draft, [source])


def test_speculative_config_is_explicit_mtp_one_step_and_same_tp_world():
    value = census.parse_mtp_speculative_config(
        '{"method":"mtp","num_speculative_tokens":1,"draft_tensor_parallel_size":2}',
        world=2)
    assert value == {"method": "mtp", "num_speculative_tokens": 1,
                     "draft_tensor_parallel_size": 2}
    for raw in (None, '{}', '{"method":"ngram","num_speculative_tokens":1}',
                '{"method":"mtp","num_speculative_tokens":2}',
                '{"method":"mtp","num_speculative_tokens":1,"draft_tensor_parallel_size":1}'):
        with pytest.raises(ValueError):
            census.parse_mtp_speculative_config(raw, world=2)


def test_draft_inventory_reads_only_the_draft_and_returns_no_model(monkeypatch):
    draft = Glm5NextMTP()
    target = object()
    class Worker:
        def get_model(self):
            raise AssertionError("draft inventory touched target")

        def get_draft_model(self):
            return draft

    monkeypatch.setattr(census, "census", lambda model: {"draft": {"state": "served"}})
    monkeypatch.setattr(census, "rank_identity", lambda model: {"rank": 0, "world_size": 1})
    monkeypatch.setattr(census, "lane_refusals", lambda model: {})
    source = "model.language_model.layers.45.mlp.experts"
    result = census.draft_worker_inventory(Worker(), [source])
    assert result["name_mapping"] == {source: "model.layers.45.mtp_block.mlp.experts"}
    assert result["records"] == {"draft": {"state": "served"}}
    assert result["identity"] == {"rank": 0, "world_size": 1}
    assert target not in result.values() and draft not in result.values()


def test_partition_uses_original_glm_spec_layer_range_without_renaming():
    config = {"model_type": "glm5_next",
              "architectures": ["Glm5NextForConditionalGeneration"],
              "text_config": {"num_hidden_layers": 45,
                              "num_nextn_predict_layers": 1}}
    body = "model.language_model.layers.44.mlp.experts"
    mtp = "model.language_model.layers.45.mlp.experts"
    got_body, got_draft = census.partition_glm_mtp_targets(
        config, {body: "BF16", mtp: "E4M3"})
    assert got_body == {body: "BF16"}
    assert got_draft == {mtp: "E4M3"}
    with pytest.raises(ValueError, match="architecture"):
        census.partition_glm_mtp_targets(
            {**config, "architectures": ["AnotherModel"]},
            {body: "BF16", mtp: "E4M3"})


def test_draft_census_clear_discards_stale_observations_only(monkeypatch):
    pytest.importorskip("torch")  # telemetry imports torch; the pure CI has no torch
    draft = Glm5NextMTP()
    module = SimpleNamespace(_tessera_route_state="served", packed_weight="keep")
    draft.named_modules = lambda: iter([("model.layers.45.mtp_block.mlp.experts", module)])
    class Worker:
        def get_draft_model(self):
            return draft

    assert census.clear_draft_route_records(Worker()) == 1
    assert not hasattr(module, "_tessera_route_state")
    assert module.packed_weight == "keep"


def _draft_observation(rank=0, *, world=1, refusals=None, records=None):
    source = "model.language_model.layers.45.mlp.experts"
    module = "model.layers.45.mtp_block.mlp.experts"
    return {"identity": {"rank": rank, "world_size": world,
                         "node": "spark", "device": "gb10",
                         "platform_token": "sm_121", "runtime_image": "image"},
            "name_mapping": {source: module},
            "lane_refusals": refusals or {},
            "records": records if records is not None else {
                module + ".routed_experts": {
                    "kind": "moe", "state": "served", "policy": "TESSERA_E4M3:resident",
                    "contract": "c", "symbol": "native", "decoder": "native",
                    "shape": [1, 2048, 4096, 2048]}}}


def _evaluate(observations):
    source = "model.language_model.layers.45.mlp.experts"
    module = "model.layers.45.mtp_block.mlp.experts"
    return census.validate_draft_route_records(
        {"decode": observations}, source_to_module={source: module},
        draft_declared={source: "TESSERA_E4M3"},
        target_identities=[_draft_observation(rank, world=len(observations))["identity"]
                           for rank in range(len(observations))],
        phase_regimes={"decode": "decode"}, mode="resident",
        policy_prefixes=("TESSERA_E4M3:",),
        contract_for={"TESSERA_E4M3": "c"},
        expected=lambda family, regime, kind: {("native", "native")},
        symbol_base=lambda symbol: symbol,
        shape_problem=lambda shape, regime: None)


def test_draft_validation_refuses_duplicate_rank_even_when_set_is_complete():
    result = _evaluate([_draft_observation(0, world=3),
                        _draft_observation(0, world=3),
                        _draft_observation(1, world=3)])
    assert any("duplicate" in problem for problem in result[4])


def test_draft_validation_refuses_bool_or_string_rank():
    for rank in (True, "0"):
        result = _evaluate([_draft_observation(rank)])
        assert any("rank identities" in problem for problem in result[4])


def test_draft_validation_refuses_lane_failure_despite_clean_route_record():
    result = _evaluate([_draft_observation(refusals={"model.layers.45.mtp_block.mlp.experts":
                                                      "resident lane unavailable"})])
    assert any("lane refused" in problem for problem in result[4])


def test_draft_validation_refuses_absent_fresh_route():
    result = _evaluate([_draft_observation(records={})])
    assert any("no MTP module reports" in problem for problem in result[4])


def test_draft_validation_preserves_original_owner_and_actual_module():
    records, owners, by_rank, identities, problems, refusals = _evaluate(
        [_draft_observation()])
    source = "model.language_model.layers.45.mlp.experts"
    module = "model.layers.45.mtp_block.mlp.experts.routed_experts"
    assert problems == []
    assert owners["decode"][module] == "model.layers.45.mtp_block.mlp.experts"
    assert module in records["decode"] and module in by_rank[0]["decode"]
    assert identities[0]["rank"] == 0 and refusals[0]["decode"] == {}
    assert source not in owners["decode"].values()


def test_text_only_scope_is_explicit_and_does_not_change_default_engine_kwargs():
    assert census.engine_scope_kwargs(SimpleNamespace(language_model_only=False)) == {}
    assert census.engine_scope_kwargs(SimpleNamespace(language_model_only=True)) == {
        "language_model_only": True}


def _forward_observer_fixture(monkeypatch):
    torch = pytest.importorskip("torch")

    class Draft(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.route_records = {"old": {"policy": "TESSERA_E4M3:resident",
                                          "shape": "M64:N1:K1"}}

        def forward(self, m, *, mixed=False):
            self.route_records = {"mtp": {"policy": "TESSERA_E4M3:resident",
                                          "shape": f"M{m}:N1:K1"}}
            if mixed:
                self.route_records["other"] = {
                    "policy": "TESSERA_E4M3:resident", "shape": "M1:N1:K1"}
            return m

    draft = Draft()
    worker = SimpleNamespace(get_draft_model=lambda: draft)
    monkeypatch.setattr(census, "draft_declared_in_module_space",
                        lambda model, targets: {"source": "mtp"})
    monkeypatch.setattr(census, "census", lambda model: dict(model.route_records))
    monkeypatch.setattr(census, "clear_draft_route_records",
                        lambda worker: worker.get_draft_model().route_records.clear())
    monkeypatch.setattr(census, "rank_identity",
                        lambda model: {"rank": 0, "world_size": 1})
    monkeypatch.setattr(census, "lane_refusals", lambda model: {})
    return worker, draft


def test_forward_observer_keeps_first_batch_and_latest_decode(monkeypatch):
    worker, draft = _forward_observer_fixture(monkeypatch)
    census.arm_draft_forward_observer(worker, ["source"], ("TESSERA_E4M3:",))
    draft(64)
    draft(1)
    result = census.disarm_draft_forward_observer(worker)
    assert result["calls"] == 2
    assert result["by_regime"]["batch"]["first"]["mtp"]["shape"] == "M64:N1:K1"
    assert result["by_regime"]["decode"]["latest"]["mtp"]["shape"] == "M1:N1:K1"
    assert result["unclassified_calls"] == 0 and result["mixed_shape_calls"] == 0
    assert not draft._forward_hooks and not draft._forward_pre_hooks
    batch = census.draft_observations_from_forward([result], "batch", snapshot="first")
    decode = census.draft_observations_from_forward([result], "decode", snapshot="latest")
    assert batch[0]["records"]["mtp"]["shape"] == "M64:N1:K1"
    assert decode[0]["records"]["mtp"]["shape"] == "M1:N1:K1"


def test_forward_observer_discards_stale_record_and_refuses_mixed_m(monkeypatch):
    worker, draft = _forward_observer_fixture(monkeypatch)
    census.arm_draft_forward_observer(worker, ["source"], ("TESSERA_E4M3:",))
    draft(4, mixed=True)
    result = census.disarm_draft_forward_observer(worker)
    assert "batch" not in result["by_regime"]
    assert result["mixed_shape_calls"] == 1
    assert result["calls"] == 1


def test_forward_observer_removes_hooks_after_failing_generation(monkeypatch):
    worker, draft = _forward_observer_fixture(monkeypatch)
    with pytest.raises(RuntimeError, match="generation failed"):
        census.arm_draft_forward_observer(worker, ["source"], ("TESSERA_E4M3:",))
        try:
            raise RuntimeError("generation failed")
        finally:
            census.disarm_draft_forward_observer(worker)
    assert not draft._forward_hooks and not draft._forward_pre_hooks
    assert not hasattr(worker, "_tessera_draft_forward_observer")
