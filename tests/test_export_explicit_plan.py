"""An explicit plan entry is an obligation the direct export implements or refuses (#211).

The direct (non-partitioned) serving export used to demote an explicit
quantized target to BF16 passthrough in silence -- on a shape the grid cannot
cut, on the ``--layers`` smoke bound, on fused-group disagreement, and under
``--passthrough-unrouted`` -- and then published ``config.json`` without ever
checking the emitted roles against the plan.  The whole-layer merge path
already refuses through ``serving_parts.validate_explicit_plan``; these tests
hold the direct publication seam to the same rule.

ONE SNAPSHOT, FOUR USES (#301).  The gate above was static: it reread the
mutable ``--plan-json`` path at each of the four moments that consume it, so a
planner that regenerated or atomically replaced its output while a long export
ran made the encoded artifact and the published plan describe different
allocations -- and the gate accepted that, because an entry that had become
``"PASSTHROUGH"`` carried no obligation to check.  The tests below replace the
file mid-export and hold the publication to the plan that drove the encode,
and hold the shared gate to explicit NEGATIVE obligations.

Deliberate behaviour that must NOT change: an *implicitly* planned tensor (the
``--grid``/``--q256`` default) still passes through on a shape failure, and an
explicit ``"PASSTHROUGH"``/``"BF16"`` entry is still a passthrough.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "export_tessera_serving", ROOT / "experiments" / "export_tessera_serving.py")
exporter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exporter)

BODY = "model.language_model.layers."


def _write(tmp_path: Path, tensors) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": 32, "moe_intermediate_size": 32},
    }))
    return src


def _run(tmp_path, monkeypatch, tensors, plan, *extra):
    _write(tmp_path, tensors)
    out = tmp_path / "out"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(
        "sys.argv",
        ["export_tessera_serving.py", str(tmp_path / "src"), str(out),
         "--grid", "E4M3", "--q256", "1024", "--device", "cpu", "--no-verify",
         "--plan-json", str(plan_path), *extra])
    exporter.main()
    return out


def _tensor(rows, cols, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(rows, cols, generator=g).bfloat16()


def _rewrite_plan_at_first_encode(monkeypatch, plan_path: Path, replacement):
    """Model an independent planner replacing its output mid-export.

    The hook fires inside the numerical encode -- after planning, the
    construction gate and (in a partitioned run) the sealed export identity,
    and before the plan gate and the manifest.  ``replacement is None`` deletes
    the file instead, which is what proves the exporter never reads it twice.
    """
    real = exporter.encode_linear_planes
    fired: list[int] = []

    def encode(*args, **kwargs):
        if not fired:
            fired.append(1)
            if replacement is None:
                plan_path.unlink()
            else:
                plan_path.write_text(json.dumps(replacement))
        return real(*args, **kwargs)

    monkeypatch.setattr(exporter, "encode_linear_planes", encode)
    return fired


def test_an_explicit_target_whose_shape_fails_is_refused_not_demoted(tmp_path, monkeypatch):
    """The #211 direct-writer repro: two down_proj at 31x32 and 32x32, both
    explicitly E4M3 q1024.  The 31-row tensor cannot be cut (rows % 32), and
    the export used to pass it through as BF16, name it in ignore, and publish
    a config.json that does not implement its plan."""
    bad = BODY + "0.mlp.down_proj.weight"
    good = BODY + "1.mlp.down_proj.weight"
    plan = {bad: {"grid": "E4M3", "q256": 1024}, good: {"grid": "E4M3", "q256": 1024}}
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, monkeypatch, {bad: _tensor(31, 32, 0), good: _tensor(32, 32, 1)}, plan)
    message = str(caught.value)
    assert bad in message and "rows" in message
    assert not (tmp_path / "out" / "config.json").exists()


def test_an_implicit_default_tensor_whose_shape_fails_still_passes_through(tmp_path, monkeypatch):
    """The deliberate default is untouched: no plan entry, no obligation."""
    bad = BODY + "0.mlp.down_proj.weight"
    good = BODY + "1.mlp.down_proj.weight"
    out = _run(tmp_path, monkeypatch,
               {bad: _tensor(31, 32, 0), good: _tensor(32, 32, 1)},
               {good: {"grid": "E4M3", "q256": 1024}})
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    assert BODY + "0.mlp.down_proj" in config["ignore"]
    targets = sorted(t for g in config["config_groups"].values() for t in g["targets"])
    assert targets == [BODY + "1.mlp.down_proj"]


def test_an_explicit_target_beyond_the_smoke_bound_is_refused(tmp_path, monkeypatch):
    """--layers N excluding a tensor the plan gives a rung is a contradiction,
    exactly as it already is for an expert stack; one of the two is wrong."""
    first = BODY + "0.mlp.down_proj.weight"
    second = BODY + "1.mlp.down_proj.weight"
    plan = {first: "PASSTHROUGH", second: {"grid": "E4M3", "q256": 1024}}
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, monkeypatch,
             {first: _tensor(32, 32, 0), second: _tensor(32, 32, 1)},
             plan, "--layers", "1")
    message = str(caught.value)
    assert second in message and "--layers 1" in message
    assert not (tmp_path / "out" / "config.json").exists()


def test_a_fused_group_with_an_explicit_member_and_passthrough_siblings_is_refused(tmp_path, monkeypatch):
    """q explicitly quantized, k/v explicitly PASSTHROUGH: one vLLM method per
    fused module, so the exporter cannot implement q's rung.  It used to drop
    q from the plan with a print."""
    q = BODY + "0.self_attn.q_proj.weight"
    k = BODY + "0.self_attn.k_proj.weight"
    v = BODY + "0.self_attn.v_proj.weight"
    plan = {q: {"grid": "E4M3", "q256": 1024}, k: "PASSTHROUGH", v: "PASSTHROUGH"}
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, monkeypatch,
             {name: _tensor(64, 32, i) for i, name in enumerate((q, k, v))}, plan)
    message = str(caught.value)
    assert BODY + "0.self_attn.qkv_proj" in message and q in message
    assert not (tmp_path / "out" / "config.json").exists()


def test_a_fused_group_whose_explicit_members_disagree_on_scheme_is_refused(tmp_path, monkeypatch):
    q = BODY + "0.self_attn.q_proj.weight"
    k = BODY + "0.self_attn.k_proj.weight"
    v = BODY + "0.self_attn.v_proj.weight"
    plan = {q: {"grid": "E4M3", "q256": 1024},
            k: {"grid": "E2M1x2", "q256": 896},
            v: {"grid": "E4M3", "q256": 1024}}
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, monkeypatch,
             {name: _tensor(64, 32, i) for i, name in enumerate((q, k, v))}, plan)
    message = str(caught.value)
    assert BODY + "0.self_attn.qkv_proj" in message
    assert not (tmp_path / "out" / "config.json").exists()


def test_passthrough_unrouted_refuses_to_drop_an_explicit_target(tmp_path, monkeypatch):
    """The flag resolves *implicitly* planned unrouted modules; an explicit rung
    on a module the runtime will not route is a contradiction to surface, not
    a target to silently shed.  (The whole-layer merge already refuses this
    artifact through validate_explicit_plan.)"""
    o = BODY + "0.self_attn.o_proj.weight"
    plan = {o: {"grid": "E4M3", "q256": 1024}}
    with pytest.raises(SystemExit) as caught:
        _run(tmp_path, monkeypatch, {o: _tensor(32, 32, 0)}, plan,
             "--passthrough-unrouted")
    message = str(caught.value)
    assert BODY + "0.self_attn.o_proj" in message and "plan" in message
    assert not (tmp_path / "out" / "config.json").exists()


def test_the_emitted_roles_are_validated_against_the_plan_before_publication(tmp_path, monkeypatch):
    """The direct export runs serving_parts.validate_explicit_plan -- the same
    gate the whole-layer merge runs -- before config.json exists.  Proved by
    injection: a validator that refuses must stop publication."""
    name = BODY + "0.mlp.down_proj.weight"

    def refuse(plan, modules, config_groups, *, source_tensors=None):
        raise ValueError("plan-verification-marker")

    monkeypatch.setattr(exporter, "validate_explicit_plan", refuse)
    with pytest.raises(SystemExit, match="plan-verification-marker"):
        _run(tmp_path, monkeypatch, {name: _tensor(32, 32, 0)},
             {name: "PASSTHROUGH"}, "--layers", "0")
    assert not (tmp_path / "out" / "config.json").exists()


def test_a_compliant_passthrough_export_still_publishes(tmp_path, monkeypatch):
    """The gate accepts the artifact that implements its plan."""
    name = BODY + "0.mlp.down_proj.weight"
    out = _run(tmp_path, monkeypatch, {name: _tensor(32, 32, 0)},
               {name: "PASSTHROUGH"}, "--layers", "0")
    assert (out / "config.json").exists()


def test_the_manifest_records_the_plan_content_not_only_its_path(tmp_path, monkeypatch):
    """The merged path seals options.plan into export_identity; the direct
    manifest stored only a pathname, so a later sidecar check without
    --plan-json could not recover the obligation (#211).  The content rides
    in the manifest beside the path."""
    name = BODY + "0.mlp.down_proj.weight"
    plan = {name: "PASSTHROUGH"}
    out = _run(tmp_path, monkeypatch, {name: _tensor(32, 32, 0)}, plan,
               "--layers", "0")
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["plan"] == plan
    assert manifest["plan_json"] is not None


# --- one snapshot from planning to publication (#301) -----------------------

NAME = BODY + "0.mlp.down_proj.weight"
MODULE = BODY + "0.mlp.down_proj"
QUANTIZED = {"grid": "E4M3", "q256": 1024}


def _emitted_role(out: Path) -> dict:
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    roles = [r for m in manifest["modules"].values() for r in m["roles"]]
    assert len(roles) == 1, roles
    return roles[0]


def test_a_plan_replaced_during_encode_does_not_change_the_published_plan(tmp_path, monkeypatch):
    """The #301 repro.  An E4M3 q1024 plan drives the encode; the file is
    replaced with ``PASSTHROUGH`` while the unit is being encoded.  The export
    used to publish a manifest recording PASSTHROUGH for a tensor it had just
    written as an E4M3 q1024 wire -- and the pre-publication gate accepted it,
    because it reread the same replaced file and a PASSTHROUGH entry carried
    no obligation."""
    plan = {NAME: dict(QUANTIZED)}
    fired = _rewrite_plan_at_first_encode(monkeypatch, tmp_path / "plan.json",
                                          {NAME: "PASSTHROUGH"})
    out = _run(tmp_path, monkeypatch, {NAME: _tensor(32, 32, 0)}, plan)
    assert fired, "the encode hook never ran; the repro proves nothing"
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["plan"] == plan
    role = _emitted_role(out)
    assert (role["grid"], role["q256"]) == ("E4M3", 1024)
    config = json.loads((out / "config.json").read_text())["quantization_config"]
    targets = sorted(t for g in config["config_groups"].values() for t in g["targets"])
    assert targets == [MODULE] and MODULE not in config["ignore"]


def test_a_plan_replaced_after_validation_does_not_change_the_published_plan(tmp_path, monkeypatch):
    """The second acceptance case: the replacement lands between the
    pre-publication gate and the manifest write, which used to be two separate
    reads of the same mutable path."""
    plan = {NAME: dict(QUANTIZED)}
    plan_path = tmp_path / "plan.json"
    real = exporter.validate_explicit_plan
    fired: list[int] = []

    def validate_then_replace(*args, **kwargs):
        real(*args, **kwargs)
        fired.append(1)
        plan_path.write_text(json.dumps({NAME: "PASSTHROUGH"}))

    monkeypatch.setattr(exporter, "validate_explicit_plan", validate_then_replace)
    out = _run(tmp_path, monkeypatch, {NAME: _tensor(32, 32, 0)}, plan)
    assert fired, "the gate never ran; the repro proves nothing"
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["plan"] == plan
    assert (_emitted_role(out)["grid"], _emitted_role(out)["q256"]) == ("E4M3", 1024)


def test_the_plan_file_is_read_once_and_never_reread(tmp_path, monkeypatch):
    """Deleting the file after planning is the strongest statement of the same
    rule: an export that still needs it has more than one snapshot."""
    plan = {NAME: dict(QUANTIZED)}
    fired = _rewrite_plan_at_first_encode(monkeypatch, tmp_path / "plan.json", None)
    out = _run(tmp_path, monkeypatch, {NAME: _tensor(32, 32, 0)}, plan)
    assert fired and not (tmp_path / "plan.json").exists()
    assert json.loads((out / "tessera_serving_manifest.json").read_text())["plan"] == plan


def test_a_partitioned_export_seals_the_snapshot_it_planned(tmp_path, monkeypatch):
    """The partition path seals ``options.plan`` into ``export_identity``, which
    every sibling part is compared against at merge.  It is the third read of
    the same path and must be the same snapshot."""
    plan = {NAME: dict(QUANTIZED)}
    fired = _rewrite_plan_at_first_encode(monkeypatch, tmp_path / "plan.json", None)
    out = _run(tmp_path, monkeypatch, {NAME: _tensor(32, 32, 0)}, plan,
               "--partition", "0/1", "--partition-runtime-image",
               "example/runtime@sha256:" + "0" * 64)
    assert fired
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["export_partition"]["identity"]["options"]["plan"] == plan
    assert manifest["plan"] == plan


def test_a_passthrough_plan_entry_refuses_a_quantized_emitted_role():
    """The complete-plan gate skipped ``PASSTHROUGH``/``BF16`` entries, so an
    explicit negative obligation was the one plan statement nothing checked.
    It is the same shared gate the whole-layer merge runs."""
    from tessera.serving_parts import validate_explicit_plan

    modules = {MODULE: {"family": "TESSERA_FP8", "structure": "dense", "roles": [
        {"tensor": NAME, "role": "down_proj", "rows": 32, "cols": 32,
         "grid": "E4M3", "q256": 1024}]}}
    validate_explicit_plan({NAME: dict(QUANTIZED)}, modules, {})   # the control
    for spelling in ("PASSTHROUGH", "BF16"):
        with pytest.raises(ValueError) as caught:
            validate_explicit_plan({NAME: spelling}, modules, {})
        assert NAME in str(caught.value) and spelling in str(caught.value)


def test_a_passthrough_stack_entry_refuses_an_emitted_routed_moe_module():
    from tessera.serving_parts import validate_explicit_plan

    stack = BODY + "0.mlp.experts"
    modules = {stack: {"family": "TESSERA_FP8", "structure": "routed_moe", "roles": []}}
    with pytest.raises(ValueError) as caught:
        validate_explicit_plan({stack: "PASSTHROUGH"}, modules, {})
    assert stack in str(caught.value) and "PASSTHROUGH" in str(caught.value)
