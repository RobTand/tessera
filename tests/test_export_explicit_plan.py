"""An explicit plan entry is an obligation the direct export implements or refuses (#211).

The direct (non-partitioned) serving export used to demote an explicit
quantized target to BF16 passthrough in silence -- on a shape the grid cannot
cut, on the ``--layers`` smoke bound, on fused-group disagreement, and under
``--passthrough-unrouted`` -- and then published ``config.json`` without ever
checking the emitted roles against the plan.  The whole-layer merge path
already refuses through ``serving_parts.validate_explicit_plan``; these tests
hold the direct publication seam to the same rule.

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
