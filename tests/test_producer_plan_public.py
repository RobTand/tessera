"""Public producer geometry requires the declared scientific dependencies."""
import json
import os
import sys
from pathlib import Path

import torch
import pytest


@pytest.mark.parametrize("packed", [False, True])
def test_substack_projection_keeps_uniform_plan_and_prices_units(packed, tmp_path):
    from tessera.export_serving import project_expert_plan, expert_work_units

    stack = "model.layers.2.mlp.experts"
    config = {"num_experts": 2, "hidden_size": 128, "moe_intermediate_size": 128}
    if packed:
        shapes = {f"{stack}.gate_up_proj": (2, 256, 128),
                  f"{stack}.down_proj": (2, 128, 128)}
        layout = "out_first_chunked"
    else:
        shapes = {f"{stack}.{e}.{p}.weight": (128, 128) for e in range(2)
                  for p in ("gate_proj", "up_proj", "down_proj")}
        layout = "unpacked_per_expert"
    choice = {"grid": "E4M3", "q256": 1024, "source_layout": layout}
    original = project_expert_plan(shapes, config, {stack: choice})
    units = original["stacks"][stack]["units"]
    names = [u["tensor"].removesuffix(".weight") for u in units]
    uniform = project_expert_plan(shapes, config, {stack: dict(
        choice, unit_q256=dict.fromkeys(names, 1024))})
    assert uniform == original
    picked = names[:2]
    assignments = dict.fromkeys(picked, 1088)
    mixed = project_expert_plan(shapes, config, {stack: dict(
        choice, unit_q256=assignments)})["stacks"][stack]
    work = expert_work_units(stack, mixed)
    assert [u.get("q256", mixed["q256"]) for u in work] == [1088, 1088, 1024] + [1024] * 3
    assert mixed["expert_ids"] == [1, 0]
    assert mixed["expert_classes"] == [
        {"start": 0, "end": 1, "q256": {"w13": [1024, 1024], "w2": [1024]}},
        {"start": 1, "end": 2, "q256": {"w13": [1088, 1088], "w2": [1024]}}]
    source_fields = {"q256", "stack", "storage_expert", "wire"}
    assert [{k: v for k, v in u.items() if k not in source_fields} for u in work] == [
        {k: v for k, v in u.items() if k not in source_fields} for u in units]
    for unit in work:
        assert mixed["expert_ids"][unit["storage_expert"]] == unit["expert"]
        assert unit["source_slice"]["expert"] == unit["expert"]
        assert unit["wire"] == f"{stack}.{unit['storage_expert']}.{unit['projection']}.wire"
    with pytest.raises(SystemExit, match="unknown.*unit|unit.*unknown"):
        project_expert_plan(shapes, config, {stack: dict(
            choice, unit_q256={stack + ".999.up_proj": 1088})})
    from tessera.serving.contract import load_serving_contract, validate_serving_contract
    from tessera.serving_plan import ROUTED_UNIT_ASSIGNMENT
    import copy
    contract = load_serving_contract()
    assert contract["contract_version"] >= 57
    assert contract["producer_interface"]["routed_units"] == ROUTED_UNIT_ASSIGNMENT
    wrong = copy.deepcopy(contract)
    wrong["producer_interface"]["routed_units"]["plannable_unit"] = "stack"
    with pytest.raises(ValueError, match="routed_units"):
        validate_serving_contract(wrong)
    missing = copy.deepcopy(contract)
    del missing["producer_interface"]["routed_units"]
    with pytest.raises(ValueError, match="routed_units"):
        validate_serving_contract(missing)
    import subprocess
    from safetensors.torch import save_file, load_file
    from tessera.fused import parse_fused
    from tessera.unit_artifact import read_unit_artifact
    source = tmp_path / "source"
    source.mkdir()
    tensors = {name: torch.randn(shape, generator=torch.Generator().manual_seed(i)).bfloat16()
               for i, (name, shape) in enumerate(shapes.items())}
    save_file(tensors, str(source / "model.safetensors"), metadata={"format": "pt"})
    (source / "config.json").write_text(json.dumps(dict(
        config, architectures=["Glm5NextForConditionalGeneration"])))
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / "src"))
    artifacts = {}
    for label, spec in [("r1024", choice), ("r1088", dict(choice, q256=1088)),
                        ("mixed", dict(choice, unit_q256=assignments))]:
        plan = tmp_path / (label + ".json")
        plan.write_text(json.dumps({stack: spec}))
        out = tmp_path / label
        command = [sys.executable, "-m", "tessera.export_serving", str(source), str(out),
                   "--grid", "E4M3", "--q256", "1024", "--device", "cpu",
                   "--no-verify", "--plan-json", str(plan), "--allow-unrouted"]
        proc = subprocess.run(command, cwd=root, env=env, capture_output=True, text=True, timeout=600)
        assert proc.returncode == 0, proc.stderr[-4000:]
        artifacts[label] = {key: value for shard in out.glob("*.safetensors")
                            for key, value in load_file(str(shard)).items()}
    for unit in units:
        label = "r1088" if unit["tensor"].removesuffix(".weight") in picked else "r1024"
        emitted = next(u for u in work if u["tensor"] == unit["tensor"])
        mixed_blob = artifacts["mixed"][emitted["wire"]].numpy().tobytes()
        uniform_blob = artifacts[label][unit["wire"]].numpy().tobytes()
        assert mixed_blob == uniform_blob
        mixed_unit = parse_fused(mixed_blob)[0]
        uniform_unit = parse_fused(uniform_blob)[0]
        assert torch.equal(read_unit_artifact(mixed_unit.blob),
                           read_unit_artifact(uniform_unit.blob))


def test_mixed_rank_layout_refuses_unpredictable_cut_before_write():
    from tessera import export_serving as owner
    from tessera.grammar import bresenham_rate_schedule, root_from_q256
    rates = bresenham_rate_schedule(root_from_q256(1088), 32, cap=8)
    layout = {"group": "w2", "projection": "down_proj", "rows": 64, "cols": 32,
              "window_bits": 14, "rates": rates}
    owner.require_plannable_unit_layout(layout, 1088, "unit-X")
    # Same full quota, but a non-aligned importance placement changes both
    # rank-local sizes. A q256-only declaration cannot price/preallocate it.
    wrong = dict(layout, rates=(5,) * 8 + (4,) * 24)
    with pytest.raises(SystemExit, match="unit-X.*TP2.*rank"):
        owner.require_plannable_unit_layout(wrong, 1088, "unit-X")

def test_public_producer_api_and_safe_path_cli_use_real_expert_geometry(tmp_path):
    """An installed producer needs no experiment/sibling repository imports."""
    import hashlib
    import subprocess
    import torch
    from safetensors.torch import save_file
    from tessera import producer_plan

    stack = "model.layers.2.feed_forward.experts"
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "num_experts": 2, "hidden_size": 128, "moe_intermediate_size": 128}))
    save_file({f"{stack}.{expert}.{role}.weight": torch.zeros(128, 128, dtype=torch.bfloat16)
               for expert in range(2) for role in ("w1", "w2", "w3")},
              str(source / "model.safetensors"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({stack: {"grid": "E4M3", "q256": 1024,
                                        "source_layout": "unpacked_per_expert"}}))
    expected = producer_plan.producer_projection(source, plan)
    assert expected["schema"] == "tessera.expert_projection.v1"
    assert [u["projection"] for u in expected["stacks"][stack]["units"]] == [
        "gate_proj", "up_proj", "down_proj"] * 2
    assert expected["stacks"][stack]["groups"]["w13"]["rows"] == 256
    assert expected["source"]["files"]["model.safetensors"] == hashlib.sha256(
        (source / "model.safetensors").read_bytes()).hexdigest()
    output = tmp_path / "projection.json"
    env = {key: value for key, value in os.environ.items()
           if key not in {"TESSERA_REPO", "PRISMAQUANT_REPO", "PRISMABUILD_REPO"}}
    env["PYTHONSAFEPATH"] = "1"
    # Keep API and CLI on the same package after cwd changes. In an installed
    # qualifier this is site-packages; in a source run it is the PB snapshot.
    # Do not inherit relative paths or experiment/sibling repository roots.
    env["PYTHONPATH"] = str(Path(producer_plan.__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-m", "tessera.producer_plan", str(source),
         "--stack-plan", str(plan), "--out", str(output)],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr or result.stdout
    # tensor_names returns a set; JSON object member order is not the v1 contract.
    assert json.loads(output.read_text()) == expected
