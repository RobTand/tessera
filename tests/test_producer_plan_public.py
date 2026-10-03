"""Public producer geometry requires the declared scientific dependencies."""
import json
import os
import sys

import torch


def test_public_producer_api_and_safe_path_cli_use_real_expert_geometry(tmp_path):
    """An installed producer needs no experiment/sibling repository imports."""
    import hashlib
    import subprocess
    import torch
    from safetensors.torch import save_file
    from tessera.producer_plan import producer_projection

    stack = "model.layers.2.feed_forward.experts"
    source = tmp_path / "checkpoint"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({
        "num_experts": 2, "hidden_size": 32, "moe_intermediate_size": 32}))
    save_file({f"{stack}.{expert}.{role}.weight": torch.zeros(32, 32, dtype=torch.bfloat16)
               for expert in range(2) for role in ("w1", "w2", "w3")},
              str(source / "model.safetensors"))
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({stack: {"grid": "E4M3", "q256": 1024,
                                        "source_layout": "unpacked_per_expert"}}))
    expected = producer_projection(source, plan)
    assert expected["schema"] == "tessera.expert_projection.v1"
    assert [u["projection"] for u in expected["stacks"][stack]["units"]] == [
        "gate_proj", "up_proj", "down_proj"] * 2
    assert expected["stacks"][stack]["groups"]["w13"]["rows"] == 64
    assert expected["source"]["files"]["model.safetensors"] == hashlib.sha256(
        (source / "model.safetensors").read_bytes()).hexdigest()
    output = tmp_path / "projection.json"
    env = {key: value for key, value in os.environ.items()
           if key not in {"TESSERA_REPO", "PRISMAQUANT_REPO", "PRISMABUILD_REPO"}}
    env["PYTHONSAFEPATH"] = "1"
    subprocess.run([sys.executable, "-m", "tessera.producer_plan", str(source),
                    "--stack-plan", str(plan), "--out", str(output)],
                   check=True, cwd=tmp_path, env=env, capture_output=True, text=True)
    # tensor_names returns a set; JSON object member order is not the v1 contract.
    assert json.loads(output.read_text()) == expected
