"""Uniform-control --model uses the real public checkpoint source classifier."""
import json

import torch
from safetensors.torch import save_file

from tessera.uniform_control import main


def test_model_plan_prices_dense_and_unpacked_experts_without_pinned_head(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'config.json').write_text(json.dumps({
        'model_type': 'lfm2_moe', 'hidden_size': 32, 'moe_intermediate_size': 32,
        'num_experts': 2,
    }))
    dense = {'model.layers.0.self_attn.q_proj.weight': (32, 32),
             'model.layers.0.self_attn.o_proj.weight': (32, 32)}
    routed = {f'model.layers.0.feed_forward.experts.{expert}.{role}.weight': (32, 32)
              for expert in range(2) for role in ('w1', 'w3', 'w2')}
    body = {**dense, **routed}
    tensors = {name: torch.zeros(shape, dtype=torch.bfloat16) for name, shape in body.items()}
    tensors['lm_head.weight'] = torch.zeros(64, 32, dtype=torch.bfloat16)
    save_file(tensors, str(source / 'model.safetensors'))
    plan = tmp_path / 'candidate.json'
    plan.write_text(json.dumps({name: {'grid': 'E4M3', 'q256': 1024} for name in body}))
    output = tmp_path / 'control.json'
    report = tmp_path / 'control-block.json'
    assert main(['plan', str(plan), '--model', str(source), '--out', str(output),
                 '--report', str(report)]) == 0
    projected = json.loads(output.read_text())
    assert set(projected) == set(body)
    assert 'lm_head.weight' not in projected
    assert all(entry == {'grid': 'E4M3', 'q256': 1024} for entry in projected.values())
    block = json.loads(report.read_text())
    assert block['schema'] == 'tessera.uniform_control.v1'
    assert block['control']['units'] == 8
    match = block['control']['match']
    assert match['varying_params'] == 8 * 32 * 32
    assert match['candidate_bits'] == match['control_bits']
    assert match['byte_matched'] is True
    assert block['verdict']['measured'] is False
