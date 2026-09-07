"""GLM E288/top8 loader/operator control with one explicitly repeated source expert.

This is not a model-quality experiment or a proof of expert permutation. The
producer encodes three independently bound trained matrices; the consumer loads
those templates through all 864 stock RoutedExperts projection loader calls.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import traceback

from experiments.glm_native_construction import digest, install, plain, write


def encode(args, request, control):
    import torch
    from safetensors.torch import save_file
    from tessera.alphabet import E4M3_GRID
    from tessera.export import DEFAULT_CODE, encode_linear_planes
    from tessera.fused import pack_fused
    from tessera.stock import materialize_stock

    assert os.environ.get('PRISMABUILD_CONTAINER_OWNER'), 'Producer requires PB admission'
    row = control['source_tensors'][args.role]
    payload = row['payload']
    assert digest(payload['path']) == payload['sha256']
    weight = torch.frombuffer(bytearray(Path(payload['path']).read_bytes()),
                             dtype=torch.bfloat16).reshape(row['spec']['shape']).cuda()
    exported, unit, forests = encode_linear_planes(
        weight, grid=E4M3_GRID, q256=args.q256, name=args.role, verify=True)
    blob = pack_fused([(args.role, weight.shape[0], exported.blob)])
    (args.out / 'projection.wire').write_bytes(blob)
    stock = {name: value.contiguous().cpu() for name, value in
             materialize_stock(unit, forests, DEFAULT_CODE).items()}
    save_file(stock, args.out / 'independent-stock.safetensors')
    return {'status': 'encoded_source_projection', 'source_tensor': row,
            'q256': args.q256, 'role': args.role, 'wire_bytes': len(blob),
            'wire_sha256': digest(args.out / 'projection.wire'),
            'stock_sha256': digest(args.out / 'independent-stock.safetensors')}


def error(got, expected):
    import torch
    g, e = got.float(), expected.float()
    delta = g - e
    return {'finite': bool(torch.isfinite(g).all() and torch.isfinite(e).all()),
            'max_abs': float(delta.abs().max()),
            'relative_l2': float(delta.norm() / e.norm().clamp_min(1e-20))}


def consume(args, request, control):
    import torch
    from safetensors.torch import load_file
    from tools import tessera_construction_census as census
    from tessera.serving.config import TesseraConfig
    from vllm.config import set_current_vllm_config
    from vllm.v1.worker.workspace import init_workspace_manager
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        make_fp8_moe_kernel, make_fp8_moe_quant_config)

    baseline = json.loads(args.construction.read_text())
    assert baseline['request_sha256'] == digest(args.request)
    target = baseline['owners'][0]['prefix']
    text = json.loads((Path(request['bounded_config']) / 'config.json').read_text())['text_config']
    e, h, n = int(text['n_routed_experts']), int(text['hidden_size']), int(text['moe_intermediate_size'])
    assert e == control['experts'] and text['num_experts_per_tok'] == control['top_k']
    templates, stock, source, input_receipts = {}, {}, {}, {}
    producer_manifest = None
    if args.producer_manifest:
        producer_manifest = json.loads(args.producer_manifest.read_text())
        assert producer_manifest['control_request_sha256'] == digest(args.control_request)
        assert set(producer_manifest['projections']) == set(control['source_tensors'])
    for role, row in control['source_tensors'].items():
        root = (Path(producer_manifest['projections'][role]['directory']) if producer_manifest
                else args.control_request.parent / f'encode-{args.q256}-{role}')
        if producer_manifest:
            assert digest(root / 'receipt.json') == producer_manifest['projections'][role]['receipt_sha256']
        receipt = json.loads((root / 'receipt.json').read_text())
        launch = json.loads((root / 'launcher-result.json').read_text())
        assert receipt['status'] == 'encoded_source_projection'
        assert receipt['control_request_sha256'] == digest(args.control_request)
        assert receipt['role'] == role and receipt['q256'] == args.q256
        assert launch['returncode'] == launch['container_exit_code'] == 0
        assert launch['container_removed'] and launch['controls_unchanged']
        assert digest(root / 'projection.wire') == receipt['wire_sha256']
        assert digest(root / 'independent-stock.safetensors') == receipt['stock_sha256']
        templates[role] = torch.frombuffer(bytearray((root / 'projection.wire').read_bytes()), dtype=torch.uint8)
        stock[role] = load_file(root / 'independent-stock.safetensors', device='cuda')
        assert digest(row['payload']['path']) == row['payload']['sha256']
        source[role] = torch.frombuffer(bytearray(Path(row['payload']['path']).read_bytes()),
                                      dtype=torch.bfloat16).reshape(row['spec']['shape']).cuda()
        input_receipts[role] = {'path': str(root / 'receipt.json'), 'sha256': digest(root / 'receipt.json')}
    from experiments.glm_native_construction import scheme_for
    scheme = scheme_for(text)
    for group in scheme['groups'].values():
        group['q256'] = args.q256
        group['wire_stride'] = max(templates[role].numel() for role, _ in group['roles'])
    quant_dict = {'quant_method': 'tessera', 'format': 'tessera',
        'config_groups': {'glm_experts': {'format': 'TESSERA',
            'targets': ['model.language_model.layers.3.mlp.experts'], 'scheme': scheme}},
        'ignore': sorted({p for p, _ in baseline['offered_modules'] if p != target})}
    quant = TesseraConfig.from_config(quant_dict)
    # First use the actual complete model factory for mapper/config setup. Then
    # instantiate its actual MoE class alone on CUDA, avoiding unrelated weights.
    model, mapped_quant, config = census.build_model(request['bounded_config'], 'meta', 512, quant_config=quant)
    assert target in mapped_quant.target_scheme
    moe_type = type(model.get_submodule(baseline['owners'][0]['name'].rsplit('.experts.', 1)[0]))
    assert moe_type.__name__ == 'Glm5NextMoE'
    # The constructor registers its custom-op owner on this config. Retire
    # only the exact meta owner before constructing its CUDA replacement.
    registered = config.compilation_config.static_forward_context.pop(target)
    assert registered is model.get_submodule(baseline['owners'][0]['name'].rsplit('.', 1)[0])
    del registered
    del model
    gc.collect()
    with set_current_vllm_config(config, check_compile=False):
        init_workspace_manager(torch.device('cuda'))
        with census._set_default_torch_dtype()(torch.bfloat16), torch.device('cuda'):
            moe = moe_type(config.model_config.hf_text_config, config.parallel_config,
                             mapped_quant, prefix=target.removesuffix('.experts'))
        layer = moe.experts.routed_experts
        method = layer.quant_method
        assert type(method).__name__ == 'TesseraMoEMethod'
        assert layer.swiglu_limit == 10 and layer.global_num_experts == e
        assert layer.moe_config.experts_per_token == 8
        loaded = sorted(layer.load_weights((f'{expert}.{role}.wire', wire)
            for expert in range(e) for role, wire in templates.items()))
        write(args.out, 'load-progress.json', {'loaded_names': loaded, 'projections_supplied': e * 3,
              'owner': target, 'backend': plain(method.fp8_backend), 'scheme': scheme})
        method.process_weights_after_loading(layer)
        # Independent producer materialization is compared with every decoded
        # consumer byte and channel scale, not merely one sampled expert.
        exact = True
        for expert in range(e):
            for role, got_w, got_s in (
                ('gate_proj', layer.w13_weight[expert, :n], layer.w13_weight_scale[expert, :n]),
                ('up_proj', layer.w13_weight[expert, n:], layer.w13_weight_scale[expert, n:]),
                ('down_proj', layer.w2_weight[expert], layer.w2_weight_scale[expert])):
                exact &= bool(torch.equal(got_w.view(torch.uint8), stock[role]['weight'].view(torch.uint8)))
                exact &= bool(torch.equal(got_s.reshape(-1), stock[role]['weight_scale'].reshape(-1)))
        assert exact, 'Consumer tile differs from independently materialized producer bytes/scales'
        write(args.out, 'tile-progress.json', {'all_864_projection_tiles_and_scales_exact': exact})
        # The independent stock leg gets producer tensors. It does not borrow
        # the consumer parameters it is supposed to check.
        w13 = torch.cat([stock[r]['weight'] for r in ('gate_proj', 'up_proj')]).unsqueeze(0).repeat(e, 1, 1)
        w2 = stock['down_proj']['weight'].unsqueeze(0).repeat(e, 1, 1)
        s13 = torch.cat([stock[r]['weight_scale'].reshape(-1) for r in ('gate_proj', 'up_proj')]).view(1, 2*n, 1).repeat(e, 1, 1)
        s2 = stock['down_proj']['weight_scale'].view(1, h, 1).repeat(e, 1, 1)
        qc = make_fp8_moe_quant_config(fp8_backend=method.fp8_backend,
            w1_scale=s13, w2_scale=s2, a1_scale=None, a2_scale=None,
            per_act_token_quant=True, per_out_ch_quant=True, block_shape=None,
            gemm1_alpha=layer.swiglu_alpha, gemm1_beta=layer.swiglu_beta,
            swiglu_limit=layer.swiglu_limit, layer=layer)
        kernel = make_fp8_moe_kernel(moe_quant_config=qc, moe_config=method.moe,
            fp8_backend=method.fp8_backend, experts_cls=method.experts_cls,
            routing_tables=layer._expert_routing_tables())
        results = []
        for name, tokens, multiplier in [('decode', 1, 1.), ('all_expert_slots', 36, 1.), ('clamp_stress', 4, 64.)]:
            gen = torch.Generator(device='cuda').manual_seed(573 + tokens)
            x = torch.randn(tokens, h, generator=gen, device='cuda', dtype=torch.bfloat16) * multiplier
            ids = (torch.arange(tokens * 8, device='cuda').reshape(tokens, 8) % e).to(torch.int32)
            weights = torch.rand(tokens, 8, generator=gen, device='cuda', dtype=torch.float32)
            weights *= float(text['routed_scaling_factor']) / weights.sum(-1, keepdim=True)
            got = method.apply(layer, x, weights, ids, None, None)
            native = kernel.apply(x, w13, w2, weights, ids, activation=layer.activation,
                global_num_experts=e, expert_map=layer.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=None, shared_experts_input=None)
            # A BF16-source arithmetic screen over the one repeated expert.
            gate = x.float() @ source['gate_proj'].float().T
            up = x.float() @ source['up_proj'].float().T
            clipped = int(((gate > 10) | (up.abs() > 10)).sum())
            act = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
            source_out = (act @ source['down_proj'].float().T) * weights.sum(-1, keepdim=True)
            torch.cuda.synchronize()
            comparison = error(got, native)
            assert comparison['finite'] and comparison['max_abs'] == 0, comparison
            if name == 'clamp_stress':
                assert clipped > 0, 'Stress did not exercise source clamp'
            results.append({'case': name, 'tokens': tokens, 'input_multiplier': multiplier,
                'expert_slots_covered': int(ids.unique().numel()), 'vs_independent_stock_fp8': comparison,
                'vs_bf16_source_arithmetic_screen': error(got, source_out),
                'source_gate_or_up_values_clipped': clipped})
            write(args.out, 'operator-progress.json', results)
        return {'status': 'repeated_source_control_passed', 'q256': args.q256,
            'input_receipts': input_receipts, 'backend': plain(method.fp8_backend),
            'moe_class': type(moe).__module__ + '.' + type(moe).__qualname__,
            'owner_class': type(layer).__module__ + '.' + type(layer).__qualname__,
            'owner_prefix': layer.layer_name, 'scheme': scheme,
            'all_864_projection_tiles_and_scales_exact': exact, 'operator_controls': results,
            'weight_shapes': [list(layer.w13_weight.shape), list(layer.w2_weight.shape)],
            'resident_parameter_bytes': sum(p.numel() * p.element_size() for p in layer.parameters())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--control-request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('encode', 'control'), required=True)
    parser.add_argument('--role', choices=('gate_proj', 'up_proj', 'down_proj'))
    parser.add_argument('--q256', type=int, required=True)
    parser.add_argument('--construction', type=Path)
    parser.add_argument('--producer-manifest', type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    control = json.loads(args.control_request.read_text())
    assert digest(args.request) == control['base_request']['sha256']
    assert args.q256 in control['q256']
    for row in (request['source_config'], request['source_archive'], request['core_manifest'],
                *request['config_files'].values()):
        assert digest(row['path']) == row['sha256']
    core, core_files = install(request, args.out)
    import torch
    from _pb_native_moe_measure.per_job_install import files
    from experiments.original_wire_generation import observed
    record = {'schema': 'tessera.glm_repeated_expert_control_result.v1',
        'request_sha256': digest(args.request), 'control_request_sha256': digest(args.control_request),
        'control_scope': control['control_scope'], 'runtime_cell_promoted': False,
        'stage': args.stage, 'tp_size': 1, 'ep_size': 1}
    if args.producer_manifest:
        record['producer_manifest'] = {'path': str(args.producer_manifest),
                                       'sha256': digest(args.producer_manifest)}
    rc = 0
    try:
        record.update((encode if args.stage == 'encode' else consume)(args, request, control))
    except Exception as exc:
        record.update(status='failed', reason=str(exc), traceback=traceback.format_exc())
        rc = 1
    finally:
        empty = type('Empty', (), {'named_modules': lambda self: []})()
        identity = observed(empty)['package_identity']
        assert identity['loaded_module_origins_verified']
        write(args.out, 'package-identity.json', identity)
        assert files(core) == core_files, 'Installed stock vLLM changed'
        record.update(stock_core_unchanged=True, package_identity_sha256=digest(args.out / 'package-identity.json'),
            cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        write(args.out, 'receipt.json', record)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
