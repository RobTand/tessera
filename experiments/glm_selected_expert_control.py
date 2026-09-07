"""Opt-in eager selected-expert control; no production route or wire changes."""
from __future__ import annotations

from experiments.glm_native_construction import write
from experiments.glm_repeated_expert_control import error


def run(args, selected, templates, scheme, layer, method, w13, w2, s13, s2):
    if not isinstance(selected.get('cases'), list) or not selected['cases']:
        raise ValueError('selected-expert control requires a nonempty list of cases')
    import gc
    import torch
    from tessera.serving.fp8_route import PreparedTesseraFp8Module, prepare_tessera_fp8_module
    from tessera.serving.scheme import (expert_role_declarations, parse_tessera_expert_blob,
                                         validate_tessera_moe_scheme)
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        make_fp8_moe_kernel, make_fp8_moe_quant_config)

    e, h, n = w2.shape
    chunk = selected['max_experts_per_chunk']
    assert type(chunk) is int and chunk > 0
    declared = validate_tessera_moe_scheme(scheme, layer.layer_name)
    packed = {}
    for group, spec in declared['groups'].items():
        roles = []
        for declaration in expert_role_declarations(spec):
            role = declaration['roles'][0][0]
            roles.extend(parse_tessera_expert_blob(
                templates[role].numpy().tobytes(), declaration, layer.layer_name, device='cuda'))
        prepared = prepare_tessera_fp8_module(roles, device='cuda')
        packed[group] = PreparedTesseraFp8Module.stack([prepared] * e)
        del prepared, roles
    gc.collect()
    torch.cuda.synchronize()

    # A nine-bit row signature makes every global expert's effective down
    # matrix distinct, while preserving the repeated source wire control.
    # These are diagnostic weights, not 288 distinct trained experts or quality.
    bits = (torch.arange(h, device='cuda') % (e - 1).bit_length()).view(1, h, 1)
    signature = 1.0 + ((torch.arange(e, device='cuda').view(e, 1, 1) >> bits) & 1).float()
    diagnostic_s2 = s2 * signature
    assert torch.unique(signature.reshape(e, h), dim=0).shape[0] == e

    def kernel(scales1, scales2):
        qc = make_fp8_moe_quant_config(fp8_backend=method.fp8_backend,
            w1_scale=scales1, w2_scale=scales2, a1_scale=None, a2_scale=None,
            per_act_token_quant=True, per_out_ch_quant=True, block_shape=None,
            gemm1_alpha=layer.swiglu_alpha, gemm1_beta=layer.swiglu_beta,
            swiglu_limit=layer.swiglu_limit, layer=layer)
        return make_fp8_moe_kernel(moe_quant_config=qc, moe_config=method.moe,
            fp8_backend=method.fp8_backend, experts_cls=method.experts_cls,
            routing_tables=layer._expert_routing_tables())

    def apply(op, x, first, second, weights, ids, expert_map):
        return op.apply(x, first, second, weights, ids, activation=layer.activation,
            global_num_experts=e, expert_map=expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=None, shared_experts_input=None)

    resident = kernel(s13, diagnostic_s2)
    results = []
    for case in selected['cases']:
        tokens = int(case['tokens'])
        generator = torch.Generator(device='cuda').manual_seed(int(case['seed']))
        x = torch.randn(tokens, h, generator=generator, device='cuda', dtype=torch.bfloat16)
        x *= float(case['input_multiplier'])
        # A coprime affine permutation crosses non-adjacent global expert IDs.
        ids = ((torch.arange(tokens * 8, device='cuda') * 37 + 19) % e).reshape(tokens, 8).int()
        weights = torch.rand(tokens, 8, generator=generator, device='cuda')
        weights *= 2.5 / weights.sum(-1, keepdim=True)
        expected = apply(resident, x, w13, w2, weights, ids, None)
        selected_ids = torch.unique(ids).flip(0)
        expert_map = torch.full((e,), -1, dtype=torch.int32, device='cuda')
        expert_map.scatter_(0, selected_ids.long(), torch.arange(selected_ids.numel(), device='cuda', dtype=torch.int32))
        compact_s13 = packed['w13'].row_scale(selected_ids).unsqueeze(-1)
        compact_s2 = packed['w2'].row_scale(selected_ids).unsqueeze(-1) * signature.index_select(0, selected_ids)
        assert torch.equal(compact_s13, s13.index_select(0, selected_ids))
        assert torch.equal(compact_s2, diagnostic_s2.index_select(0, selected_ids))
        compact_kernel = kernel(compact_s13, compact_s2)
        # Establish stock mapping semantics independently of the wire decoder.
        compact_first = w13.index_select(0, selected_ids)
        compact_second = w2.index_select(0, selected_ids)
        mapped = apply(compact_kernel, x, compact_first, compact_second, weights, ids, expert_map)
        mapping_error = error(mapped, expected)
        assert mapping_error['finite'] and mapping_error['max_abs'] == 0, mapping_error
        del compact_first, compact_second

        allocated_before = torch.cuda.memory_allocated()
        with torch.profiler.record_function('selected_expert_window_decode'):
            first = packed['w13'].decode(selected_ids, max_experts_per_chunk=chunk).view(torch.float8_e4m3fn)
            second = packed['w2'].decode(selected_ids, max_experts_per_chunk=chunk).view(torch.float8_e4m3fn)
        # Every selected expert byte, in the actual nontrivial compact order.
        assert torch.equal(first.view(torch.uint8), w13.index_select(0, selected_ids).view(torch.uint8))
        assert torch.equal(second.view(torch.uint8), w2.index_select(0, selected_ids).view(torch.uint8))
        got = apply(compact_kernel, x, first, second, weights, ids, expert_map)
        torch.cuda.synchronize()
        decoded_error = error(got, expected)
        assert decoded_error['finite'] and decoded_error['max_abs'] == 0, decoded_error
        # Swapping two compact-to-global associations must change the answer.
        wrong_map = expert_map.clone()
        wrong_map[selected_ids[:2].long()] = wrong_map[selected_ids[:2].flip(0).long()]
        wrong = apply(compact_kernel, x, first, second, weights, ids, wrong_map)
        wrong_error = error(wrong, expected)
        assert wrong_error['finite'] and wrong_error['max_abs'] > 0, 'Expert permutation oracle insensitive'
        results.append({'case': case, 'selected_experts': selected_ids.numel(),
            'stock_compact_mapping': mapping_error, 'selected_decode_vs_stock': decoded_error,
            'deliberately_wrong_mapping': wrong_error,
            'all_selected_tiles_and_scales_exact': True,
            'allocated_before_decode': allocated_before,
            # Preserve the whole-control allocator high-water mark. These
            # functional controls do not claim an isolated per-case peak.
            'process_peak_allocated_after_case': torch.cuda.max_memory_allocated(),
            'fresh_decoded_weight_bytes': first.numel() + second.numel()})
        write(args.out, 'selected-expert-progress.json', results)
        del first, second, got, wrong, expected, mapped, compact_kernel
    return {'schema': 'tessera.glm_selected_expert_control.v1',
        'status': 'selected_expert_functional_control_passed', 'cases': results,
        'packed_owner_bytes': {name: p.resident_bytes() for name, p in packed.items()},
        'max_experts_per_chunk': chunk, 'dynamic_cardinality_sync': 'torch.unique eager',
        'diagnostic_expert_signature': 'down row scale times 1 + bit(global expert, row modulo 9)',
        'performance_claim': False, 'production_route_changed': False,
        'runtime_cell_promoted': False, 'all_trained_experts': False}
