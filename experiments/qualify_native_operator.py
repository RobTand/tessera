"""GPU instrumentation fixture, never a PrismaQuant row or release admission.

A fixed synthetic BF16 unit and basis-vector inputs exercise original-wire
preparation, numerical gates, timing and native scratch collection together.
Basis vectors make the independently decoded wire's BF16 reference exact.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--runtime-image', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from experiments.native_operator_resources import NativeMemoryCollector
    collector = NativeMemoryCollector(args.library)
    import torch
    from tessera.alphabet import BF16_GRID
    from tessera.cached_unit import encoding_input_identity, make_unit_record
    from tessera.export import encode_linear
    from tessera.unit_artifact import read_unit_artifact
    from experiments import bench_native_operator as bench
    unit, fmt = 'fixture.dense', 'TESSERA_BF16_K1_R512'
    torch.manual_seed(376)
    weight = (torch.randn(32, 32, device='cuda') * 0.02).bfloat16()
    encoded = encode_linear(weight.float(), grid=BF16_GRID, q256=512, name=fmt, verify=True)
    identity = encoding_input_identity(weight, unit, BF16_GRID, 512)
    record = make_unit_record(encoded.blob, identity, filename='fixture.tessera')
    rendered = read_unit_artifact(encoded.blob, device='cuda').bfloat16()
    prepared = bench.prepare_native_operator(encoded.blob, record, weight, rendered, unit=unit,
        format_name=fmt, runtime_image=args.runtime_image)
    operator = prepared['operator']
    fixture_sha = lambda label: hashlib.sha256(('instrumentation fixture ONLY: ' + label).encode()).hexdigest()
    joint = {'schema': 'prismaquant.joint_aura.operator.v1', 'qname': unit, 'format': fmt,
        'probe_identity_sha256': fixture_sha('probe'), 'source_weight': bench.tensor_identity(weight),
        'rendered_weight': bench.tensor_identity(rendered),
        'activation': {'clip_enabled': False, 'input_global_scale': None}}
    route = {'kind': 'dense', 'policy': 'TESSERA_BF16:resident', 'symbol': 'torch.mm',
        'decoder': 'torch_window', 'contract': 'bf16_unquantized'}
    phases, tensors = {}, {}
    for phase, m in [('prefill', 32), ('decode', 1)]:
        x = torch.eye(32, device='cuda', dtype=torch.bfloat16)[:m].contiguous()
        reference = torch.mm(x.float(), rendered.float().T).bfloat16()
        tensors[phase] = {'input': x, 'reference_qdq': x.clone(), 'reference_output': reference}
        phases[phase] = {'m': m, 'expected_route': route,
            **{key: bench.tensor_identity(value) for key, value in tensors[phase].items()}}
    panel = {'schema': bench.PANEL_SCHEMA, 'unit': unit, 'format': fmt, 'shape': [32, 32],
        'source_sha256': fixture_sha('synthetic source'), 'calibration_sha256': fixture_sha('basis vectors'),
        'cost_sha256': fixture_sha('NO joint cost'), 'probe_identity_sha256': fixture_sha('probe'),
        'joint_operator_identity_sha256': bench.identity_sha256(joint), 'joint_operator_identity': joint,
        'wire': {'blob_sha256': record['blob_sha256'], 'blob_bytes': len(encoded.blob), 'record': record},
        'execution': dict(bench.EXECUTION), 'runtime': prepared['runtime'],
        'native_tensors_sha256': bench.identity_sha256(operator['native_tensors']),
        'scheme_sha256': operator['scheme_sha256'], 'numerics': {'atol': 0.0, 'rtol': 0.0}, 'phases': phases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (args.out.parent / 'fixture.tessera').write_bytes(encoded.blob)
    (args.out.parent / 'fixture-panel.json').write_text(json.dumps(panel, indent=2) + '\n')
    try:
        receipt = bench.measure_prepared_operator(prepared, panel, tensors,
            warmup_iterations=8, iterations=8, resource_collector=collector)
    finally:
        trace = collector.finish(args.out.with_suffix('.memory.json'))
    bench.attach_resource_trace(receipt, trace)
    receipt['qualification_scope'] = 'synthetic BF16 instrumentation fixture; no actual PWC/joint-cost/release evidence'
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    summary = {'status': receipt['status'], 'resources': receipt['resources']['status'],
               'phases': {phase: {'numerics': receipt['phases'][phase]['numerics'],
                                 'bound': receipt['resources']['phases'][phase].get('bound')}
                          for phase in bench.PHASES}}
    print(json.dumps(summary, sort_keys=True))
    return 0 if receipt['status'] == 'timing_admissible' and receipt['resources']['status'] == 'complete_operator_bound' else 2


if __name__ == '__main__':
    raise SystemExit(main())
