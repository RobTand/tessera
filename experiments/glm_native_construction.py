"""Bounded GLM construction and actual Tessera method selection on stock vLLM.

This reads configuration only. Meta tensors are not loaded weights, a selected
backend is not an executed kernel, and an expected refusal is retained as one.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import traceback


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def plain(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [plain(x) for x in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return {f.name: plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if hasattr(value, 'shape') and hasattr(value, 'dtype'):
        return {'shape': list(value.shape), 'dtype': str(value.dtype),
                'device': str(value.device)}
    return str(value)


def write(root, name, value):
    path = root / name
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'artifact': str(path), 'sha256': digest(path),
                      'bytes': path.stat().st_size}), flush=True)


def install(request, root):
    # Reuse the original qualification's exact whole-core inventory algorithm.
    from _pb_native_moe_measure.per_job_install import files
    core = Path(importlib.util.find_spec('vllm').origin).parent
    expected = json.loads(Path(request['core_manifest']['path']).read_text())['files']
    assert files(core) == expected, 'Installed vLLM differs from the official image inventory'
    with tempfile.TemporaryDirectory(prefix='glm-tessera-install-') as temp:
        with tarfile.open(request['source_archive']['path']) as archive:
            archive.extractall(temp, filter='data')
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps',
                        '--no-build-isolation', '--no-cache-dir', temp], check=True)
    assert files(core) == expected, 'Plugin installation modified stock vLLM'
    # Match the existing qualification installer's root-to-runtime transition;
    # the shared evidence mount deliberately does not grant root write access.
    owner = root.stat()
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(owner.st_gid)
        os.setuid(owner.st_uid)
    assert os.getuid() == owner.st_uid and os.getgid() == owner.st_gid
    write(root, 'installation.json', {'source_commit': request['source_commit'],
        'source_archive': request['source_archive'], 'core_manifest': request['core_manifest'],
        'stock_core_files': len(expected), 'stock_core_unchanged': True})
    return core, expected


def scheme_for(text):
    hidden, intermediate = int(text['hidden_size']), int(text['moe_intermediate_size'])
    # Stride 1 describes a constructor-only parameter. No bytes are supplied or
    # decoded, and this scheme must never be exported as a loaded artifact.
    return {'family': 'TESSERA_FP8', 'structure': 'routed_moe', 'grid': 'E4M3',
        'body': 'WINDOW', 'plane': 'CHANNEL', 'experts': int(text['n_routed_experts']),
        'groups': {
            'w13': {'rows': 2 * intermediate, 'columns': hidden, 'q256': 1024,
                    'wire_stride': 1, 'roles': [['gate_proj', intermediate], ['up_proj', intermediate]]},
            'w2': {'rows': hidden, 'columns': intermediate, 'q256': 1024,
                   'wire_stride': 1, 'roles': [['down_proj', hidden]]}}}


def owners(model):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    result = []
    attrs = ('global_num_experts', 'local_num_experts', 'top_k', 'activation',
             'swiglu_limit', 'swiglu_alpha', 'swiglu_beta', 'scoring_func',
             'renormalize', 'use_grouped_topk', 'num_expert_group', 'topk_group',
             'routed_scaling_factor', 'apply_router_weight_on_input',
             'e_score_correction_bias', 'tessera_family', 'tessera_mode')
    for name, module in model.named_modules():
        if not isinstance(module, RoutedExperts):
            continue
        method = module.quant_method
        result.append({'name': name, 'prefix': module.layer_name,
            'class': type(module).__module__ + '.' + type(module).__qualname__,
            'method': type(method).__module__ + '.' + type(method).__qualname__,
            'backend': plain(getattr(method, 'fp8_backend', None)),
            'attributes': {key: plain(getattr(module, key, None)) for key in attrs},
            'moe_config': plain(module.moe_config),
            'parameters': {key: plain(value) for key, value in module.named_parameters(recurse=False)}})
    return result


def verify_owners(rows, text):
    expected_count = sum(x == 'sparse' for x in text['mlp_layer_types'])
    assert len(rows) == expected_count, (len(rows), expected_count)
    for row in rows:
        config, attributes = row['moe_config'], row['attributes']
        for key, expected in {'num_experts': text['n_routed_experts'],
            'num_local_experts': text['n_routed_experts'],
            'experts_per_token': text['num_experts_per_tok'],
            'hidden_dim': text['hidden_size'],
            'intermediate_size': text['moe_intermediate_size']}.items():
            assert config[key] == expected, (key, config[key], expected)
        for key, expected in {'scoring_func': text['scoring_func'],
            'renormalize': text['norm_topk_prob'], 'swiglu_limit': text['swiglu_limit'],
            'routed_scaling_factor': text['routed_scaling_factor']}.items():
            assert attributes[key] == expected, (key, attributes[key], expected)
        e, h, n = text['n_routed_experts'], text['hidden_size'], text['moe_intermediate_size']
        assert row['parameters']['w13_weight']['shape'] == [e, 2*n, h]
        assert row['parameters']['w2_weight']['shape'] == [e, h, n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('construction', 'resident', 'streamed'), required=True)
    parser.add_argument('--construction', type=Path)
    args = parser.parse_args()
    root = args.out
    request = json.loads(args.request.read_text())
    assert request['schema'] == 'tessera.glm_native_qualification_request.v1'
    for row in (request['source_config'], request['source_archive'], request['core_manifest'],
                *request['config_files'].values()):
        assert digest(row['path']) == row['sha256'], row['path']
    core, core_files = install(request, root)
    from _pb_native_moe_measure.per_job_install import files
    from experiments.original_wire_generation import observed
    from tools import tessera_construction_census as census
    import torch
    import vllm
    text = json.loads((Path(request['bounded_config']) / 'config.json').read_text())['text_config']
    quant = None
    if args.stage != 'construction':
        assert args.construction is not None
        baseline = json.loads(args.construction.read_text())
        assert baseline['status'] == 'constructed_on_meta'
        assert baseline['request_sha256'] == digest(args.request)
        assert len(baseline['owners']) == 1
        target = baseline['owners'][0]['prefix']
        # The source spelling must be mapped by stock initialize_model; ignores
        # come from the actual preceding construction, not a handwritten menu.
        source_target = 'model.language_model.layers.3.mlp.experts'
        quant_dict = {'quant_method': 'tessera', 'format': 'tessera',
            'config_groups': {'glm_experts': {'format': 'TESSERA', 'targets': [source_target],
                                             'scheme': scheme_for(text)}},
            'ignore': sorted({p for p, _ in baseline['offered_modules'] if p != target})}
        os.environ['TESSERA_SERVE_MODE'] = args.stage
        from tessera.serving.config import TesseraConfig
        quant = TesseraConfig.from_config(quant_dict)
        write(root, 'constructor-only-quant-config.json', quant_dict)
    model = None
    record = {'schema': 'tessera.glm_native_construction.v1', 'stage': args.stage,
        'request_sha256': digest(args.request), 'control_scope': request['control_scope'],
        'runtime': {'vllm': vllm.__version__, 'torch': torch.__version__,
                    'image': request['runtime_image']},
        'tensor_device': 'meta', 'weight_bytes_loaded': 0, 'forward_executed': False,
        'runtime_cell_promoted': False, 'tp_size': 1, 'ep_size': 1}
    exit_code = 0
    try:
        model, used_quant, config = census.build_model(request['bounded_config'], 'meta', 512,
                                                      quant_config=quant)
        rows = owners(model)
        verify_owners(rows, text)
        record.update(status='constructed_on_meta', owners=rows,
            model_class=type(model).__module__ + '.' + type(model).__qualname__,
            model=census.model_stamp(config, request['bounded_config']),
            packed_modules_mapping=getattr(type(model), 'packed_modules_mapping', None),
            hf_to_vllm_mapper_unstacked=census._weights_mapper_table(type(model)),
            supports_quant=census._supports_quant(type(model)))
        if args.stage == 'construction':
            record.update(census.census(model, used_quant))
            record['offered_modules'] = used_quant.asked
        else:
            record['mapped_targets'] = list(used_quant.target_scheme)
            assert rows[0]['prefix'] in used_quant.target_scheme
            assert 'TesseraMoEMethod' in rows[0]['method']
        if args.stage == 'streamed':
            record['status'] = 'unexpected_streamed_construction'
            exit_code = 1
    except Exception as exc:
        record.update(status='refused', exception_type=type(exc).__name__, reason=str(exc),
                      traceback=traceback.format_exc())
        expected = (args.stage == 'streamed' and isinstance(exc, ValueError)
                    and "expert route serves 'resident' only" in str(exc))
        record['expected_streamed_refusal'] = expected
        exit_code = 0 if expected else 1
    finally:
        # This observer checks the actual loaded module files as well as the
        # recomputed installed source roster, even on an unsuccessful route.
        empty = type('Empty', (), {'named_modules': lambda self: []})()
        identity = observed(empty)['package_identity']
        assert identity['loaded_module_origins_verified']
        assert identity['fresh_encoder_source_sha256'] == identity['independently_recomputed_source_sha256']
        write(root, 'package-identity.json', identity)
        assert files(core) == core_files, 'Qualification modified stock vLLM'
        record['stock_core_unchanged'] = True
        record['package_identity_sha256'] = digest(root / 'package-identity.json')
        record['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        record['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        write(root, 'receipt.json', record)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
