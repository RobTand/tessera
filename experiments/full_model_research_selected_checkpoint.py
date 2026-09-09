"""Encode one full-model E4M3/R1024 checkpoint that declares research selected MoE.

The reference at full-model-original-r1024 reuses the priced campaign wires and
carries no execution declaration, and its cached bundle seals the producer tree
that predates the packed selected bridge, so ``verify_cached_unit`` refuses it
here.  This driver therefore encodes every planned unit on the named producer
freeze instead of reusing a receipt, and seals one execution declaration into
the same export identity.  It qualifies nothing by itself: the checkpoint it
writes is the input to ordinary stock discovery, load and generation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from safetensors import safe_open
import export_tessera_serving as exporter
from tessera.cached_unit import encoder_source_sha256
from tessera.fused import parse_fused
from tessera.moe_execution import ResearchSelectedMoeInput
from tessera.serving_parts import source_identity, sha256_file

SRC = Path('/mnt/shared/models/LFM2.5-8B-A1B-BF16')
MERGED = Path('/mnt/shared/tessera-measurements/first-model-20260907/full-model-anchors-merged-01')
CENSUS = Path('/mnt/shared/tessera-measurements/first-model-20260907/full-model-anchors-02/census.json')
CENSUS_SHA = '62d41825f84edd280de46bbb89893676fae8203563834dc95d8ad29c05bad04d'
HESSIAN_PROVENANCE_SHA = 'e2781e6c12097b50d2edb4d57fa48112eb9782e7ce776f161546d25b3ac654c2'
# The producer freeze this encode is priced on.  Any edit under src/tessera moves
# it, and the export must then be re-submitted against the new value rather than
# relabelled: the reference bundle at 57809bff86 is refused here for that reason.
ENCODER_SOURCE_SHA256 = '959a1a43b26865e5e04dfacbabd627634ade4537979568af9c00c76d9604f9ea'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--execution-json', type=Path, required=True,
                    help='sealed tessera.research_selected_moe.v1 declaration')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args(argv)
    started = time.time()
    torch.set_num_threads(4)
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=False)

    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
        print(json.dumps({'artifact': str(path), 'sha256': sha256_file(path),
                          'bytes': path.stat().st_size}), flush=True)

    observed = encoder_source_sha256()
    if observed != ENCODER_SOURCE_SHA256:
        raise SystemExit(f'producer freeze moved: {observed} is not {ENCODER_SOURCE_SHA256}')
    assert sha256_file(CENSUS) == CENSUS_SHA
    census = json.loads(CENSUS.read_text())
    expected = set(census['dense_targets']) | set(census['expert_targets'])
    assert len(expected) == 2142 and len(census['dense_targets']) == 30
    stacks = census['expert_projection']['stacks']
    assert len(stacks) == 22
    assert {name for units in stacks.values() for name in units} == set(census['expert_targets'])

    execution = ResearchSelectedMoeInput.read(args.execution_json)
    write('execution-input.json', execution.record())

    _, dense, _, _ = exporter.quantizable(SRC)
    plan = {name: 'PASSTHROUGH' for name in dense if not exporter.MOE_ROUTER.match(name)}
    for name in census['dense_targets']:
        plan[name + '.weight'] = {'grid': 'E4M3', 'q256': 1024}
    for stack in sorted(stacks):
        plan[stack] = {'grid': 'E4M3', 'q256': 1024, 'source_layout': 'unpacked_per_expert'}
    write('plan.json', plan)
    source = source_identity(SRC)
    write('source-identity.json', source)

    hessian = MERGED / 'cache/hessian_capture.pt'
    hp = hessian.with_name(hessian.name + '.provenance.json')
    assert json.loads(hp.read_text())['capture_sha256'] == HESSIAN_PROVENANCE_SHA
    write('hessian-input.json', {'path': str(hessian), 'sha256': sha256_file(hessian),
                                 'provenance_sha256': sha256_file(hp)})

    out = root / 'checkpoint'
    cmd = [sys.executable, str(Path(exporter.__file__)), str(SRC), str(out),
           '--grid', 'E4M3', '--q256', '1024', '--device', args.device,
           '--plan-json', str(root / 'plan.json'), '--hessian', str(hessian),
           '--research-selected-moe-json', str(args.execution_json.resolve())]
    write('export-command.json', {'argv': cmd, 'encoder_source_sha256': observed})
    exporter_started = time.time()
    import subprocess
    subprocess.run(cmd, check=True)
    exporter_finished = time.time()

    config = json.loads((out / 'config.json').read_text())
    qconfig = config['quantization_config']
    assert qconfig['quant_method'] == 'tessera'
    # The declaration is a config field, not a tensor: the encode above is what
    # makes these bytes, and this asserts the seal travelled with them.
    assert qconfig['research_selected_moe'] == execution.config.as_checkpoint()
    manifest = json.loads((out / 'tessera_serving_manifest.json').read_text())
    assert ResearchSelectedMoeInput.from_record(
        manifest['research_selected_moe']).config == execution.config

    routed = {name: group for name, group in qconfig['config_groups'].items()
              if group['scheme']['structure'] == 'routed_moe'}
    assert len(routed) == 22
    declared_stacks = {target for group in routed.values() for target in group['targets']}
    assert declared_stacks == set(stacks)
    for group in routed.values():
        scheme = group['scheme']
        assert scheme['family'] == 'TESSERA_FP8' and scheme['grid'] == 'E4M3'
        assert scheme['experts'] == 32

    replacements = {name + '.weight' for name in expected}
    wires, passthrough, output_wires = {}, {}, set()
    with safe_open(str(SRC / 'model.safetensors'), framework='pt') as before, \
            safe_open(str(out / 'model.safetensors'), framework='pt') as after:
        for module, info in manifest['modules'].items():
            if module in stacks:
                for role in info['roles']:
                    name = role['tensor'].removesuffix('.weight')
                    key = name + '.wire'
                    members = parse_fused(after.get_tensor(key).numpy().tobytes())
                    assert len(members) == 1
                    wires[name] = {'tensor': key, 'bytes': len(members[0].blob),
                                   'sha256': hashlib.sha256(members[0].blob).hexdigest()}
                    output_wires.add(key)
            else:
                key = module + '.wire_bytes'
                members = parse_fused(after.get_tensor(key).numpy().tobytes())
                assert len(members) == len(info['roles'])
                for member, role in zip(members, info['roles']):
                    name = role['tensor'].removesuffix('.weight')
                    assert member.name == role['role'] and member.rows == role['rows']
                    wires[name] = {'tensor': key, 'role': member.name, 'bytes': len(member.blob),
                                   'sha256': hashlib.sha256(member.blob).hexdigest()}
                output_wires.add(key)
        # Every planned unit, every stack and every projection, from the emitted
        # tensors rather than the exporter summary.
        assert set(wires) == expected
        for stack, units in stacks.items():
            for name, unit in units.items():
                assert name in wires, name
                assert unit['tensor'].removesuffix('.weight') == name
        assert {unit['projection'] for units in stacks.values() for unit in units.values()} == {
            'gate_proj', 'up_proj', 'down_proj'}
        keep = set(before.keys()) - replacements
        assert set(after.keys()) == keep | output_wires
        for name in sorted(keep):
            a, b = before.get_tensor(name), after.get_tensor(name)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), name
            passthrough[name] = {'shape': list(a.shape), 'dtype': str(a.dtype)}

    write('export-proof.json', {
        'schema': 'tessera.full_model_research_selected_checkpoint.v1', 'status': 'passed',
        'scope': 'Uniform E4M3/R1024 research checkpoint encoded on one named producer freeze, '
                 'declaring research selected MoE execution. All 2142 planned units freshly '
                 'encoded; every other tensor bit-exact source precision. Not a reuse of the '
                 'priced campaign blobs, not allocator-derived, and not a production default. '
                 'It establishes no serving result on its own.',
        'encoder_source_sha256': observed,
        'execution': execution.record(),
        'census_sha256': CENSUS_SHA,
        'plan_sha256': sha256_file(root / 'plan.json'),
        'source_identity_sha256': sha256_file(root / 'source-identity.json'),
        'stacks': sorted(stacks), 'wires': wires, 'passthrough': passthrough,
        'checkpoint_files': {p.name: {'sha256': sha256_file(p), 'bytes': p.stat().st_size}
                             for p in sorted(out.iterdir()) if p.is_file()},
        'started_epoch': started, 'export_started_epoch': exporter_started,
        'export_finished_epoch': exporter_finished, 'finished_epoch': time.time()})


if __name__ == '__main__':
    main()
