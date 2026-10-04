"""The published serve-comparison intake binds the actual audited export.

Ported from the independently reviewed deployment intake tests (issue #885):
approved-artifact binding, stale/unbound export refusals before any output,
serialized-identity consistency, gate consumption of the frozen artifact
instead of a hardcoded export, runtime same-path substitution refusals, shell
TS_PIN identifier safety, and the retained historical pathname-only mode.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INTAKE = _load('comparison_input_intake')
OWNER = _load('comparison_arm_identity')


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / 'source'; source.mkdir()
    p = source / 'src/tessera/serving/runtime_contract.json'
    p.parent.mkdir(parents=True); p.write_text('{"contract_version":55}')
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', '-C', str(source), 'add', '.'], check=True)
    subprocess.run(['git', '-C', str(source), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test',
                    'commit', '-qm', 'Fixture common source'], check=True)
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    artifact = tmp_path / 'actual-export'; artifact.mkdir()
    metadata = {'config.json': {'num_hidden_layers': 1},
                'model.safetensors.index.json': {'weight_map': {'weight': 'model.safetensors'}},
                'tessera_serving_manifest.json': {'git': 'c' * 40, 'source': '/qualified/bf16',
                    'serving_gate': {'contract_version': 54},
                    'totals': {'checkpoint_bytes': 3, 'quantized_params': 6, 'passthrough_bytes': 2}}}
    for name, contents in metadata.items():
        (artifact / name).write_text(json.dumps(contents))
    (artifact / 'model.safetensors').write_bytes(b'abc')
    files = [{'name': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
             for p in artifact.iterdir()]
    audit = tmp_path / 'audit.json'
    audit.write_text(json.dumps({'schema': 'pact.u2.a8s-audit.v1', 'artifact': str(artifact), 'files': files,
                                'all_files_bytes': sum(row['bytes'] for row in files),
                                'manifest_sha256': hashlib.sha256((artifact / 'tessera_serving_manifest.json').read_bytes()).hexdigest()}))
    for arm in ('BASE', 'CANDIDATE'):
        (tmp_path / (arm + '.env')).write_text('\n'.join([
            'ARM_ARTIFACT=' + shlex.quote(str(artifact)), 'ARM_AUDIT_ROSTER=' + shlex.quote(str(audit)),
            'ARM_AUDIT_ROSTER_SHA256=' + hashlib.sha256(audit.read_bytes()).hexdigest()]) + '\n')
    manifest = tmp_path / 'comparison-manifest.json'
    manifest.write_text(json.dumps({'identities': {'comparison': 'sealed elsewhere'}}))
    args = argparse.Namespace(source=str(source), commit=commit, manifest=str(manifest),
                              output=str(tmp_path / 'intake'),
                              reference_arm=str(tmp_path / 'BASE.env'), candidate_arm=str(tmp_path / 'CANDIDATE.env'),
                              artifact_exception_reason='Explicit reviewed test export/common-source difference')
    return source, args


def test_intake_binds_only_approved_artifact_and_common_source_identity(fixture):
    source, args = fixture
    result = INTAKE.intake(args)
    bound = json.loads((Path(args.output) / 'manifest.json').read_text())
    assert bound['identities']['tessera_artifact'] == str(source.parent / 'actual-export')
    assert bound['identities']['tessera_artifact_binding'] == result['artifact_identity']
    assert bound['identities']['tessera_candidate_commit'] == args.commit
    assert bound['identities']['tessera_candidate_contract_sha256'] == result['contract_sha256']
    assert bound['identities']['comparison'] == 'sealed elsewhere'
    shell = (Path(args.output) / 'pin-env.sh').read_text()
    assert "export LEAD_BASE_ARM=BASE" in shell and "export LEAD_GATE_ARM=CANDIDATE" in shell
    assert "export TS_PIN_BASE=" + args.commit[:8] in shell
    assert "export TS_PIN_CANDIDATE=" + args.commit[:8] in shell
    assert "_ARTIFACT_EXPORT_COMMIT=" + 'c' * 40 in shell
    assert "_ARTIFACT_CONTRACT_VERSION=54" in shell
    # shlex.quote wraps the reviewed reason in single quotes in the emitted pin-env.
    assert ("export TS_" + args.commit[:8] + "_ARTIFACT_EXCEPTION_REASON="
            + shlex.quote('Explicit reviewed test export/common-source difference')) in shell
    assert result['artifact_identity']['serialized_weight_bytes'] == 3
    assert result['artifact_identity']['all_files_bytes'] > 3
    assert result['artifact_identity']['metadata_bytes'] == result['artifact_identity']['all_files_bytes'] - 3
    assert (Path(args.output) / 'artifact-audit.json').read_bytes() == (source.parent / 'audit.json').read_bytes()
    freeze = json.loads((Path(args.output) / 'FREEZE.json').read_text())
    # The files table is computed before FREEZE.json itself is written, exactly
    # like the accepted deployment freeze's own receipt.
    assert set(freeze['files']) == {str(Path(args.output) / name) for name in
                                    ('manifest.json', 'artifact-audit.json', 'pin-env.sh')}
    assert 'not live GO' in freeze['status']


@pytest.mark.parametrize('change', ['audit_digest', 'artifact_path', 'manifest_bytes', 'index_bytes',
                                    'config_bytes', 'missing_exception', 'different_arms', 'same_arm'])
def test_intake_refuses_stale_or_unbound_export_before_writing(fixture, change):
    source, args = fixture
    root = source.parent
    artifact = root / 'actual-export'
    if change == 'audit_digest': (root / 'audit.json').write_text('{}')
    elif change == 'artifact_path':
        p = Path(args.candidate_arm); p.write_text(p.read_text().replace(str(artifact), '/wrong/export'))
    elif change == 'manifest_bytes': (artifact / 'tessera_serving_manifest.json').write_text('{}')
    elif change == 'index_bytes': (artifact / 'model.safetensors.index.json').write_text('{}')
    elif change == 'config_bytes': (artifact / 'config.json').write_text('{}')
    elif change == 'missing_exception': args.artifact_exception_reason = ''
    elif change == 'different_arms':
        p = Path(args.candidate_arm); p.write_text(p.read_text().replace(str(root / 'audit.json'), '/different/audit.json'))
    elif change == 'same_arm': args.candidate_arm = args.reference_arm
    with pytest.raises(ValueError): INTAKE.intake(args)
    assert not Path(args.output).exists()


@pytest.mark.parametrize('change', ['duplicate', 'index_roster', 'serialized_vs_total', 'total_bytes', 'export_revision', 'config_size'])
def test_authenticated_export_still_requires_consistent_serialized_identity(fixture, change):
    source, args = fixture
    root = source.parent
    artifact, audit = root / 'actual-export', root / 'audit.json'
    roster = json.loads(audit.read_text())
    if change == 'duplicate': roster['files'].append(roster['files'][0])
    elif change == 'index_roster':
        path = artifact / 'model.safetensors.index.json'
        path.write_text(json.dumps({'weight_map': {'weight': 'unrostered.safetensors'}}))
    elif change in ('serialized_vs_total', 'export_revision'):
        path = artifact / 'tessera_serving_manifest.json'
        manifest = json.loads(path.read_text())
        if change == 'serialized_vs_total': manifest['totals']['checkpoint_bytes'] = roster['all_files_bytes']
        else: manifest['git'] = 'moving-branch'
        path.write_text(json.dumps(manifest))
    elif change == 'total_bytes': roster['all_files_bytes'] += 1
    elif change == 'config_size':
        next(row for row in roster['files'] if row['name'] == 'config.json')['bytes'] += 1
        roster['all_files_bytes'] += 1
    if change in ('index_roster', 'serialized_vs_total', 'export_revision'):
        row = next(row for row in roster['files'] if row['name'] == path.name)
        row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
        roster['all_files_bytes'] = sum(row['bytes'] for row in roster['files'])
        roster['manifest_sha256'] = next(row['sha256'] for row in roster['files'] if row['name'] == 'tessera_serving_manifest.json')
    audit.write_text(json.dumps(roster))
    digest = hashlib.sha256(audit.read_bytes()).hexdigest()
    for name in (args.reference_arm, args.candidate_arm):
        p = Path(name)
        lines = p.read_text().splitlines()
        p.write_text('\n'.join('ARM_AUDIT_ROSTER_SHA256=' + digest if line.startswith('ARM_AUDIT_ROSTER_SHA256=') else line for line in lines) + '\n')
    with pytest.raises(ValueError): INTAKE.intake(args)
    assert not Path(args.output).exists()


def test_published_gate_consumes_frozen_artifact_not_hardcoded_export(fixture):
    source, args = fixture
    INTAKE.intake(args)
    harness = source.parent / 'harness'
    (harness / 'arms').mkdir(parents=True)
    for filename in (args.reference_arm, args.candidate_arm):
        (harness / 'arms' / Path(filename).name).write_text(Path(filename).read_text())
    gates = (ROOT / 'tools' / 'serve_comparison_gates.sh').read_text()
    gate = re.search(r'^artifact_gate\(\) \{.*?^\}', gates, re.M | re.S).group()
    script = 'BASE_ARM=BASE; ARMS="BASE CANDIDATE"; MTP_ARM=""\n' + gate + '\nartifact_gate\n'
    manifest = Path(args.output) / 'manifest.json'
    env = {**os.environ, 'COMPARISON_MANIFEST': str(manifest), 'H': str(harness), 'LEAD_PIN': args.commit[:8],
           'SERVE_GATE_TOOL_DIR': str(ROOT / 'tools')}
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip() == str(source.parent / 'actual-export')
    raw = json.loads(manifest.read_text()); raw['identities']['tessera_artifact'] = '/stale/export'
    manifest.write_text(json.dumps(raw))
    result = subprocess.run(['bash', '-c', script], env=env, capture_output=True, text=True)
    assert result.returncode != 0


@pytest.mark.parametrize('change', ['arm_bytes', 'same_basename', 'valid_audit_drift', 'different_audits',
                                    'ordered_names', 'metadata_binding', 'malformed_binding'])
def test_runtime_refuses_same_path_frozen_identity_substitution(fixture, change):
    source, args = fixture
    INTAKE.intake(args)
    manifest_path = Path(args.output) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    binding = manifest['identities']['tessera_artifact_binding']
    paths = [Path(args.reference_arm), Path(args.candidate_arm)]
    if change == 'arm_bytes': paths[0].write_text(paths[0].read_text() + '# different loaded definition\n')
    elif change == 'same_basename':
        alternate = source.parent / 'substituted'; alternate.mkdir()
        path = alternate / paths[0].name
        path.write_text(paths[0].read_text() + '# same basename, different bytes\n')
        paths[0] = path
    elif change in ('valid_audit_drift', 'different_audits'):
        audit = source.parent / 'audit.json'
        roster = json.loads(audit.read_text())
        if change == 'valid_audit_drift':
            roster['audit_revision'] = 'new valid roster, same artifact pathname'
            audit.write_text(json.dumps(roster))
            indices = (0, 1)
        else:
            audit = source.parent / 'alternate-audit.json'; audit.write_text(json.dumps(roster))
            indices = (1,)
        digest = hashlib.sha256(audit.read_bytes()).hexdigest()
        for index in indices:
            path = paths[index]
            lines = path.read_text().splitlines()
            path.write_text('\n'.join(
                'ARM_AUDIT_ROSTER_SHA256=' + digest if line.startswith('ARM_AUDIT_ROSTER_SHA256=')
                else 'ARM_AUDIT_ROSTER=' + shlex.quote(str(audit)) if line.startswith('ARM_AUDIT_ROSTER=')
                else line for line in lines) + '\n')
            # Isolate the semantic audit check even when loaded file hashes match.
            binding['arm_inputs'][index].update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    elif change == 'ordered_names': paths.reverse()
    elif change == 'metadata_binding': binding['serialized_weight_bytes'] += 1
    elif change == 'malformed_binding': manifest['identities']['tessera_artifact_binding'] = None
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        OWNER.comparison_artifact(manifest_path, paths, args.commit[:8])


@pytest.mark.parametrize('stem,allowed', [('A_B', True), ('123', True), ('A-B', False), ('CANDIDATÉ', False)])
def test_comparison_arm_can_form_real_shell_pin_identifier(fixture, stem, allowed):
    source, args = fixture
    path = source.parent / (stem + '.env')
    path.write_bytes(Path(args.candidate_arm).read_bytes())
    args.candidate_arm = str(path)
    if allowed:
        INTAKE.intake(args)
        assert 'export TS_PIN_' + stem + '=' in (Path(args.output) / 'pin-env.sh').read_text()
    else:
        with pytest.raises(ValueError): INTAKE.intake(args)
        assert not Path(args.output).exists()


def test_fieldless_historical_manifest_is_explicit_path_only_not_bad_binding_fallback(fixture):
    source, args = fixture
    manifest = source.parent / 'historical.json'
    manifest.write_text(json.dumps({'identities': {'tessera_artifact': str(source.parent / 'actual-export')}}))
    paths = [args.reference_arm, args.candidate_arm]
    assert OWNER.comparison_artifact(manifest, paths, args.commit[:8]) == str(source.parent / 'actual-export')
    manifest.write_text(json.dumps({'identities': {'tessera_artifact': str(source.parent / 'actual-export'),
                                                  'tessera_artifact_binding': {}}}))
    with pytest.raises(ValueError): OWNER.comparison_artifact(manifest, paths, args.commit[:8])
