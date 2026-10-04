#!/usr/bin/env python3
"""Bind a serve comparison's input intake to the actual audited export identity.

Published from the independently reviewed deployment intake (issue #885). The
deployment's input freeze keeps its own common-source and native-bank
authentication; this module is the export seam that freeze now drives: exact
owned arm .env files declare the export, the shared artifact-identity owner
(`comparison_arm_identity`) authenticates the audit roster and the
config/index/manifest bytes against it, and the comparison manifest is bound
to the derived export identity BEFORE anything is written.

Refusals happen before any output exists:
  - the common source must name an exact commit, whose
    ``src/tessera/serving/runtime_contract.json`` supplies the expected
    contract version;
  - the two arms must be distinct owned files binding exactly the same
    artifact/audit identity (same-artifact, ordered population);
  - a differing export source/contract requires an explicit
    --artifact-exception-reason; that string is a reviewed deviation record,
    never admission authority;
  - the audit roster must not change between identity derivation and
    publication.

The binding records the export's serialized shard bytes, metadata bytes and
full-file bytes as separate currencies. Historical manifests without a
``tessera_artifact_binding`` field keep their explicit pathname-only model
check in the owner; a malformed or drifting new binding never falls back to
that historical mode. This intake is distinct from quality, native, admission
and serve authorization: no existing gate changes owner, and no serving
default, pin, image or panel changes here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparison_arm_identity as owner  # noqa: E402


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args])


def frozen_blob(source, commit, relative):
    raw = git(source, 'show', f'{commit}:{relative}')
    if not raw:
        raise ValueError(f'common source differs from committed bytes: {relative}')
    return raw


def artifact_from_arms(reference_arm, candidate_arm, pin):
    """Use the existing shared owner for exact owned arm files and export metadata."""
    arms = owner.read_comparison_arms([reference_arm, candidate_arm], pin)
    if len(arms) != 2 or arms[0][1] != arms[1][1]:
        raise ValueError('matched arms must bind exactly the same artifact/audit')
    values = arms[0][1]
    identity = owner.export_identity(Path(values['ARM_ARTIFACT']), Path(values['ARM_AUDIT_ROSTER']),
                                     values['ARM_AUDIT_ROSTER_SHA256'])
    identity['arm_inputs'] = [arm[2] for arm in arms]
    return arms[0][0], arms[1][0], identity


def intake(args):
    source = Path(args.source).resolve()
    commit = args.commit
    try:
        actual = git(source, 'rev-parse', commit + '^{commit}').decode().strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError('common source commit unavailable') from exc
    if actual != commit:
        raise ValueError('common source must name an exact commit')
    contract_raw = frozen_blob(source, commit, 'src/tessera/serving/runtime_contract.json')
    contract = json.loads(contract_raw)
    contract_sha = hashlib.sha256(contract_raw).hexdigest()
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_bytes())
    reference_arm, candidate_arm, artifact = artifact_from_arms(args.reference_arm, args.candidate_arm, commit[:8])
    exception = artifact['export_commit'] != commit or artifact['contract_version'] != contract['contract_version']
    if exception and not args.artifact_exception_reason.strip():
        raise ValueError('different export source/contract requires an explicit root-reviewed artifact exception reason')
    manifest['identities']['tessera_candidate_commit'] = commit
    manifest['identities']['tessera_candidate_contract_sha256'] = contract_sha
    manifest['identities']['tessera_artifact'] = artifact['artifact']
    manifest['identities']['tessera_artifact_binding'] = artifact
    audit_raw = Path(artifact['audit_roster']).read_bytes()
    if hashlib.sha256(audit_raw).hexdigest() != artifact['audit_sha256']:
        raise ValueError('audit roster changed before input publication')
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'artifact-audit.json').write_bytes(audit_raw)
    pin = commit[:8]
    env = {
        f'TS_PIN_{reference_arm}': pin, f'TS_PIN_{candidate_arm}': pin,
        'LEAD_BASE_ARM': reference_arm, 'LEAD_GATE_ARM': candidate_arm,
        'COMPARISON_MANIFEST': str(output / 'manifest.json'),
    }
    if exception:
        env.update({f'TS_{pin}_ARTIFACT_EXPORT_COMMIT': artifact['export_commit'],
                    f'TS_{pin}_ARTIFACT_CONTRACT_VERSION': str(artifact['contract_version']),
                    f'TS_{pin}_ARTIFACT_EXCEPTION_REASON': args.artifact_exception_reason})
    (output / 'pin-env.sh').write_text(''.join(f'export {key}={shlex.quote(value)}\n' for key, value in env.items()))
    files = {str(p): {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}
             for p in output.iterdir()}
    result = {'common_commit': commit, 'contract_sha256': contract_sha, 'files': files,
              'artifact_identity': artifact,
              'status': 'input artifact intake only; not deployed, not live GO; '
                        'no quality, admission or serve authorization; source/native/quality gates keep their owners'}
    (output / 'FREEZE.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', required=True, help='common tessera source checkout owning the compared commit')
    ap.add_argument('--commit', required=True, help='exact common source commit')
    ap.add_argument('--manifest', required=True, help='existing comparison manifest JSON to bind')
    ap.add_argument('--reference-arm', required=True)
    ap.add_argument('--candidate-arm', required=True)
    ap.add_argument('--artifact-exception-reason', default='', help='Explicit root-reviewed deviation only; not admission authority')
    ap.add_argument('--output', required=True)
    print(json.dumps(intake(ap.parse_args()), indent=2))


if __name__ == '__main__':
    main()
