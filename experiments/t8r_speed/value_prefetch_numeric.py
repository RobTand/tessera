"""Frozen syntheticgeometry/nonshipping T16 two-arm qualification (#874).

CPU preparation seals fixtures and retained native bytes. GPU consumption uses
PB pinned ranges and the existing native build callback; no JIT or origin reads.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'src'))
from test_window_gemm_grouped import Expert
from test_routed_fused_window import Q256_CASES, _sched, _init, _bundles, _staged_check
from tessera import routed_fused as rf
from experiments.t8r_speed.pb_staged_store import NativeCallback, StagedInputs

SCOPE = 'syntheticgeometry/nonshipping'
SOURCE_SHA = '51f6f7d76e0f7f54e448202599b98007c88a7ebb1e2d9c69ba0324258fb892c9'
ARMS = {
    'baseline': ('tessera_routed_fused_value', 0, 'ba853f65afa3755df443ac903b74f06fda5fa5cb8c1d8c9b26e5a4c9843f8072'),
    'candidate': ('tessera_routed_fused_value_prefetch4', 4, '52e8c6c8ac3cc8a61f0041b43240f61076a2281f842af13703f693695547319e'),
}
SANITIZER_SHA = '7a7fcdefb67042731daf021478176f4919e1843d0b10cb697af28a7d8a3d108b'
MS = [1, 7, 71]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def entry(path):
    return {'path': str(path), 'offset': 0, 'bytes': path.stat().st_size, 'sha256': digest(path)}


def compare_bits(left, right):
    if left.dtype != right.dtype or left.shape != right.shape:
        raise ValueError('two-arm output metadata differs')
    if not torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)):
        raise ValueError('two-arm output bits differ')


def prepare(bank, native_root, sanitizer):
    bank.mkdir(parents=True, exist_ok=False)
    if digest(sanitizer) != SANITIZER_SHA:
        raise ValueError('sanitizer tool identity differs')
    source = ROOT / 'src/tessera/serving/csrc/routed_fused_window.cu'
    if digest(source) != SOURCE_SHA:
        raise ValueError('native source identity differs')
    retained_source = bank / "routed_fused_window.cu"
    shutil.copyfile(source, retained_source)
    records = [entry(retained_source)]
    (bank / "sanitizer").mkdir()
    for tool_file in sorted(sanitizer.parent.iterdir()):
        if tool_file.is_file():
            retained_tool = bank / "sanitizer" / tool_file.name
            shutil.copyfile(tool_file, retained_tool)
            retained_tool.chmod(tool_file.stat().st_mode & 0o777)
            records.append(entry(retained_tool))
    for arm, (module, distance, expected) in ARMS.items():
        original = native_root / ("ext-" + arm) / (module + "_sm_121_tessera_guarded_v1")
        for name in [module + ".so", "native-finalization.json", "build.ninja"]:
            target = bank / (arm + "-" + name)
            shutil.copyfile(original / name, target)
            records.append(entry(target))
        if digest(bank / (arm + "-" + module + ".so")) != expected:
            raise ValueError("retained native arm differs")
    cases = []
    for q in Q256_CASES:
        stacks = []
        for role, rows, cols, offset in [('gate', 576, 256, 0), ('up', 576, 256, 10), ('down', 256, 576, 20)]:
            stack = [Expert(rows, cols, _sched(cols, q), 300 + offset + e,
                            family='value', device='cpu',
                            init=_init(cols, 340 + offset + e) if e % 2 else None)
                     for e in range(5)]
            stacks.append(stack)
        samples = []
        for m in MS:
            g = torch.Generator().manual_seed(874000 + q + m)
            x = torch.randn(m, 256, generator=g).bfloat16()
            ids = torch.randint(0, 5, (m, 3), generator=g, dtype=torch.int32)
            weights = torch.rand(m, 3, generator=g)
            weights /= weights.sum(1, keepdim=True)
            samples.append({'m': m, 'x': x, 'ids': ids, 'weights': weights})
        target = bank / f'q{q}.pt'
        torch.save({'scope': SCOPE, 'q256': q, 'stacks': stacks, 'samples': samples}, target)
        records.append(entry(target))
        cases.append({'q256': q, 'path': str(target), 'ms': MS})
    packet = {'scope': SCOPE, 'family': 'value', 'arithmetic': 'folded',
              'hidden': 256, 'inter': 576, 'experts': 5, 'top_k': 3, 'bm': 64,
              'cases': cases, 'arms': ARMS, 'source_sha256': SOURCE_SHA,
              'sanitizer_sha256': SANITIZER_SHA, 'actual_calibrated_acceptance': False}
    packet_path = bank / 'packet.json'
    packet_path.write_text(json.dumps(packet, indent=2) + '\n')
    records.append(entry(packet_path))
    manifest = {'schema': 'prismaquant.prismabuild.data_manifest.v1',
                'produced_by': {'tool': 'T16 synthetic fixture sealer', 'action_key': os.environ.get('PRISMABUILD_ACTION_KEY')},
                'mount_prefix': '/mnt/shared', 'entries': records,
                'entry_count': len(records), 'total_bytes': sum(e['bytes'] for e in records)}
    path = bank / 'readset.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'scope': SCOPE, 'readset': str(path), 'sha256': digest(path),
                      'packet_sha256': digest(packet_path), 'entry_count': len(records), 'total_bytes': manifest['total_bytes']}))


def move(value):
    if isinstance(value, torch.Tensor):
        return value.cuda()
    if dataclasses.is_dataclass(value):
        return dataclasses.replace(value, **{f.name: move(getattr(value, f.name)) for f in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {k: move(v) for k, v in value.items()}
    return value


def consume(bank, manifest, out):
    if not torch.cuda.is_available():
        raise ValueError('synthetic numeric consumer requires admitted CUDA')
    out.mkdir(parents=True, exist_ok=False)
    reader = StagedInputs(manifest)
    reports = []
    try:
        packet = json.loads(reader.read(bank / 'packet.json'))
        if packet['scope'] != SCOPE or packet['source_sha256'] != SOURCE_SHA or packet['arms'] != {k: list(v) for k,v in ARMS.items()}:
            raise ValueError("frozen synthetic packet differs")
        expected_cases = [{"q256": q, "path": str(bank / f"q{q}.pt"), "ms": MS} for q in Q256_CASES]
        if packet["cases"] != expected_cases:
            raise ValueError("synthetic case matrix differs")
        for case in packet['cases']:
            # Pickle is permitted only for this digested, explicitly trusted synthetic producer.
            payload = torch.load(io.BytesIO(reader.read(case['path'])), map_location='cpu', weights_only=False)
            if payload['scope'] != SCOPE or payload['q256'] != case['q256']:
                raise ValueError("synthetic fixture differs")
            if [s["m"] for s in payload["samples"]] != MS:
                raise ValueError("synthetic sample matrix differs")
            stacks = payload['stacks']
            for stack in stacks:
                for expert in stack:
                    expert.unit = move(expert.unit)
                    expert.scale = expert.scale.cuda()
            bundles = _bundles('value', stacks)
            samples = [{**s, 'x': s['x'].cuda(), 'ids': s['ids'].cuda(), 'weights': s['weights'].cuda()} for s in payload['samples']]
            baseline = []
            for arm, (module, distance, expected) in ARMS.items():
                rf._ext.cache_clear()
                os.environ['TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH'] = str(distance)
                owner = NativeCallback(reader, bank / (arm + '-' + module + '.so'), rf,
                                       out / f'q{case["q256"]}' / arm,
                                       expected_sha256=expected, source_sha256=SOURCE_SHA,
                                       module=module, source_module='tessera_routed_fused_value')
                try:
                    for i, sample in enumerate(samples):
                        outputs = _staged_check(stacks, bundles, sample['x'], sample['ids'], sample['weights'],
                                                'value', f'{SCOPE}:{arm}:q{case["q256"]}:M{sample["m"]}', compact=False)
                        frozen = [o.detach().clone() for o in outputs]
                        if arm == 'baseline':
                            baseline.append(frozen)
                        else:
                            for left, right in zip(baseline[i], frozen):
                                compare_bits(left, right)
                            reports.append({'q256': case['q256'], 'm': sample['m'], 'stages': ['mode1', 'activation', 'mode2', 'mode0'], 'bitwise': True})
                finally:
                    owner.finish(torch.cuda.synchronize)
                    rf._ext.cache_clear()
        if len(reports) != len(Q256_CASES) * len(MS):
            raise ValueError("incomplete synthetic numeric result")
        (out / "numeric.json").write_text(json.dumps({"scope": SCOPE, "results": reports, "readset_sha256": reader.manifest_sha256}, indent=2) + "\n")
    finally:
        reader.close()


def sanitize(bank, manifest, out):
    import subprocess
    reader = StagedInputs(manifest)
    tool_dir = out / "sanitizer"
    tool_dir.mkdir(parents=True, exist_ok=False)
    try:
        for path, offset in reader.entries:
            if Path(path).parent == bank / "sanitizer":
                target = tool_dir / Path(path).name
                target.write_bytes(reader.read(path, offset))
                target.chmod(0o755)
        tool = tool_dir / "compute-sanitizer"
        if digest(tool) != SANITIZER_SHA:
            raise ValueError("staged sanitizer identity differs")
        command = [str(tool), "--tool", "memcheck", "--error-exitcode", "99",
                   "--target-processes", "all", sys.executable, str(Path(__file__).resolve()),
                   "consume", "--bank", str(bank), "--manifest", str(manifest),
                   "--out", str(out / "numeric")]
        subprocess.run(command, check=True)
    finally:
        reader.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("operation", choices=["prepare", "consume", "sanitize"])
    ap.add_argument('--bank', type=Path, required=True)
    ap.add_argument('--native-root', type=Path)
    ap.add_argument('--sanitizer', type=Path)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    if a.operation == 'prepare':
        prepare(a.bank, a.native_root, a.sanitizer)
    elif a.operation == "sanitize":
        sanitize(a.bank, a.manifest, a.out)
    else:
        consume(a.bank, a.manifest, a.out)


if __name__ == '__main__':
    main()
