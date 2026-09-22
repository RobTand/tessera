"""Publish corrected native residency metadata without re-encoding immutable wires.

Writes a fresh artifact; safetensors payloads are hard-linked and never modified.
The source artifact is retained and each native role is verified by the normal
metadata parser before its footprint is re-derived. This is no serving receipt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from safetensors import safe_open
from tessera.fused import parse_fused
from tessera.kernel_window_gemv import TILE_ROWS
from tessera.serving_parts import dense_resident_bytes_resident_mode, summarize_modules
from tessera.unit_artifact import parse_unit_metadata


def refresh(source: Path, output: Path):
    source = source.resolve()
    original = (source / 'tessera_serving_manifest.json').read_bytes()
    manifest = json.loads(original)
    index_path = source / 'model.safetensors.index.json'
    index = json.loads(index_path.read_text())['weight_map'] if index_path.exists() else None
    updates = {}
    for name, record in manifest['modules'].items():
        if record['family'] not in ('TESSERA_FP8', 'TESSERA_BF16', 'TESSERA_NVFP4'):
            continue
        if record.get('structure') == 'routed_moe':
            raise ValueError('native dense refresh cannot price routed MoE')
        key = name + '.wire_bytes'
        filename = index[key] if index is not None else 'model.safetensors'
        shard = source / filename
        if shard.resolve().parent != source:
            raise ValueError('checkpoint shard must be local to the source artifact')
        before = shard.stat()
        with safe_open(str(shard), framework='pt', device='cpu') as reader:
            blob = reader.get_tensor(key).numpy().tobytes()
        after = shard.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('source checkpoint changed during footprint derivation')
        roles, trellises = [], set()
        members = parse_fused(blob)
        if [member.name for member in members] != [role['role'] for role in record['roles']]:
            raise ValueError('wire roles disagree with export manifest')
        for member, declared in zip(members, record['roles']):
            metadata = parse_unit_metadata(member.blob)
            a4 = record['family'] == 'TESSERA_NVFP4'
            expected_grid = {'TESSERA_BF16': 'BF16', 'TESSERA_FP8': 'E4M3',
                             'TESSERA_NVFP4': 'E2M1x2'}[record['family']]
            if (metadata.grid.name != expected_grid or metadata.body.name != ('TCQ' if a4 else 'WINDOW')
                    or metadata.rows != declared['rows'] or metadata.columns != declared['cols']
                    or metadata.rows != member.rows):
                raise ValueError('verified wire geometry/family disagrees with manifest')
            role = {'rows': metadata.rows, 'cols': metadata.columns, 'rates': metadata.rates}
            if a4:
                if metadata.span != 2 or metadata.manifest.scale_plane.kind.name != 'LUT':
                    raise ValueError('native A4 needs a span2 LUT wire')
                role.update(arity=metadata.grid.arity, memory=metadata.code.memory,
                            half=metadata.manifest.geometry.half_weights,
                            lut_entries=int(metadata.scale_lut.numel()))
                for rate in set(metadata.rates):
                    trellises.add((metadata.forests[rate], metadata.code))
            else:
                role.update(window_bits=metadata.manifest.window_bits, tile_rows=TILE_ROWS)
            roles.append(role)
        from tessera.decode import replay_table_bytes
        table_bytes = sum(replay_table_bytes(forest, code) for forest, code in trellises)
        old = record['resident_bytes_resident_mode']
        new = dense_resident_bytes_resident_mode(record['family'], record['rows'], record['cols'],
                                               native_roles=roles, trellis_table_bytes=table_bytes)
        record['resident_bytes_resident_mode'] = new
        updates[name] = {'before': old, 'after': new}
    totals = manifest['totals']
    manifest['totals'] = summarize_modules(manifest['modules'], totals['passthrough_bytes'],
                                           totals['checkpoint_bytes'])
    if (source / 'tessera_serving_manifest.json').read_bytes() != original:
        raise ValueError('source manifest changed during footprint derivation')
    output.mkdir(parents=True, exist_ok=False)
    for entry in source.iterdir():
        if entry.is_symlink() or not entry.is_file():
            raise ValueError('source artifact must contain only regular files')
        if entry.name == 'tessera_serving_manifest.json':
            continue
        if entry.suffix == '.safetensors':
            os.link(entry, output / entry.name)
        else:
            shutil.copy2(entry, output / entry.name)
    receipt = {'schema': 'tessera.native_resident_manifest_refresh.v1',
               'source': str(source), 'source_manifest_sha256': hashlib.sha256(original).hexdigest(),
               'producer_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'scope': 'native dense resident footprint only; checkpoint bytes hard-linked unchanged; no new serving qualification',
               'updates': updates}
    manifest['native_resident_refresh'] = {key: value for key, value in receipt.items() if key != 'updates'}
    (output / 'tessera_serving_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'native-resident-refresh.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'native_modules': len(updates),
                      'before': sum(row['before'] for row in updates.values()),
                      'after': sum(row['after'] for row in updates.values())}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    refresh(args.source, args.output)


if __name__ == '__main__':
    main()
