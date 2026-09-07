"""PB CPU qualification of an explicit reader alongside an unchanged producer.

This reads retained small wire fixtures; production-shape GPU qualification and
performance measurements remain separate. No producer or serving files change.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*')
                       if p.suffix in {'.py', '.cu', '.cuh', '.cpp', '.h'}):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0')
        digest.update(path.read_bytes())
        digest.update(b'\0')
    return digest.hexdigest()


def load_package(name, root):
    if name in sys.modules:
        raise RuntimeError(f'{name} was imported before the explicit source binding')
    spec = importlib.util.spec_from_file_location(
        name, root / '__init__.py', submodule_search_locations=[str(root)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-package', type=Path, required=True)
    parser.add_argument('--producer-source-sha256', required=True)
    parser.add_argument('--reader-source-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    producer_root = args.producer_package.resolve()
    checkout = Path(__file__).resolve().parents[1]
    reader_root = checkout / 'src/tessera'
    assert source_digest(producer_root) == args.producer_source_sha256
    assert source_digest(reader_root) == args.reader_source_sha256
    producer = load_package('tessera', producer_root)
    reader_name = 'tessera_reader_' + args.reader_source_sha256
    reader = load_package(reader_name, reader_root)
    old = importlib.import_module('tessera.unit_artifact')
    new = importlib.import_module(reader_name + '.unit_artifact')
    old_cache = importlib.import_module('tessera.cached_unit')
    new_cache = importlib.import_module(reader_name + '.cached_unit')
    old_container = importlib.import_module('tessera.container')
    old_error = importlib.import_module('tessera.errors').TesseraError
    new_error = importlib.import_module(reader_name + '.errors').TesseraError
    import torch
    torch.set_num_threads(1)
    rows = []
    for path in sorted((checkout / 'tests/data/legacy').glob('*.tessera')):
        blob = path.read_bytes()
        a = old.read_unit_artifact(blob, device='cpu')
        b = new.read_unit_artifact(blob, device='cpu')
        assert a.dtype == b.dtype and torch.equal(a, b), path.name
        artifact = old_container.parse(blob)
        forged_manifest = replace(artifact.manifest,
                                  encoder_profile_id=hashlib.sha256(b'unknown grid').digest())
        forged = old_container.serialize(forged_manifest, artifact.plane_region)
        damaged = bytearray(blob); damaged[-1] ^= 1
        refused = []
        for kind, data in [('profile', forged), ('payload', bytes(damaged))]:
            for label, module in [('producer', old), ('reader', new)]:
                try:
                    module.read_unit_artifact(data, device='cpu')
                except (old_error, new_error) as exc:
                    refused.append({'case': kind, 'implementation': label,
                                    'error': str(exc)})
                else:
                    raise AssertionError(f'{path.name}: {label} accepted {kind} corruption')
        rows.append({'fixture': path.name, 'wire_sha256': hashlib.sha256(blob).hexdigest(),
                     'shape': list(a.shape), 'dtype': str(a.dtype),
                     'render_sha256': hashlib.sha256(a.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(),
                     'equal': True, 'refusals': refused})
    assert rows
    assert sys.modules['tessera'] is producer and sys.modules[reader_name] is reader
    assert old_cache.encoder_source_sha256() == args.producer_source_sha256
    assert new_cache.encoder_source_sha256() == args.reader_source_sha256
    # The reader closure must stay in its explicitly bound package, and loading
    # it must not redirect any primary producer module to the reader checkout.
    loaded = {}
    for name, module in list(sys.modules.items()):
        if name == 'tessera' or name.startswith('tessera.'):
            root = producer_root
        elif name == reader_name or name.startswith(reader_name + '.'):
            root = reader_root
        else:
            continue
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(root), (name, path, root)
        loaded[name] = str(path)
    assert source_digest(producer_root) == args.producer_source_sha256
    assert source_digest(reader_root) == args.reader_source_sha256
    result = {'passed': True, 'scope': 'CPU retained small-fixture namespace compatibility; not production-shape or performance evidence',
              'producer_source_sha256': args.producer_source_sha256,
              'reader_source_sha256': args.reader_source_sha256,
              'reader_namespace': reader_name, 'torch': str(torch.__version__),
              'python': sys.version, 'device': 'cpu', 'fixtures': rows, 'modules': loaded}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passed': True, 'fixtures': len(rows), 'refusals': sum(len(r['refusals']) for r in rows),
                      'result': str(args.out), 'sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
