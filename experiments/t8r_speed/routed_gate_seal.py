"""PB-only bounded frame sealer for the existing routed diagnosis readset.

Independent publisher cached-unit identities authorize canonical outer frames;
unknown outer hashes are recorded only after checking the SAME owned bytes.
This performs no tensor/GPU decode, timings, cache or origin fallback.
"""
import argparse
import base64
import hashlib
import io
import json
import os
import tarfile
from pathlib import Path

from pb_staged_store import StagedInputs

ROOT = '/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported'
MODULE = 'model.language_model.layers.10.mlp.experts'


def seal(manifest_path, *, root=ROOT, modules=(MODULE,)):
    reader = StagedInputs(manifest_path, allow_unsealed_wires=True)
    try:
        reader.json(root + '/config.json')
        index = reader.json(root + '/model.safetensors.index.json')['weight_map']
        published = reader.json(root + '/tessera_serving_manifest.json')
        proof = {}
        for module in modules:
            reader.bind_roles(root, published['modules'][module]['roles'], module=module)
            for name in reader.roles:
                raw = reader.wire(root, name, index=index)
                proof[name] = dict(reader.reads[-1]['cached_member'])
                del raw
        # Every declared range, including the routing capture, must be owned and
        # hash-checked before emitting a complete readset. No staged path reread.
        seen = {(r['path'],r['offset']) for r in reader.reads}
        for identity in reader.entries.keys() - seen:
            reader.read(*identity)
        doc = json.loads(json.dumps(reader.manifest))
        observed = {(r['path'],r['offset']):r for r in reader.reads}
        for entry in doc['entries']:
            got = observed[(entry['path'],entry['offset'])]
            if got['bytes'] != entry['bytes']:
                raise ValueError('sealed length changed')
            entry['sha256'] = got['sha256']
        doc['produced_by'] = {'tool':'PB pinned cached-inner/canonical-outer sealer',
                              'commit':os.environ['TESSERA_HEAD'],
                              'action_key':os.environ['PRISMABUILD_ACTION_KEY']}
        doc['produced_by']['modules'] = list(modules)
        doc['annotations']['row_id'] = ('routed-gate826-sealed-one-BMT128' if modules == (MODULE,)
                                         else 'stageprev793-original-three-layer-readset')
        evidence = {'source':os.environ['TESSERA_HEAD'],
                    'action_key':os.environ['PRISMABUILD_ACTION_KEY'],
                    'provisional_manifest_sha256':reader.manifest_sha256,
                    'wire_count':len(proof), 'entries':len(observed),
                    'total_bytes':doc['total_bytes'], 'members':proof,
                    'reads':reader.reads, 'timing_claim':None, 'gpu_decode':False}
    finally:
        reader.close()
    return doc,evidence



def prepare_stageprev(native_paths, routing_files):
    """Metadata/header inventory only; unknown outer frames are sealed via PB next."""
    import struct
    root = Path("/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported")
    from stageprev_probe import MODULES
    entries, seen, headers = [], set(), {}

    def add(path, offset, length, digest):
        key = (str(path), offset)
        if key in seen:
            raise ValueError(f"duplicate prepared range: {key}")
        seen.add(key)
        entries.append(dict(path=str(path), offset=offset, bytes=length, sha256=digest))

    def full(path):
        raw = path.read_bytes()
        add(path, 0, len(raw), hashlib.sha256(raw).hexdigest())
        return raw

    index = json.loads(full(root / "model.safetensors.index.json"))["weight_map"]
    full(root / "config.json")
    published = json.loads(full(root / "tessera_serving_manifest.json"))
    for module in MODULES:
        roles = published["modules"][module]["roles"]
        checker = StagedInputs.__new__(StagedInputs)
        checker.bind_roles(str(root), roles, module=module)
        for name in checker.roles:
            path = root / index[name]
            if path not in headers:
                with path.open("rb") as handle:
                    first = handle.read(8)
                    if len(first) != 8: raise ValueError("short safetensors header")
                    length = struct.unpack("<Q", first)[0]
                    raw = first + handle.read(length)
                if len(raw) != length + 8: raise ValueError("short safetensors header")
                add(path, 0, len(raw), hashlib.sha256(raw).hexdigest())
                headers[path] = (length, json.loads(raw[8:]))
            length, header = headers[path]
            record = header[name]
            start, end = record["data_offsets"]
            if (record["dtype"] != "U8" or record["shape"] != [end-start] or end <= start):
                raise ValueError("prepared wire extent differs")
            add(path, 8 + length + start, end-start, None)
    if len(native_paths) != 2 or len(set(native_paths)) != 2:
        raise ValueError("prepare requires two distinct original native artifacts")
    for name in native_paths:
        path = Path(name)
        if path.suffix != ".so": raise ValueError("native artifact must be an ELF .so")
        full(path)
    for name in routing_files:
        full(Path(name))
    total = sum(e["bytes"] for e in entries)
    return dict(schema="prismaquant.prismabuild.data_manifest.v1",
                mount_prefix="/mnt/shared", entries=entries, entry_count=len(entries), total_bytes=total,
                produced_by={"tool":"stageprev metadata/header inventory; NOT SEALED",
                             "action_key":os.environ["PRISMABUILD_ACTION_KEY"]},
                annotations={"row_id":"stageprev793-provisional-original-ranges",
                             "modules":list(MODULES), "routing_files":list(routing_files)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('manifest')
    ap.add_argument('--stageprev', action='store_true', help='exact original #793 three-layer readset')
    ap.add_argument('--prepare-stageprev', action='store_true', help='inventory metadata/header ranges only')
    ap.add_argument('--native', action='append', default=[])
    ap.add_argument('--routing-file', action='append', default=[])
    args = ap.parse_args()
    if args.prepare_stageprev:
        prepared = prepare_stageprev(args.native, args.routing_file)
        Path(args.manifest).write_text(json.dumps(prepared, indent=2) + '\n')
        print(json.dumps({'entry_count':prepared['entry_count'],'total_bytes':prepared['total_bytes'],
                          'path':args.manifest,'sealed':False}))
        return
    if args.stageprev:
        from stageprev_probe import MODULES
        manifest,proof = seal(args.manifest,
            root='/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported',
            modules=MODULES)
    else:
        manifest,proof = seal(args.manifest)
    manifest_name = 'stageprev_793_inputs.json' if args.stageprev else 'routed_gate_826_inputs.json'
    files = {manifest_name:json.dumps(manifest,indent=2).encode()+b'\n',
             'sealed-wire-proof.json':json.dumps(proof,indent=2).encode()+b'\n'}
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w:gz') as archive:
        for name,raw in files.items():
            info=tarfile.TarInfo(name);info.size=len(raw);info.mtime=0
            archive.addfile(info,io.BytesIO(raw))
    print(json.dumps({'wire_count':proof['wire_count'],'entries':proof['entries'],
                      'bytes':proof['total_bytes'],
                      'manifest_sha256':hashlib.sha256(files[manifest_name]).hexdigest()}))
    print('ROUTED_GATE_SEAL_BEGIN')
    print(base64.b64encode(stream.getvalue()).decode())
    print('ROUTED_GATE_SEAL_END')


if __name__ == '__main__':
    main()
