"""CPU-only build of the MLA prefill DSO through the ONE build owner.

Runs inside the pinned serving image with no GPU. It builds exactly the DSO
that ``MlaPrefillLibrary`` later loads -- same source, flags, includes,
toolchain resolver and build id -- retains its manifest, and dumps SASS and
resource usage from that artifact.

It does not run the kernel and does not gate on a CUDA device: the SM121 gate
stays in ``MlaPrefillLibrary``, the runtime path. Printing nvcc's version here
is not enforcement; ``resolve_toolchain`` is.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import hashlib
import torch

from tessera.serving.mla_prefill import MlaPrefillBuild


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    ap.add_argument('--p0-buffers', action='store_true')
    ap.add_argument('--p0-wrong-pass', action='store_true')
    args = ap.parse_args()

    if torch.cuda.is_available():
        raise RuntimeError('this row requires disabled GPU visibility')
    # The wrapper has a read-only image root, mounting only /work and the
    # output. Bind the canonical include roots inside that immutable image;
    # no mounted include override can borrow its identity.
    import flashinfer
    from flashinfer.jit.env import FLASHINFER_INCLUDE_DIR, CCCL_INCLUDE_DIRS
    package = Path(flashinfer.__file__).resolve().parent
    if not str(package).startswith('/usr/local/lib/') or '/site-packages/' not in str(package) and '/dist-packages/' not in str(package):
        raise RuntimeError('FlashInfer must come from the immutable image install')
    roots = [Path(p).resolve() for p in CCCL_INCLUDE_DIRS] + [Path(FLASHINFER_INCLUDE_DIR).resolve()]
    if any(not p.is_relative_to(package / 'data') for p in roots):
        raise RuntimeError('FlashInfer/CCCL includes escape the immutable image package')

    build = MlaPrefillBuild(args.out, p0_buffers=args.p0_buffers, p0_wrong_pass=args.p0_wrong_pass)
    print(json.dumps(build.manifest, sort_keys=True), flush=True)

    # Disassemble with the toolchain that compiled it, so the dump describes
    # the retained artifact rather than a different assembler.
    cuobjdump = os.path.join(os.path.dirname(build.nvcc), 'cuobjdump')
    artifacts = []
    for flag, suffix in (('-sass', '.sass'), ('-res-usage', '.res')):
        out = os.path.join(args.out, build.name + suffix)
        with open(out, 'w') as fh:
            subprocess.run([cuobjdump, flag, str(build.library_path)],
                           stdout=fh, stderr=subprocess.STDOUT, check=True)
        data = Path(out).read_bytes()
        if not data:
            raise RuntimeError(f'empty compiler inspection artifact: {out}')
        artifacts.append({'path':out,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    proof = {'build_manifest':str(build.manifest_path), 'build':build.manifest,
             'cuobjdump':cuobjdump,'cuobjdump_sha256':hashlib.sha256(Path(cuobjdump).read_bytes()).hexdigest(),
             'immutable_image_include_roots':[str(p) for p in roots], 'artifacts':artifacts}
    (Path(args.out) / 'build-result.json').write_text(json.dumps(proof,indent=2)+'\n')
    print(f'built {build.library_path} bytes={build.manifest["library_bytes"]} '
          f'sha256={build.manifest["library_sha256"]}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
