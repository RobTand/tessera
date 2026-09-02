"""Byte and decode baselines for the 2026-09-02 math-audit fix passes.

Fifteen checkpoints exist and the wire is a compatibility surface, so every
audit fix has to say -- and prove -- whether it changes the bytes an encoder
emits or the tensor a reader decodes.  This is the proof harness both halves
of that claim are made with.

    python experiments/audit_byte_baseline.py before.json     # at HEAD
    ...apply the fix...
    python experiments/audit_byte_baseline.py after.json
    python experiments/audit_byte_baseline.py --diff before.json after.json

``encode`` hashes the serialised unit for a fixed matrix of (grid, rung, shape,
seed, weighting) -- the recipes the exporter actually selects, plus the shapes
that exercise a partial trailing superblock.  ``decode`` hashes the tensor every
``.tessera`` file on this box decodes to, which is the half that matters for the
artifacts already written: a fix may legitimately change future bytes, but it
may never change what today's bytes mean.

CPU only, by construction -- it runs while a GPU measurement is in flight.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import zlib

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from tessera.alphabet import E2M1_GRID, E4M3_GRID, tuple_grid
from tessera.export import encode_linear

ARTIFACT_GLOBS = (
    "/home/rob/tessera-runs/*/*/cache/wire/*.tessera",
    "/home/rob/tessera-runs/*/cache/wire/*.tessera",
)

# (label, grid factory, q256, rows, cols).  Column counts 640 and 384 carry a
# partial trailing superblock on purpose -- that is the shape the granule
# arithmetic floors.
def _cases():
    g2 = tuple_grid(E2M1_GRID, 2)
    return [
        ("e2m1x2-cap-512c",   g2,         896, 64, 512),
        ("e2m1x2-cap-640c",   g2,         896, 64, 640),
        ("e2m1x2-cap-384c",   g2,         896, 32, 384),
        ("e2m1x2-sub-512c",   g2,         512, 64, 512),
        ("e2m1-256-512c",     E2M1_GRID,  256, 64, 512),
        ("e4m3-1024-256c",    E4M3_GRID, 1024, 32, 256),
        ("e4m3-1024-320c",    E4M3_GRID, 1024, 32, 320),
    ]


def encode_hashes() -> dict:
    out = {}
    for label, grid, q256, rows, cols in _cases():
        for weighting in ("none", "scale"):
            # zlib, not hash(): PYTHONHASHSEED is randomised per process, so
            # a built-in hash would reseed the weights on every run and every
            # digest below would be noise.
            torch.manual_seed(zlib.crc32(f"{label}/{weighting}".encode()) & 0xFFFF)
            w = torch.randn(rows, cols)
            key = f"{label}/{weighting}"
            try:
                unit = encode_linear(
                    w, grid=grid, q256=q256, trellis_weighting=weighting
                )
            except Exception as exc:            # a refusal is part of the baseline
                out[key] = f"REFUSED {type(exc).__name__}: {exc}"
                continue
            out[key] = hashlib.sha256(unit.blob).hexdigest()
    return out


def decode_hashes() -> dict:
    from tessera.unit_artifact import read_unit_artifact  # late: keeps import cheap

    out = {}
    paths = sorted({p for g in ARTIFACT_GLOBS for p in glob.glob(g)})
    for path in paths:
        key = path.replace("/home/rob/tessera-runs/", "")
        try:
            blob = open(path, "rb").read()
            tensor = read_unit_artifact(blob)
            digest = hashlib.sha256(
                tensor.to(torch.float32).contiguous().numpy().tobytes()
            ).hexdigest()
        except Exception as exc:
            digest = f"FAILED {type(exc).__name__}: {exc}"
        out[key] = digest
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--encode-only", action="store_true")
    a = ap.parse_args()

    if a.diff:
        before = json.load(open(a.diff[0]))
        after = json.load(open(a.diff[1]))
        changed = 0
        for section in ("encode", "decode"):
            b, c = before.get(section, {}), after.get(section, {})
            for key in sorted(set(b) | set(c)):
                if b.get(key) != c.get(key):
                    changed += 1
                    print(f"{section} CHANGED {key}\n    before {b.get(key)}\n    after  {c.get(key)}")
        print(f"{changed} changed of {sum(len(before.get(s, {})) for s in ('encode','decode'))}")
        return 1 if changed else 0

    report = {"encode": encode_hashes()}
    if not a.encode_only:
        report["decode"] = decode_hashes()
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.path:
        open(a.path, "w").write(text + "\n")
        print(f"wrote {a.path}: {len(report['encode'])} encodes, "
              f"{len(report.get('decode', {}))} decodes")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
