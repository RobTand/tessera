#!/usr/bin/env python
"""Byte baseline for the LUT-plane block-scale refit's Gauss-Seidel option (#35).

The Gauss-Seidel sweep is an ENCODER-SIDE option on a code path that produces
wire bytes, so the claim "with the option off the encode is byte-identical" is
a claim that has to be proved by re-encoding, not by reading the diff.  This
harness hashes the artifact blob of every arm that touches
``_refit_scales_lut_metric`` -- crucially including the full-H (``metric.ndim
!= 1``) branch, which is the branch the change edits.  An arm matrix that only
covered the plain and diagonal paths would prove nothing about it.

    python experiments/gs_refit_byte_baseline.py before.json   # at HEAD
    ...apply the change...
    python experiments/gs_refit_byte_baseline.py after.json
    python experiments/gs_refit_byte_baseline.py --diff before.json after.json

Real weights and a real Hessian where they exist (``layers.0.self_attn.q_proj``
of Qwen3-0.6B, the unit the LUT receipt argued the mechanism on) plus synthetic
shapes with a synthetic SPD Hessian so the matrix still runs where the capture
does not.  ``--device`` picks where; the encoder is deterministic on either,
and the comparison only ever compares two runs on the SAME device, because a
device change moves matmul reductions and would confound the very digest this
harness exists to hold still.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from tessera.alphabet import E2M1_GRID, tuple_grid  # noqa: E402
from tessera.compensate import block_ldl, regularize_hessian  # noqa: E402
from tessera.export import encode_linear_planes  # noqa: E402

MODEL = "/home/rob/models/Qwen3-0.6B"
H_PATH = "/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"
REAL_UNIT = "model.layers.0.self_attn.q_proj"
Q256 = 896


def _synthetic(rows: int, cols: int, seed: str):
    torch.manual_seed(zlib.crc32(seed.encode()) & 0xFFFF)
    W = torch.randn(rows, cols)
    # An SPD Hessian with real off-diagonal coupling -- a diagonal one would
    # take the separable branch and never exercise the code under test.
    X = torch.randn(4 * cols, cols)
    H = (X.T @ X) / X.shape[0] + torch.eye(cols) * 1e-3
    return W, H


def _cases(device: str):
    g2 = tuple_grid(E2M1_GRID, 2)
    out = []
    if Path(H_PATH).exists() and Path(f"{MODEL}/model.safetensors").exists():
        Hall = torch.load(H_PATH, map_location="cpu", weights_only=False)["H"]
        with safe_open(f"{MODEL}/model.safetensors", framework="pt") as f:
            W = f.get_tensor(REAL_UNIT + ".weight").float().contiguous()
        out.append(("qwen-q_proj", g2, Q256, W.to(device),
                    Hall[REAL_UNIT].float().to(device)))
    for rows, cols in ((64, 512), (32, 384)):
        W, H = _synthetic(rows, cols, f"{rows}x{cols}")
        out.append((f"synth-{rows}x{cols}", g2, Q256, W.to(device), H.to(device)))
    return out


def encode_hashes(device: str) -> dict:
    digests = {}
    for label, grid, q256, W, H in _cases(device):
        h = H.diagonal().clone()
        hn = h / h.mean()
        L = block_ldl(regularize_hessian(H, sigma_reg=1.0), 32)
        arms = {
            "plain": {},
            "refit h^1.0": {"refit_metric": hn},
            "refit full-H": {"refit_metric": H},
            "LDLQ + refit h^1.0": {"ldl": L, "ldl_block": 32, "refit_metric": hn},
            "LDLQ + refit full-H": {"ldl": L, "ldl_block": 32, "refit_metric": H},
        }
        for arm, kw in arms.items():
            key = f"{label}/{arm}"
            try:
                exported, _, _ = encode_linear_planes(
                    W, grid=grid, q256=q256, name=label, verify=False, **kw)
                digests[key] = hashlib.sha256(exported.blob).hexdigest()
            except Exception as exc:                 # a refusal is part of the baseline
                digests[key] = f"REFUSED {type(exc).__name__}: {exc}"
    return digests


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--diff", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.diff:
        before, after = (json.load(open(p)) for p in a.diff)
        changed = [k for k in sorted(set(before) | set(after))
                   if before.get(k) != after.get(k)]
        for k in changed:
            print(f"CHANGED {k}\n    before {before.get(k)}\n    after  {after.get(k)}")
        print(f"{len(changed)} changed of {len(set(before) | set(after))}")
        return 1 if changed else 0
    report = encode_hashes(a.device)
    text = json.dumps(report, indent=2, sort_keys=True)
    if a.path:
        open(a.path, "w").write(text + "\n")
        print(f"wrote {a.path}: {len(report)} encodes")
        print(text)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
