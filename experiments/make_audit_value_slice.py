"""Cut the real (weight, H) slice ``audit_byte_baseline.py``'s value cases encode.

A committed binary nobody can regenerate is an opaque fixture, so this is the
recipe that made it, kept next to the harness that reads it.  It runs once, on
the box that holds the source model and the capture; the fixture it writes is
what every later run of the byte proof uses, on any box.

    python experiments/make_audit_value_slice.py

**Why real, and why this unit.**  The condition the CHANNEL refit's ``B > 0``
hold guards is ``B = w H u^T <= 0``, and it is a property of ``H``'s
*off-diagonal* structure, not of the weight distribution or of ``H``'s diagonal
spread.  Measured on this slice: a synthetic ``H = X^T X / n`` built from
independent columns never fires the condition -- not at a diagonal max/median of
4.4e12, and not with a rank-4/16/64 factor structure -- while the real capture
fires it on 9 of 128 row-refits against the *same* ``randn`` rows those
synthetic arms used, and on 14 of 128 against the real weight.  The like-for-
like pair is the 0 and the 9: swapping the metric is what turns the condition
on, so a synthetic metric would have been another blind row.
``model.layers.2.mlp.down_proj`` is the unit the fix's own census was taken on
(commit 45a8f19: 172 candidates, 97 taken, 78 collapsed at q256=1024 under the
``hessian`` objective).

This says nothing about how often the condition fires in the field -- the census
in 45a8f19 is that measurement.  The slice's only job is to be a weight and a
metric under which the encoder's value arithmetic is *reachable*, so a change to
it moves a digest instead of moving nothing.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
from safetensors import safe_open

from tessera.export import HESSIAN_IDENTITY

MODEL = "/home/rob/models/Qwen3-0.6B/model.safetensors"
CAPTURE = "/home/rob/tessera-runs/ldlq/h_full_qwen06b.pt"
UNIT = "model.layers.2.mlp.down_proj"
ROWS, COLS = 32, 256

OUT = Path(__file__).resolve().parents[1] / "tests" / "data" / "audit_value_slice.pt"


def main() -> int:
    with safe_open(MODEL, framework="pt") as f:
        weight = f.get_slice(UNIT + ".weight")[0:ROWS, 0:COLS].clone()
    payload = torch.load(CAPTURE, map_location="cpu", weights_only=False)
    # A principal submatrix of a second-moment matrix is the second-moment
    # matrix of the same columns, so the slice is a real H for a real weight
    # and not a truncation of one.
    H = payload["H"][UNIT][0:COLS, 0:COLS].float().clone()
    provenance = payload["provenance"]
    identity = {f: provenance[f] for f in HESSIAN_IDENTITY}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "weight": weight,                 # the source dtype, not a cast
            "H": H,                           # fp32: a bf16 cast moves which rows fire
            "unit": UNIT,
            "rows": [0, ROWS],
            "cols": [0, COLS],
            "model": provenance.get("model", MODEL),
            "capture": CAPTURE,
            **identity,
        },
        OUT,
    )
    print(f"wrote {OUT}  weight {tuple(weight.shape)} {weight.dtype}  H {tuple(H.shape)}")
    print(f"  {OUT.stat().st_size} bytes; H identity {identity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
