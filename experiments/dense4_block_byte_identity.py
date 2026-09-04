"""Hash a fresh export's wire blobs against an existing arm's, unit by unit.

Two claims need this, and one check settles both:

* **The derived-block wiring moves no bytes at a stated block.**  ``for_unit``
  used to call ``block_ldl(regularize_hessian(H, sigma), block)`` inline; it
  now hoists the regularised Hessian out so ``choose_ldl_block`` can see it.
  Same tensor, same call -- but "same by reading" is not the standard here.
* **The commits between an existing arm and this checkout move no bytes
  either.**  ``ldlq-block-serve/b32-tessera`` was written at ``82cdf513``, and
  three merges have touched ``encode.py``/``export.py`` since (#50's exact LUT
  table fit, #75's refit diagnostic, #95's chooser signature).  Each is
  supposed to be opt-in or diagnostic.  If a fresh export at this checkout
  reproduces that arm's blobs exactly, they are, and that arm's served A/B can
  be read as a pair at *this* commit rather than only at its own.

Compares the ``.tessera`` blob of every unit both checkpoints hold, by sha256
of the raw bytes, plus the ``activation_aware`` config block.  A unit only one
side has is reported, not silently skipped -- a subset export is expected
(``--layers``), a *missing* unit inside the overlap is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def blobs(path: Path) -> "dict[str, str]":
    out = {}
    with safe_open(str(path / "model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.endswith(".tessera"):
                t = handle.get_tensor(key).contiguous().cpu()
                out[key] = hashlib.sha256(
                    memoryview(t.numpy()).tobytes()).hexdigest()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", type=Path, help="the existing arm")
    ap.add_argument("fresh", type=Path, help="the arm exported by this checkout")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    ref, new = blobs(args.reference), blobs(args.fresh)
    shared = sorted(set(ref) & set(new))
    same = [k for k in shared if ref[k] == new[k]]
    differ = [k for k in shared if ref[k] != new[k]]

    cfg_ref = json.loads((args.reference / "tessera_config.json").read_text()) \
        if (args.reference / "tessera_config.json").exists() else {}
    cfg_new = json.loads((args.fresh / "tessera_config.json").read_text()) \
        if (args.fresh / "tessera_config.json").exists() else {}
    aware_ref = cfg_ref.get("activation_aware")
    aware_new = cfg_new.get("activation_aware")

    result = {
        "schema": "tessera.dense4_block_byte_identity/1",
        "reference": str(args.reference), "fresh": str(args.fresh),
        "units_reference": len(ref), "units_fresh": len(new),
        "units_compared": len(shared),
        "identical": len(same), "differing": len(differ),
        "differing_names": differ[:20],
        "only_in_reference": len(set(ref) - set(new)),
        "only_in_fresh": sorted(set(new) - set(ref)),
        "activation_aware_reference": aware_ref,
        "activation_aware_fresh": aware_new,
        "activation_aware_equal": aware_ref == aware_new,
        "verdict": ("BYTE-IDENTICAL" if shared and not differ else
                    "DIFFERS" if differ else "NOTHING COMPARED"),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
    raise SystemExit(0 if result["verdict"] == "BYTE-IDENTICAL" else 1)


if __name__ == "__main__":
    main()
