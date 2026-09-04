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


#: Where the ``activation_aware`` block actually lives.  It is written into the
#: serving manifest, not into a ``tessera_config.json`` -- an earlier draft of
#: this script looked only at the latter, found neither side had one, and would
#: have reported ``activation_aware_equal: true`` from ``None == None``.  A
#: comparison that passes because it compared nothing is worse than no
#: comparison, so the source file is reported alongside the verdict and an
#: empty side is called ``NOT COMPARED``.
_AWARE_FILES = ("tessera_serving_manifest.json", "tessera_config.json")


def activation_aware(path: Path) -> "tuple[dict | None, str | None]":
    for name in _AWARE_FILES:
        f = path / name
        if f.exists():
            block = json.loads(f.read_text()).get("activation_aware")
            if block is not None:
                return block, name
    return None, None


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

    aware_ref, src_ref = activation_aware(args.reference)
    aware_new, src_new = activation_aware(args.fresh)

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
        "activation_aware_reference_from": src_ref,
        "activation_aware_fresh": aware_new,
        "activation_aware_fresh_from": src_new,
        "activation_aware_equal": aware_ref == aware_new,
        "activation_aware_compared": bool(aware_ref) and bool(aware_new),
        "verdict": ("BYTE-IDENTICAL" if shared and not differ else
                    "DIFFERS" if differ else "NOTHING COMPARED"),
        "config_verdict": (
            "EQUAL" if aware_ref and aware_new and aware_ref == aware_new else
            "UNEQUAL" if aware_ref and aware_new else "NOT COMPARED"),
    }
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1))
    raise SystemExit(0 if result["verdict"] == "BYTE-IDENTICAL" else 1)


if __name__ == "__main__":
    main()
