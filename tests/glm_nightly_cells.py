"""The eight GLM cells contract v48 mints on the vLLM nightly image (tessera#702).

One place for their runtime and ids, so the tests that enumerate the packaged
cells name them the same way.  The ids carry the runtime suffix the validator
derives from the scope (``contract.cell_runtime_id_suffix``), because the
f8dbe1a0 GLM cells hold the bare scope ids; the suffix is recomputed here from
the scope rather than copied, so a changed image or mode changes every id.
"""
from __future__ import annotations

import hashlib
import json

#: The image the GLM-5.3 release serves on: eugr nightly 155ce16b plus the
#: nccl230 layer.
NIGHTLY_IMAGE = ("localhost/prismaquant/spark-vllm-nccl230@sha256:"
                 "5be13705acaecc7b4aaf342a84f80d67844c9970ff8375bf9fbeecc9c98ce84a")
NIGHTLY_RUNTIME = {"image": NIGHTLY_IMAGE, "execution_modes": ["eager"],
                   "vllm": "0.30.1rc1.dev336+gaf5b4857e.d20260929", "torch": "2.13.0+cu130"}
NIGHTLY_SUFFIX = "_runtime_" + hashlib.sha256(json.dumps(
    {"image": NIGHTLY_IMAGE, "execution_modes": NIGHTLY_RUNTIME["execution_modes"]},
    sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
#: (family, structure) -> the rungs the qualified nightly stub-B receipt carries.
NIGHTLY_RUNGS = {("TESSERA_E4M3_K1", "dense"): [832, 960, 1024, 1088],
                 ("TESSERA_BF16_K1", "dense"): [832, 880, 960, 1024, 1088],
                 ("TESSERA_E4M3_K1", "routed_moe"): [896, 928, 1024, 1088],
                 ("TESSERA_BF16_K1", "routed_moe"): [1024]}


def nightly_id(family: str, structure: str, regime: str) -> str:
    return f"{family.lower()}_{structure}_sm121_{regime}_resident{NIGHTLY_SUFFIX}"


def nightly_ids(structure: str | None = None) -> list[str]:
    return sorted(nightly_id(family, s, regime) for family, s in NIGHTLY_RUNGS
                  if structure in (None, s) for regime in ("decode", "batch"))
