"""Regenerate the fp4 quantizer table and require the runtime to still emit it (#484).

This module needs the serving runtime itself, so it skips at import wherever
vLLM is absent -- the ``pure`` job, and any box outside the serve image.  A
skip is not a pass and this repository already says so out loud: ``conftest``
prints every skip reason verbatim and counts them, so a run that never reached
this module says as much in its own summary.  The check that actually runs is
the one inside the image the platform's cells attest::

    docker run --gpus all -v <checkout>:/tessera:ro --entrypoint python3 \\
        <the platform's serve_image> -m pytest /tessera/tests/test_activation_attestation_runtime.py

What it asserts is the whole contract of the table: the packaged vectors are
what THIS runtime emits today.  When the pinned vLLM moves and this fails, the
answer is to regenerate the table, never to widen anything -- a published code
that no longer matches the kernel is a stale attestation, and a consumer gate
reading it would price an activation the runtime does not execute.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="the serve runtime's own torch")
pytest.importorskip("vllm", reason="the fp4 activation quantizer is vLLM's operator")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from tessera.serving.activation_attestation import PROBES

CHECKOUT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def emitted():
    if not torch.cuda.is_available():
        pytest.fail("the serve image is present but this process has no CUDA device; "
                    "a table cannot be attested against a runtime that did not run")
    import attest_activation_quantizer as generator
    return {row["id"]: row for row in generator._run_probes(PROBES)}


def test_the_runtime_still_emits_every_published_vector(emitted):
    import json
    from tessera.serving.contract import contract_path

    published = json.loads(contract_path().read_text())["activation_quantizers"]
    entries = published["platforms"]["sm_121"]
    assert isinstance(entries, list) and entries, (
        "sm_121 must publish one attestation per image, not a bare table")
    matched = [entry["generated"]["image"] for entry in entries
               if all(emitted[row["id"]] == row
                      for row in entry["contracts"]["e2m1_group16_ue4m3_static"]["vectors"])]
    assert matched, (
        "this runtime reproduces none of the packaged sm_121 attestations; "
        "regenerate the table with experiments/attest_activation_quantizer.py, "
        "and do not widen a consumer's tolerance to absorb the difference")


def test_the_generated_table_passes_the_packaged_grammar(emitted):
    """The generator's output is admissible on its own, not only equal."""
    from tessera.serving.activation_attestation import validate_activation_quantizers
    from tessera.serving.contract import contract_path, require_runtime_image
    import json

    raw = json.loads(contract_path().read_text())
    lane = raw["lane_eligibility"]
    served: dict = {}
    for cell in lane["cells"]:
        served.setdefault(cell["platform"], set()).add(cell["activation_contract"])
    block = raw["activation_quantizers"]
    fresh = json.loads(json.dumps(block))
    first = fresh["platforms"]["sm_121"][0]
    first["contracts"]["e2m1_group16_ue4m3_static"]["vectors"] = [
        emitted[row["id"]] for row in
        block["platforms"]["sm_121"][0]["contracts"]["e2m1_group16_ue4m3_static"]["vectors"]]
    validate_activation_quantizers(fresh, platforms=sorted(lane["platforms"]),
                                   cell_contracts=served,
                                   require_image=require_runtime_image)
