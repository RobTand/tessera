"""Regress the historical FP4 quantizer receipt against a real runtime.

The archive preserves measured outputs from the old T4 serving images.
These checks do not create a current WINDOW serving attestation.
The tests skip at import where vLLM is absent. A skip is not a pass.
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
ARCHIVE = CHECKOUT / "experiments" / "results" / "t4_activation_quantizers_historical_20261007.json"


@pytest.fixture(scope="module")
def emitted():
    if not torch.cuda.is_available():
        pytest.fail("the serve image is present but this process has no CUDA device; "
                    "a table cannot be attested against a runtime that did not run")
    import attest_activation_quantizer as generator
    return {row["id"]: row for row in generator._run_probes(PROBES)}


def test_the_runtime_still_emits_every_published_vector(emitted):
    import json
    published = json.loads(ARCHIVE.read_text())
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
    """The measured operator output must satisfy the historical grammar."""
    from tessera.serving.activation_attestation import validate_activation_quantizers
    from tessera.serving.contract import require_runtime_image
    import json

    block = json.loads(ARCHIVE.read_text())
    fresh = json.loads(json.dumps(block))
    first = fresh["platforms"]["sm_121"][0]
    first["contracts"]["e2m1_group16_ue4m3_static"]["vectors"] = [
        emitted[row["id"]] for row in
        block["platforms"]["sm_121"][0]["contracts"]["e2m1_group16_ue4m3_static"]["vectors"]]
    platforms = ["sm_121"]
    served = {"sm_121": {"e2m1_group16_ue4m3_static"}}
    validate_activation_quantizers(fresh, platforms=platforms,
                                   cell_contracts=served,
                                   require_image=require_runtime_image)
