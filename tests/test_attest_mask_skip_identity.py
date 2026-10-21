"""Identity-version compatibility for the read-only mask-skip trace audit."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def trace_audit(tmp_path):
    tool = Path(__file__).resolve().parents[1] / "tools" / "tessera_attest.py"
    spec = importlib.util.spec_from_file_location("_mask_skip_identity_attest", tool)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    document = {
        "schema": module.MASK_SKIP_TRACE_SCHEMA,
        "rank": 0, "world_size": 1, "rank_conflict": None,
        "dispatch_coverage": {
            "python_dispatches_counted": True,
            "torch_compile_tracing_counted": False,
            "cuda_graph_replays_counted": False,
        },
        "entries": [
            {"kind": module.MASK_SKIP_KIND, "policy": module.MASK_SKIP_POLICY,
             "contract": module.MASK_SKIP_CONTRACT, "decoder": decoder,
             "symbol": symbol, "launches": 1}
            for decoder, symbol in (
                (module.NATIVE_DECODER, module.NATIVE_SYMBOL),
                (module.STOCK_DECODER, module.STOCK_SYMBOL),
            )
        ],
    }
    path = tmp_path / "trace.json"

    def audit():
        path.write_text(json.dumps(document))
        before = path.read_bytes()
        verdict = module.audit_mla_mask_skip_traces([path], world_size=1)
        assert path.read_bytes() == before
        return verdict

    return document, audit


@pytest.mark.parametrize("version", [2, 0, -1, None, True, 1.0, "1", [], {}])
def test_explicit_unsupported_identity_version_is_refused(trace_audit, version):
    document, audit = trace_audit
    document["identity_version"] = version
    verdict = audit()
    assert verdict["ok"] is False, verdict
    assert any("identity_version" in problem for problem in verdict["problems"]), verdict


@pytest.mark.parametrize("explicit", [False, True], ids=["legacy_missing", "version_1"])
def test_supported_identity_remains_accepted(trace_audit, explicit):
    document, audit = trace_audit
    if explicit:
        document["identity_version"] = 1
    assert audit() == {"ok": True, "problems": []}


def test_unknown_identity_is_refused_before_interpreting_headers(trace_audit):
    document, audit = trace_audit
    document.update(identity_version=2, rank=[0], world_size="future semantics")
    verdict = audit()
    assert verdict["ok"] is False, verdict
    assert any("identity_version" in problem for problem in verdict["problems"]), verdict
    assert not any("rank must" in problem or "world_size does not match" in problem
                   for problem in verdict["problems"]), verdict
