"""Per-rank eager trace audit for the mask-skip override (tessera#1177).

The override emits route records under one policy; the audit reads the
per-rank trace files back and proves the override ran on eager prefill on
every rank while decode stayed stock. It fails closed on any unreadable,
mismatched or silent file.
"""

import importlib.util
import json
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "tessera_attest.py"


def _attest():
    spec = importlib.util.spec_from_file_location("_tessera_attest_mask_skip", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _trace(rank, world, entries, coverage=None):
    return {
        "schema": "tessera.route_trace/1",
        "identity_version": 1,
        "rank": rank,
        "world_size": world,
        "rank_source": "torch.distributed",
        "rank_conflict": None,
        "platform": "sm_121",
        "serving_source_sha256": "0" * 64,
        "backend_execution_identity": None,
        "backend_execution_identity_conflict": None,
        "dispatch_coverage": coverage or {
            "python_dispatches_counted": True,
            "torch_compile_tracing_counted": False,
            "cuda_graph_replays_counted": False,
        },
        "pid": 1000 + rank,
        "started_utc": "2026-10-10T00:00:00+00:00",
        "flushed_utc": "2026-10-10T00:01:00+00:00",
        "flushes": 3,
        "note": "synthetic",
        "entries": entries,
    }


def _entry(decoder, symbol, launches=44):
    return {
        "policy": "mla_mask_skip_eager_dispatch",
        "shape": "T2048:H32:D512",
        "symbol": symbol,
        "decoder": decoder,
        "contract": "stock_kernel_overrides",
        "kind": "attention_backend",
        "launches": launches,
        "modules": 1,
        "module_names": ["model.layers.0.self_attn"],
        "unnamed_modules": 0,
        "dispatches_without_prefix": 0,
    }


def _write(path, payload):
    path.write_text(json.dumps(payload) + "\n")
    return str(path)


def _good_pair(tmp_path):
    entries = [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0"),
               _entry("stock", "stock", launches=12)]
    return [_write(tmp_path / f"rank{r}.json", _trace(r, 2, entries)) for r in (0, 1)]


def test_audit_accepts_override_active_on_both_ranks(tmp_path):
    attest = _attest()
    record = attest.audit_mla_mask_skip_traces(_good_pair(tmp_path), world_size=2)
    assert record["schema"] == "tessera.mask-skip-audit/1"
    assert record["passed"] is True
    assert record["problems"] == []
    assert [r["rank"] for r in record["ranks"]] == [0, 1]
    for rank in record["ranks"]:
        assert rank["native_launches"] == 44
        assert rank["stock_launches"] == 12


def test_audit_refuses_unreadable_file(tmp_path):
    attest = _attest()
    record = attest.audit_mla_mask_skip_traces([str(tmp_path / "absent.json")],
                                               world_size=1)
    assert record["passed"] is False
    assert any("absent.json" in p and "unreadable" in p for p in record["problems"])


def test_audit_refuses_foreign_schema(tmp_path):
    attest = _attest()
    payload = _trace(0, 1, [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0")])
    payload["schema"] = "tessera.route_trace/0"
    path = _write(tmp_path / "rank0.json", payload)
    record = attest.audit_mla_mask_skip_traces([path], world_size=1)
    assert record["passed"] is False
    assert any("schema" in p for p in record["problems"])


def test_audit_refuses_trace_without_rank(tmp_path):
    attest = _attest()
    payload = _trace(0, 1, [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0")])
    del payload["rank"]
    path = _write(tmp_path / "rank0.json", payload)
    record = attest.audit_mla_mask_skip_traces([path], world_size=1)
    assert record["passed"] is False
    assert any("no rank" in p for p in record["problems"])


def test_audit_refuses_missing_rank(tmp_path):
    attest = _attest()
    entries = [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0"),
               _entry("stock", "stock", launches=12)]
    path = _write(tmp_path / "rank0.json", _trace(0, 2, entries))
    record = attest.audit_mla_mask_skip_traces([path], world_size=2)
    assert record["passed"] is False
    assert any("one trace per rank" in p for p in record["problems"])


def test_audit_refuses_world_mismatch(tmp_path):
    attest = _attest()
    record = attest.audit_mla_mask_skip_traces(_good_pair(tmp_path), world_size=1)
    assert record["passed"] is False
    assert any("world" in p for p in record["problems"])


def test_audit_refuses_unknown_decoder(tmp_path):
    attest = _attest()
    entries = [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0"),
               _entry("mystery", "mystery", launches=5)]
    paths = [_write(tmp_path / "rank0.json", _trace(0, 2, entries)),
             _write(tmp_path / "rank1.json", _trace(1, 2, entries))]
    record = attest.audit_mla_mask_skip_traces(paths, world_size=2)
    assert record["passed"] is False
    assert any("mystery" in p for p in record["problems"])


def test_audit_refuses_rank_without_native_entry(tmp_path):
    attest = _attest()
    quiet = [_entry("stock", "stock", launches=12)]
    paths = [_write(tmp_path / "rank0.json", _trace(0, 2, quiet)),
             _write(tmp_path / "rank1.json", _trace(1, 2, quiet))]
    record = attest.audit_mla_mask_skip_traces(paths, world_size=2)
    assert record["passed"] is False
    assert any("native_mg_mask_skip" in p for p in record["problems"])


def test_audit_refuses_rank_without_stock_entry(tmp_path):
    attest = _attest()
    eager_only = [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0")]
    paths = [_write(tmp_path / "rank0.json", _trace(0, 2, eager_only)),
             _write(tmp_path / "rank1.json", _trace(1, 2, eager_only))]
    record = attest.audit_mla_mask_skip_traces(paths, world_size=2)
    assert record["passed"] is False
    assert any("fallback" in p for p in record["problems"])


def test_audit_refuses_graph_replays_counted(tmp_path):
    attest = _attest()
    coverage = {"python_dispatches_counted": True,
                "torch_compile_tracing_counted": False,
                "cuda_graph_replays_counted": True}
    entries = [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0"),
               _entry("stock", "stock", launches=12)]
    paths = [_write(tmp_path / f"rank{r}.json", _trace(r, 2, entries, coverage))
             for r in (0, 1)]
    record = attest.audit_mla_mask_skip_traces(paths, world_size=2)
    assert record["passed"] is False
    assert any("graph" in p for p in record["problems"])


def test_receipt_path_still_refuses_without_out(capsys):
    attest = _attest()
    assert attest.main([]) == 2
    assert "refusing" in capsys.readouterr().err


def test_cli_writes_audit_record_from_repeated_flags(tmp_path):
    attest = _attest()
    paths = _good_pair(tmp_path)
    out = str(tmp_path / "audit.json")
    argv = []
    for path in paths:
        argv += ["--mask-skip-traces", path]
    assert attest.main(argv + ["--expect-world-size", "2", "--audit-out", out]) == 0
    record = json.loads(Path(out).read_text())
    assert record["passed"] is True
    assert [r["rank"] for r in record["ranks"]] == [0, 1]


def test_cli_refuses_audit_without_world_size(tmp_path, capsys):
    attest = _attest()
    paths = _good_pair(tmp_path)
    out = str(tmp_path / "audit.json")
    assert attest.main(["--mask-skip-traces", ",".join(paths),
                        "--audit-out", out]) == 2
    assert not Path(out).exists()
    assert "refusing" in capsys.readouterr().err


@pytest.mark.parametrize("world_size", [0, True, 1.0])
def test_audit_refuses_non_integer_or_empty_world(tmp_path, world_size):
    attest = _attest()
    entries = [_entry("native_mg_mask_skip", "tessera_mla_prefill_mg_l0"),
               _entry("stock", "stock", launches=12)]
    paths = [] if world_size == 0 else [
        _write(tmp_path / "rank0.json", _trace(0, 1, entries))]
    record = attest.audit_mla_mask_skip_traces(paths, world_size=world_size)
    assert record["passed"] is False


@pytest.mark.parametrize("rank", [-1, 2, True, 1.0, "1", [], {}])
def test_audit_refuses_invalid_rank_without_raising(tmp_path, rank):
    attest = _attest()
    paths = _good_pair(tmp_path)
    payload = json.loads(Path(paths[1]).read_text())
    payload["rank"] = rank
    _write(Path(paths[1]), payload)
    record = attest.audit_mla_mask_skip_traces(paths, world_size=2)
    assert record["passed"] is False


@pytest.mark.parametrize("field", ["cuda_graph_replays_counted", "torch_compile_tracing_counted"])
def test_audit_refuses_unknown_dispatch_coverage(tmp_path, field):
    attest = _attest()
    paths = _good_pair(tmp_path)
    payload = json.loads(Path(paths[1]).read_text())
    del payload["dispatch_coverage"][field]
    _write(Path(paths[1]), payload)
    record = attest.audit_mla_mask_skip_traces(paths, world_size=2)
    assert record["passed"] is False


def test_audit_refuses_rank_conflict(tmp_path):
    attest = _attest()
    paths = _good_pair(tmp_path)
    payload = json.loads(Path(paths[1]).read_text())
    payload["rank_conflict"] = {"reported": 0, "observed": 1}
    _write(Path(paths[1]), payload)
    record = attest.audit_mla_mask_skip_traces(paths, world_size=2)
    assert record["passed"] is False


@pytest.mark.parametrize("reason,decoder,schedule", [
    (None, "native_mg_mask_skip", "mg_mask_skip_pass_buffers"),
    ("decode remains stock", "stock", "stock_mg"),
])
def test_emitter_role_selects_the_executed_decoder_and_schedule(reason, decoder, schedule):
    pytest.importorskip("torch")
    from types import SimpleNamespace
    from tessera.serving.mla_mask_skip_identity import emit_mask_skip_route
    from tessera.serving.telemetry import read_route

    layer = SimpleNamespace(prefix="model.layers.0.self_attn")
    emit_mask_skip_route(layer, shape="T2048:H32:D512",
                        symbol="stock" if reason else "tessera_mla_prefill_mg_l0",
                        reason=reason)
    record = read_route(layer)
    assert record["decoder"] == decoder
    assert record["kernel_schedule"] == schedule
    assert record["policy"] == "mla_mask_skip_eager_dispatch"
    assert record["kind"] == "attention_backend"
    assert record["contract"] == "stock_kernel_overrides"
