"""CPU policy checks for stock timing metadata; no CUDA execution is claimed."""
import contextlib
import sys
from types import SimpleNamespace as NS

import pytest

from test_kda_probe_driver import probe


@pytest.fixture
def stock_standins(probe, monkeypatch, tmp_path):
    binary = tmp_path / "stock.so"
    binary.write_bytes(b"CPU identity fixture")
    leaf = NS(__file__=str(binary))
    parent = NS(_flashkda_C=leaf)
    monkeypatch.setitem(sys.modules, "vllm", parent)
    monkeypatch.setitem(sys.modules, "vllm._flashkda_C", leaf)
    monkeypatch.setattr(probe.torch, "Generator", lambda **_: NS(manual_seed=lambda _: None))
    monkeypatch.setattr(probe.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(probe.torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(probe.torch.cuda, "get_device_properties", lambda _: NS(multi_processor_count=48))
    monkeypatch.setattr(probe.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(probe.torch.cuda, "CUDAGraph", lambda: NS(replay=lambda: None))
    monkeypatch.setattr(probe.torch.cuda, "graph", lambda _: contextlib.nullcontext())

    def call(*_):
        fn = lambda: None
        fn.workspace_bytes = 14598144  # native query stand-in, includes extra varlen tile
        fn.resident_bytes = 100 << 20
        return fn

    monkeypatch.setattr(probe, "kdafwd_call", call)
    monkeypatch.setattr(probe, "kda_steady", lambda *a: {"interval_unix": [0, 1], "graph_ms_per_call": 1.0}, raising=False)
    monkeypatch.setattr(probe, "kda_raw_netdata", lambda *a: {}, raising=False)
    monkeypatch.setattr(probe, "kda_commit", lambda *a: None, raising=False)
    args = NS(out=str(tmp_path), kda_tokens=[512], kda_heads=[32], kda_state_dtypes=["float32"],
              kda_netdata_hosts=["sparky=sparky", "sparklina=sparklina"], kda_rounds=1,
              kda_warm_s=1, kda_steady_s=1, reps=1)
    sampler = NS(window=lambda *a: {"mean_w": 10}, samples=[])
    return probe, args, sampler


@pytest.mark.parametrize("fault", ["missing_prepare", "multiple_prepare", "missing_recurrence", "multiple_recurrence"])
def test_stock_profile_refuses_absent_or_ambiguous_kernel(stock_standins, monkeypatch, fault):
    probe, args, sampler = stock_standins
    kernels = {"_flash_kda_fwd_prepare<stock>": {"mean_us": 10},
               "_flash_kda_fwd_recurrence<stock>": {"mean_us": 20}}
    role = "prepare" if "prepare" in fault else "recurrence"
    if fault.startswith("missing"):
        kernels.pop(f"_flash_kda_fwd_{role}<stock>")
    else:
        kernels[f"_flash_kda_fwd_{role}<other>"] = {"mean_us": 12}

    def profile(fn, reps, trace=None):
        if trace:
            trace.write_bytes(b"CPU profile fixture")
        return kernels

    monkeypatch.setattr(probe, "kernel_device_us", profile)
    with pytest.raises(RuntimeError, match="ambiguous or absent"):
        probe.part_kdafwd(args, sampler)


def test_stock_workspace_comes_from_actual_allocation(stock_standins, monkeypatch):
    probe, args, sampler = stock_standins

    def profile(fn, reps, trace=None):
        if trace:
            trace.write_bytes(b"CPU profile fixture")
        return {"_flash_kda_fwd_prepare<stock>": {"mean_us": 10},
                "_flash_kda_fwd_recurrence<stock>": {"mean_us": 20}}

    monkeypatch.setattr(probe, "kernel_device_us", profile)
    out = probe.part_kdafwd(args, sampler)
    cell = out["cells"][0]
    assert cell.get("workspace_bytes_per_copy", cell.get("workspace_bytes")) == 14598144
