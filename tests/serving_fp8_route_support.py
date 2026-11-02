"""Shared FP8 schemes, vLLM stubs and reference encodings."""
from __future__ import annotations
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from tessera.serving.scheme import TESSERA_FP8


CUDA = torch.cuda.is_available()


requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


def _tessera():
    return (pytest.importorskip("tessera.fused"), pytest.importorskip("tessera.export"),
            pytest.importorskip("tessera.stock"), pytest.importorskip("tessera.alphabet"))


def _scheme(rows=256, columns=1024, roles=None, **over):
    s = {"family": TESSERA_FP8, "grid": "E4M3", "body": "WINDOW", "plane": "CHANNEL", "q256": 1024,
         "rows": rows, "columns": columns, "wire_bytes": 4096,
         "roles": roles if roles is not None else [["weight", rows]]}
    s.update(over)
    return s


def _install_vllm_stubs(monkeypatch):
    class _LinearMethodBase:
        pass

    def _param(data, **_kw):
        return torch.nn.Parameter(data, requires_grad=False)

    linear = types.ModuleType("vllm.model_executor.layers.linear")
    linear.LinearMethodBase = _LinearMethodBase
    parameter = types.ModuleType("vllm.model_executor.parameter")
    parameter.ModelWeightParameter = _param
    parameter.BasevLLMParameter = _param
    for name, mod in (("vllm", types.ModuleType("vllm")),
                      ("vllm.model_executor", types.ModuleType("vllm.model_executor")),
                      ("vllm.model_executor.layers", types.ModuleType("vllm.model_executor.layers")),
                      ("vllm.model_executor.layers.linear", linear),
                      ("vllm.model_executor.parameter", parameter)):
        monkeypatch.setitem(sys.modules, name, mod)


_LAST_A = {}


FP8_MAX = 448.0


def _reference_fp8_quant(x):
    """The tests' own per-token dynamic E4M3 quantiser, recording the value it
    represents so the expectation is the A side the route consumed.  It is NOT
    vLLM's op's arithmetic: the op divides where this multiplies by a
    reciprocal and floors its scale at 1 / (448 * 512); the oracles here feed
    the same codes to both sides, so the difference is invisible to them and
    is held by ``tests/test_native_fp8_quant.py`` instead."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (xf / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    _LAST_A["value"] = q.float() * scale
    return q, scale.contiguous()


class _Layer(torch.nn.Module):
    """A vLLM ``LinearBase`` stand-in on one rank: the layer's OWN TP
    coordinates, which every ``LinearBase`` sets before ``create_weights``
    and the shard plan reads (tessera#303)."""

    tp_rank, tp_size = 0, 1


def _encode_module(roles, cols=1024, q256=1024, seed=0):
    """Encode ``roles`` = [(name, rows)] with Tessera on the E4M3 grid; return
    the container blob, the scheme and the stock reference pair."""
    fused, export, stock, alphabet = _tessera()
    torch.manual_seed(seed)
    tensors, blobs = {}, []
    for i, (name, rows) in enumerate(roles):
        w = (torch.randn(rows, cols, device="cuda") * 0.02)
        w[: rows // 8] *= 2.0 ** (i + 1)
        exported, unit, forests = export.encode_linear_planes(
            w.contiguous(), grid=alphabet.E4M3_GRID, q256=q256, name=name, verify=False)
        tensors[name] = stock.materialize_stock(unit, forests, export.DEFAULT_CODE)
        blobs.append((name, rows, exported.blob))
    blob = fused.pack_fused(blobs)
    scheme = _scheme(rows=sum(r for _, r in roles), columns=cols, wire_bytes=len(blob),
                     roles=[[n, r] for n, r in roles], q256=q256)
    weight = torch.cat([tensors[n]["weight"].view(torch.uint8) for n, _ in roles])
    scale = torch.cat([tensors[n]["weight_scale"].reshape(-1) for n, _ in roles])
    ref_w = torch.cat([stock.stock_dequant(tensors[n]) for n, _ in roles])
    return blob, scheme, weight, scale, ref_w
