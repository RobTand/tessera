"""Shared grouped-window expert builders and numeric helpers."""
import dataclasses

import torch

from tessera import kernel_window_gemv as kg
import window_pack_reference as wpr

L = 14



def _body(rows, cols, rates, seed):
    g = torch.Generator().manual_seed(seed)
    rate = torch.tensor(rates, dtype=torch.int64)
    # int32, not uint8: a rate above 8 (the dense value lane reads 1..14)
    # carries codes a byte cannot hold; at rates 1..8 the codes are unchanged
    return (torch.randint(0, 1 << 16, (rows, cols), generator=g) & ((1 << rate) - 1)).to(torch.int32)



def _states(body, rates, L, init=None):
    mask = (1 << L) - 1
    rate = torch.tensor(rates, dtype=torch.int64)
    bits = body.to(torch.int64)
    rows, cols = body.shape
    states = torch.empty(rows, cols, dtype=torch.int64)
    state = torch.zeros(cols, dtype=torch.int64) if init is None else init.to(torch.int64).clone()
    for t in range(rows):
        state = ((state << rate) | bits[t]) & mask
        states[t] = state
    return states



class Expert:
    """One stacked expert's definition-side facts and its unit."""

    def __init__(self, rows, cols, rates, seed, *, family="value", init=None, device="cuda"):
        self.rows, self.cols, self.rates = rows, cols, tuple(rates)
        self.body = _body(rows, cols, self.rates, seed)
        g = torch.Generator().manual_seed(seed + 101)
        self.values = (torch.randn(1 << L, generator=g) * 0.03).bfloat16()
        self.scale = (torch.rand(rows, generator=torch.Generator().manual_seed(seed + 202)) * 2
                      + 0.25).to(device)
        self.init = init
        rep = wpr.pack_bitstream(self.body, self.rates)
        rep = dataclasses.replace(rep, runs=rep.runs.to(device), words=rep.words.to(device),
                                  perm=rep.perm.to(device))
        if init is not None:
            rep.initial_state = init
        kwargs = {}
        if family == "e4m3":
            codes = torch.randint(0, 256, (1 << L,), generator=g).to(torch.uint8).to(device)
            native = torch.arange(256, dtype=torch.uint8)
            native[0x7F] = 0
            native[0xFF] = 0
            kwargs = {"codes_of_state": codes, "native": native.to(device)}
        self.unit = kg.WindowGemvUnit(
            rep=rep, table=self.values.to(device), scale=self.scale, window_bits=L,
            plan=kg.default_plan(rows, cols, 1), family=family, **kwargs,
        )
        self.states = _states(self.body, self.rates, L, init)

    def reference(self, x, family):
        """Return the FP32 projection with one row-scale epilogue."""
        if family == "e4m3":
            byte = self.unit.native[self.unit.codes_of_state[self.states.cuda()].long()]
            w = byte.view(torch.float8_e4m3fn).float()
        else:
            w = self.values.float().cuda()[self.states.cuda()]
        acc = x.float() @ w.t()
        return acc * self.scale[None, :]



def _quant(x):
    from tessera.serving.native_ops import native_fp8_quant, require_native_fp8_quant
    require_native_fp8_quant("grouped test")
    return native_fp8_quant(x)



def _tol(ref):
    """A SCREEN, not a derived bound: ``5e-3 + 1e-2 * max|ref|`` was picked,
    not derived.  No check in this file, ``test_routed_fused_window.py`` or
    ``test_dense_fused_window.py`` uses it any more (they hold every kernel to
    ``fused_bound``'s derived per-element bounds); it stays importable only
    because ``test_native_window_moe.py`` still imports it, and that file's
    end-to-end comparisons are outside tessera#693 follow-up 3."""
    return 5e-3 + 1e-2 * float(ref.float().abs().max())
