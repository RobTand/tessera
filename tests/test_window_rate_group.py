"""A window span's rate calls run as one group, and the group moves no byte.

The LDLQ window body used to yield one Viterbi call per rate per span and wait
for that call's answer before it built the next one, so the latency-bound step
chains of a span's two rates ran one after the other on the device.  The span
now yields every rate's call at once, and the batch driver runs them on
separate CUDA streams (``encode._run_group``).  Two properties are pinned:

* the unit's generator yields a span's calls together: one ``_TrellisCall``
  per rate present in the span, rates ascending, and together they cover the
  span's columns.  Before the change the first yield was a single call, so
  that test fails there;
* the group run on streams writes the same bytes, codes, anchors, body bits
  and planes as the same group run one rate at a time on the caller's stream
  (``TESSERA_WINDOW_RATE_STREAMS=0``), at the mixed-rate BF16 rung the PACT
  campaign prices (R4 and R5 columns inside every LDLQ block).
"""
import dataclasses

import pytest
import torch

from tessera.alphabet import BF16_GRID
from tessera.compensate import block_ldl, regularize_hessian
from tessera.encode import _TrellisCall, _encode_unit_steps
from tessera.export import (
    DEFAULT_CODE, ActivationSource, _plan_for, _resolve_recipe, encode_linears_planes,
    wire_recipe)
from tessera.manifest import BodyKind, ScalePlaneKind

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the rate streams are CUDA")

ROWS, COLS, BLOCK = 64, 256, 32
Q256 = 1088      # 4.25 bits: R4 and R5 columns inside every block


def _weights(seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    w = torch.randn(ROWS, COLS, generator=g) * 0.02
    w[3, 7] = 0.6
    return w.to(device=device, dtype=torch.bfloat16).contiguous()


def _hessian(seed, device):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(4 * COLS, COLS, generator=g)
    mix = torch.eye(COLS) + torch.randn(COLS, COLS, generator=g) / COLS ** 0.5
    x = x @ mix
    x[:, ::29] *= 5.0
    return (x.T @ x).to(device=device, dtype=torch.float32)


def test_a_window_span_yields_every_rate_at_once():
    # The exporter's resolution (``encode_linears_planes``), not the bare
    # recipe: it fills the CHANNEL plane's source spread.
    recipe = _resolve_recipe(BF16_GRID, None, None, None, None, None, None, None)(Q256)
    assert recipe.body is BodyKind.WINDOW
    assert recipe.scale_plane is ScalePlaneKind.CHANNEL
    rates, grid = _plan_for(BF16_GRID, Q256, COLS, recipe.body, recipe.channel_sigma)
    first_block = rates[COLS - BLOCK:]           # LDLQ runs the blocks last to first
    assert sorted(set(first_block)) == [4, 5], "the fixture must mix rates inside a block"
    ldl = block_ldl(regularize_hessian(_hessian(0, "cpu"), sigma_reg=1.0), BLOCK)
    steps = _encode_unit_steps(
        _weights(0, "cpu"), grid, rates, DEFAULT_CODE,
        body=recipe.body, window_bits=recipe.window_bits, window_seed=recipe.window_seed,
        window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
        span=recipe.span, scale_plane=recipe.scale_plane, trellis_weighting="scale",
        scale_refit=1, ldl=ldl, ldl_block=BLOCK,
    )
    first = steps.send(None)
    steps.close()
    assert isinstance(first, tuple), f"one call per yield, got {type(first).__name__}"
    assert all(isinstance(call, _TrellisCall) for call in first)
    assert [call.rate for call in first] == [4, 5]
    assert [call.targets.shape[1] for call in first] == [
        sum(1 for r in first_block if r == 4), sum(1 for r in first_block if r == 5)]


def _encode(weights, per_unit):
    return encode_linears_planes(
        weights, grid=BF16_GRID, q256=Q256, names=[f"u{i}" for i in range(len(weights))],
        per_unit=per_unit, verify=True)


def _state(unit):
    out = {}
    for field in dataclasses.fields(unit):
        value = getattr(unit, field.name)
        out[field.name] = value.cpu() if isinstance(value, torch.Tensor) else value
    return out


@cuda
def test_rate_streams_write_the_serial_bytes(monkeypatch):
    recipe = wire_recipe(BF16_GRID, Q256)
    weights = [_weights(seed, "cuda") for seed in (11, 12, 13)]
    source = ActivationSource(
        hessians={f"u{i}": _hessian(100 + i, "cuda") for i in range(len(weights))},
        provenance={"text_sha256": "0" * 64, "fit_tokens": 4 * COLS,
                    "fit_ids_sha256": "1" * 64},
    )
    per_unit = [source.for_unit(f"u{i}", COLS, "cuda", scale_plane=recipe.scale_plane)
                for i in range(len(weights))]
    monkeypatch.setenv("TESSERA_WINDOW_RATE_STREAMS", "0")
    serial = _encode(weights, per_unit)
    monkeypatch.delenv("TESSERA_WINDOW_RATE_STREAMS")
    grouped = _encode(weights, per_unit)
    # Twice: the second run replays the captured plans on the side streams.
    again = _encode(weights, per_unit)
    for one, two, three in zip(serial, grouped, again):
        assert two[0].blob == one[0].blob
        assert three[0].blob == one[0].blob
        a, b, c = _state(one[1]), _state(two[1]), _state(three[1])
        assert a.keys() == b.keys() == c.keys()
        for key in a:
            if isinstance(a[key], torch.Tensor):
                assert torch.equal(a[key], b[key]), key
                assert torch.equal(a[key], c[key]), key
            elif isinstance(a[key], float):
                assert a[key].hex() == b[key].hex() == c[key].hex(), key
            else:
                assert a[key] == b[key] == c[key], key
    assert serial[0][0].blob != serial[1][0].blob
