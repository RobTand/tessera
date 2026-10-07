"""``regdirect_routed.transcode_stacks`` equals ``layer_stacks`` bit for bit, on the device.

The transcode kernel replaces the per-expert repack at serve load.  On the same bundles (the
loader's tile-order layout, mixed R3/R4 k-steps, one expert with a TP-cut start state) every
plane must be identical, and a Bresenham-mixed unit must be refused the same way.
"""
from __future__ import annotations

import dataclasses

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA device required for the transcode kernel", allow_module_level=True)

from tessera import regdirect_routed as rr  # noqa: E402
from tessera.errors import GrammarError  # noqa: E402
from test_regdirect_layer_build import E, _bundle, _rates, _table  # noqa: E402

FIELDS = ("words_all", "codes_all", "native_all", "scale_all", "runs_all", "init_all", "has_init", "word_off",
          "tile_words", "total_words", "run_off", "perm_all")


def _on(bundle, device):
    return dataclasses.replace(bundle, **{f: getattr(bundle, f).to(device) for f in FIELDS})


def _layer(device, gu=None, dn=None):
    gu = gu or [_rates(256, 32, 10 + e) for e in range(E)]
    dn = dn or [_rates(128, 64, 20 + e) for e in range(E)]
    bundles = tuple(_on(b, device) for b in (_bundle("gate", gu, 1)[0], _bundle("up", gu, 2)[0],
                                              _bundle("down", dn, 3)[0]))
    return bundles, tuple(_table(b) for b in bundles)


def test_transcode_equals_the_reference_repack_plane_for_plane():
    device = torch.device("cuda", torch.cuda.current_device())
    (gate, up, down), tables = _layer(device)
    want = rr.layer_stacks(gate, up, down, tables)
    got = rr.transcode_stacks(gate, up, down, tables)
    for mode in (0, 2):
        for name, value in want[mode].items():
            if torch.is_tensor(value):
                assert torch.equal(got[mode][name].cpu(), value.cpu()), f"mode {mode} {name}"
            else:
                assert got[mode][name] == value, f"mode {mode} {name}"


def test_transcode_refuses_a_bresenham_mixed_unit():
    device = torch.device("cuda", torch.cuda.current_device())
    mixed = tuple(3 if c % 2 else 4 for c in range(256))
    (gate, up, down), tables = _layer(device, gu=[mixed] * E)
    with pytest.raises(GrammarError, match="k-step"):
        rr.transcode_stacks(gate, up, down, tables)
