"""An expert axis filled one prepared owner at a time is ``stack``, byte for byte.

The routed-expert load prepares each projection inside its own load callback
and places it on its group's expert axis at once (tessera#501), instead of
holding every per-expert owner until ``finish`` stacks them.  Pinned here,
with no device:

- a window axis filled in any order holds exactly what the ``torch.stack``
  reference below builds -- every tensor equal, with the same dtype, shape,
  stride and device -- for byte-code, BF16-value and float-value tables: the
  E4M3, BF16 and E2M1 alphabets ride the one family-agnostic axis;
- the axis keeps no reference to a window it has placed;
- a module axis joining separately placed parts is ``concatenate`` then
  ``stack``, for FP8 and BF16 modules;
- the refusals: layout, a double placement, a missing expert, reuse.
"""
from __future__ import annotations

import gc
import weakref

import pytest

torch = pytest.importorskip("torch")

from tessera.serving import bf16_route, fp8_route  # noqa: E402
from tessera.serving.window import PreparedWindow, PreparedWindowAxis, prepare_window  # noqa: E402


def _stack_reference(windows, *, with_state):
    """``PreparedWindow.stack`` as it stood at 00d517d40, read through ``tensors()``.

    A window lists (plane, gather, shift, which) per rate group, the table,
    then the inverse and the initial state when each is present; the batch
    lists the same without the initial state."""
    per = [w.tensors() for w in windows]
    first, groups = per[0], len(windows[0].rates)
    out = []
    for i in range(groups):
        out.append(torch.stack([t[4 * i] for t in per]))
        out.extend(first[4 * i + k].clone() for k in (1, 2, 3))
    out.append(torch.stack([t[4 * groups] for t in per]))
    if len(first) - 4 * groups - 1 - int(with_state):
        out.append(first[4 * groups + 1].clone())
    return out


def _assert_identical(got, want):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert (g.dtype, tuple(g.shape), g.stride(), g.device) == (
            w.dtype, tuple(w.shape), w.stride(), w.device)
        assert torch.equal(g, w)


def _table(kind, generator, window_bits):
    codes = torch.randint(0, 256, (1 << window_bits,), generator=generator).to(torch.uint8)
    if kind == "bf16_values":
        return (codes.float() / 32 - 4).bfloat16()
    if kind == "float_values":
        return codes.float() / 64 - 2
    return codes


@pytest.mark.parametrize("table_kind", ["codes", "bf16_values", "float_values"])
@pytest.mark.parametrize("rates", [[2] * 8, [1, 3, 2, 4, 1, 4, 3, 2]])
def test_a_window_axis_filled_in_any_order_is_the_stack(table_kind, rates):
    windows = []
    for expert in range(5):
        g = torch.Generator().manual_seed(811 + expert)
        body = torch.stack([torch.randint(0, 1 << r, (19,), generator=g) for r in rates],
                           1).to(torch.uint8)
        initial = torch.arange(len(rates), dtype=torch.int32) + expert * 7
        windows.append(prepare_window(body, rates, 9, _table(table_kind, g, 9), 'cpu',
                                      initial_state=initial))
    want = _stack_reference(windows, with_state=True)
    axis = PreparedWindowAxis(len(windows))
    for expert in (3, 0, 4, 1, 2):
        axis.put(expert, windows[expert])
    assert axis.placed() == 5
    batch = axis.finish()
    _assert_identical(batch.tensors(), want)
    _assert_identical(PreparedWindow.stack(windows).tensors(), want)
    ids = torch.tensor([4, 1, 3, 0, 2])
    assert torch.equal(batch.decode(ids, max_experts_per_chunk=2),
                       torch.stack([windows[int(i)].decode() for i in ids]))


def test_a_window_axis_keeps_no_reference_to_a_placed_window():
    g = torch.Generator().manual_seed(812)
    window = prepare_window(torch.randint(0, 4, (16, 4), generator=g, dtype=torch.uint8),
                            [2] * 4, 8, _table("codes", g, 8), 'cpu')
    expected = window.decode()
    plane = weakref.ref(window.tensors()[0])
    axis = PreparedWindowAxis(1)
    axis.put(0, window)
    del window
    gc.collect()
    assert plane() is None
    assert torch.equal(axis.finish().decode(torch.tensor([0]), max_experts_per_chunk=1)[0],
                       expected)


def test_a_window_axis_refuses_what_stack_refuses_and_its_misuse():
    g = torch.Generator().manual_seed(813)
    table = _table("codes", g, 8)
    body = torch.randint(0, 2, (16, 4), generator=g, dtype=torch.uint8)
    first = prepare_window(body, [2] * 4, 8, table, 'cpu')
    other_rates = prepare_window(body, [1, 3, 1, 3], 8, table, 'cpu')
    swapped = prepare_window(body, [3, 1, 3, 1], 8, table, 'cpu')   # same key, other gather
    axis = PreparedWindowAxis(3)
    with pytest.raises(ValueError, match='at least one'):
        axis.finish()
    axis.put(1, first)
    with pytest.raises(ValueError, match='layout'):
        axis.put(0, other_rates)
    with pytest.raises(ValueError, match='already placed'):
        axis.put(1, first)
    with pytest.raises(ValueError, match='not on this'):
        axis.put(3, first)
    with pytest.raises(ValueError, match='never placed'):
        axis.finish()
    axis.put(0, first)
    axis.put(2, first)
    axis.finish()
    with pytest.raises(RuntimeError, match='finished'):
        axis.put(0, first)
    mixed = PreparedWindowAxis(2)
    mixed.put(0, other_rates)
    with pytest.raises(ValueError, match='layout'):
        mixed.put(1, swapped)


FAMILIES = {
    "fp8": (fp8_route.PreparedTesseraFp8Module, fp8_route._Fp8Role, "codes"),
    "bf16": (bf16_route.PreparedTesseraBf16Module, bf16_route._Bf16Role, "bf16_values"),
}


def _module(family, expert, names, seed, rates=(2,) * 8):
    module_type, role_type, table_kind = FAMILIES[family]
    g = torch.Generator().manual_seed(seed + 31 * expert)
    roles = []
    for position, name in enumerate(names):
        body = torch.stack([torch.randint(0, 1 << r, (16,), generator=g) for r in rates],
                           1).to(torch.uint8)
        window = prepare_window(body, list(rates), 8, _table(table_kind, g, 8), 'cpu')
        roles.append(role_type(name, 16 * position, 16, window))
    return module_type(roles, rows=16 * len(names), columns=len(rates),
                       scale=torch.rand(16 * len(names), generator=g), device=torch.device('cpu'))


def _roles(module):
    return getattr(module, f"_{type(module).__name__}__roles")


def _batch_tensors(batch):
    name = type(batch).__name__
    windows = getattr(batch, f"_{name}__windows")
    return [t for w in windows for t in w.tensors()] + [getattr(batch, f"_{name}__scales")]


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_module_axis_joining_parts_in_any_order_is_concatenate_then_stack(family):
    module_type = FAMILIES[family][0]
    gate = [_module(family, e, ('gate',), 900) for e in range(4)]
    up = [_module(family, e, ('up',), 950) for e in range(4)]
    want = (_stack_reference([_roles(m)[0].window for m in gate], with_state=False)
            + _stack_reference([_roles(m)[0].window for m in up], with_state=False)
            + [torch.stack([torch.cat([g.row_scale(), u.row_scale()]) for g, u in zip(gate, up)])])
    axis = module_type.axis(4, parts=2)
    for expert in reversed(range(4)):
        axis.put(expert, up[expert], part=1)
    for expert in (2, 0, 3, 1):
        axis.put(expert, gate[expert], part=0)
    assert axis.placed() == 8 and axis.resident_bytes() > 0
    got = axis.finish()
    reference = module_type.stack([module_type.concatenate([g, u]) for g, u in zip(gate, up)])
    assert type(got) is type(reference)
    assert (got.role_names, got.rows, got.columns, got.device, got.experts) == (
        ('gate', 'up'), 32, 8, torch.device('cpu'), 4)
    assert (reference.role_names, reference.rows, reference.columns) == (('gate', 'up'), 32, 8)
    _assert_identical(_batch_tensors(got), want)
    _assert_identical(_batch_tensors(reference), want)


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_a_module_axis_refuses_what_concatenate_and_stack_refuse(family):
    module_type = FAMILIES[family][0]
    label = family.upper()
    gate, up = _module(family, 0, ('gate',), 900), _module(family, 0, ('up',), 950)
    axis = module_type.axis(2, parts=2)
    axis.put(0, gate, part=0)
    with pytest.raises(ValueError, match=f'stacked {label} modules must share roles'):
        axis.put(1, up, part=0)
    with pytest.raises(ValueError, match='already placed'):
        axis.put(0, gate, part=0)
    with pytest.raises(ValueError, match='part 2'):
        axis.put(0, gate, part=2)
    axis.put(1, gate, part=0)
    with pytest.raises(ValueError, match='never arrived'):
        axis.finish()
    twice = module_type.axis(1, parts=2)
    twice.put(0, gate, part=0)
    twice.put(0, gate, part=1)
    with pytest.raises(ValueError, match='distinct names'):
        twice.finish()
    with pytest.raises(ValueError, match='at least one'):
        module_type.axis(3).finish()
    # A module whose second role's window disagrees fails half placed; the
    # axis refuses everything after rather than finish a torn slot.
    pair = _module(family, 0, ('gate', 'up'), 900)
    torn = _module(family, 1, ('gate', 'up'), 900)
    other = _roles(_module(family, 1, ('up',), 950, rates=(1, 3) * 4))[0].window
    _roles(torn)[1].window = other
    stacked = module_type.axis(2)
    stacked.put(0, pair)
    with pytest.raises(ValueError, match='layout'):
        stacked.put(1, torn)
    with pytest.raises(RuntimeError, match='refused'):
        stacked.put(1, pair)
