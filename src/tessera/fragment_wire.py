"""CPU fragment-order BODY repack for routed T-8 units.

The disk BODY remains a per-column MSB-first stream. This module does not
select a kernel, change a rate schedule, or apply row scales.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from .errors import GrammarError
from .wire import unpack_body

__all__ = ["FragmentWire", "repack_codes", "repack_fragment", "decode_fragment"]

_TILE_ROWS = 128
_KSTEP = 32
_WINDOW = 14


@dataclass(frozen=True)
class FragmentWire:
    """One expert and one projection group in a shared word address space.

    ``words`` contains this expert's words only, as native int32 bit patterns.
    ``expert_offsets`` contains its absolute start and end word offsets.
    ``history_offsets[s]`` and ``unit_offsets[T, s, w]`` use that same base.
    Gate/up uses ``perm[s]`` for its original 32-column group. Down uses
    ``perm[s * 2 + p]`` for group p of unit k-step s.
    ``rates[s]`` gives the shared rate of that unit k-step.

    A history unit stores only lanes 24..31, at word ``i * 8 + L - 24``.
    A data unit stores all lanes, at word ``i * 32 + L``.
    """

    words: torch.Tensor
    perm: torch.Tensor
    rates: torch.Tensor
    history_offsets: torch.Tensor
    unit_offsets: torch.Tensor
    expert_offsets: torch.Tensor
    rows: int
    cols: int
    projection_group: Literal["gate_up", "down"]


def _unpack(words: np.ndarray, rate: int, lanes: int) -> np.ndarray:
    lane_words = words.reshape(rate, lanes).T.astype(">u4")
    bits = np.unpackbits(np.frombuffer(lane_words.tobytes(), dtype=np.uint8), bitorder="big")
    bits = bits.reshape(lanes, 16, 2, rate)
    weights = 1 << np.arange(rate - 1, -1, -1, dtype=np.uint16)
    fields = (bits * weights).sum(axis=3, dtype=np.uint16)
    return fields.reshape(lanes // 4, 4, 2, 8, 2).transpose(2, 0, 4, 1, 3).reshape(2, lanes // 2, 32)


def _start_state(state: torch.Tensor | None, projections: int, cols: int, device) -> torch.Tensor:
    if state is None:
        return torch.zeros((projections, cols), dtype=torch.int64, device=device)
    if state.dtype not in (torch.int16, torch.int32, torch.int64):
        raise GrammarError("start state needs an integer tensor")
    if state.shape != (projections, cols) and not (projections == 1 and state.shape == (cols,)):
        raise GrammarError(f"start state needs shape [{projections}, {cols}]")
    values = state.reshape(projections, cols).to(device=device, dtype=torch.int64)
    if bool((values < 0).any()) or bool((values >= 1 << _WINDOW).any()):
        raise GrammarError("start state does not fit the 14-bit window")
    return values


def _step_rates(rates, cols: int, groups: int) -> list:
    if len(rates) != cols:
        raise GrammarError(f"{len(rates)} rates for {cols} fragment columns")
    step_rates = []
    for start in range(0, cols, _KSTEP):
        rate = rates[start]
        if rate not in (3, 4):
            raise GrammarError(f"rate {rate} is not R3 or R4 at k-step {start // _KSTEP}")
        if any(r != rate for r in rates[start:start + _KSTEP]):
            raise GrammarError(f"rate changes inside k-step {start // _KSTEP}")
        step_rates.append(rate)
    return step_rates


def _lane_words(fields: torch.Tensor, rate: int) -> torch.Tensor:
    """int64 fields ``[..., lanes, 32]`` in pair order ``(p, j, row)`` -> int32 words ``[..., R, lanes]``:
    pair ``q = p * 8 + j`` holds rows ``2g`` then ``2g + 1`` at bit ``q * 2R`` of the MSB-first lane stream."""
    shifts = torch.arange(rate - 1, -1, -1, device=fields.device)
    bits = ((fields.unsqueeze(-1) >> shifts) & 1).flatten(-2)                 # [..., lane, 32R]
    bits = bits.reshape(*bits.shape[:-1], rate, 32)
    words = (bits << torch.arange(31, -1, -1, device=fields.device)).sum(-1)  # [..., lane, R]
    words = torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)
    return words.movedim(-1, -2)


def repack_codes(
    codes: torch.Tensor,
    rates: tuple[int, ...],
    *,
    projection_group: Literal["gate_up", "down"],
    start_state: torch.Tensor | None = None,
    word_offset: int = 0,
) -> FragmentWire:
    """The fragment repack of ``codes`` ``[projections, rows, cols]`` (BODY codes, original column
    order), on the codes' device.  :func:`repack_fragment` is this after its BODY unpack."""
    if projection_group not in ("gate_up", "down"):
        raise GrammarError(f"projection_group {projection_group!r} is not gate_up or down")
    groups = 1 if projection_group == "gate_up" else 2
    projections = 2 if projection_group == "gate_up" else 1
    if codes.dim() != 3 or codes.shape[0] != projections:
        raise GrammarError(f"{projection_group} needs {projections} BODY planes")
    _, rows, cols = codes.shape
    if rows <= 0 or cols <= 0 or cols % (_KSTEP * groups):
        raise GrammarError(f"fragment rows must be positive and cols must be a positive multiple of {32 * groups}")
    if cols // _KSTEP > torch.iinfo(torch.int16).max + 1:
        raise GrammarError("k-step permutation does not fit int16")
    if word_offset < 0:
        raise GrammarError("word_offset must be nonnegative")
    step_rates = _step_rates(rates, cols, groups)
    device = codes.device
    state = _start_state(start_state, projections, cols, device)
    permutation = sorted(range(len(step_rates)), key=step_rates.__getitem__)
    originals = torch.tensor(permutation, dtype=torch.int64).reshape(-1, groups)
    sorted_rates = [step_rates[int(pair[0])] for pair in originals]
    if any(step_rates[int(original)] != rate
           for pair, rate in zip(originals, sorted_rates) for original in pair):
        raise GrammarError("down needs an even 32-column group count at each rate")
    n_tiles = (rows + _TILE_ROWS - 1) // _TILE_ROWS
    rows_p = n_tiles * _TILE_ROWS
    padded = torch.zeros((projections, rows_p, cols), dtype=torch.int64, device=device)
    padded[:, :rows] = codes.to(torch.int64)
    lane = torch.arange(_KSTEP, device=device)
    history, tiles, history_offsets, unit_offsets = [], [], [], []
    hcursor = 0
    tcursor = 0
    slot = 0
    for rate in sorted(set(sorted_rates)):
        n = sorted_rates.count(rate)
        seg = originals[slot:slot + n].to(device)                              # [n, groups]
        # fields[slot, p, row, column]: gate/up p = projection; down p = group of the pair
        if projections == 2:
            cols_idx = (seg[:, 0:1] * _KSTEP + lane).reshape(-1)
            f = padded[:, :, cols_idx].reshape(2, rows_p, n, _KSTEP).permute(2, 0, 1, 3)
            st = state[:, cols_idx].reshape(2, n, _KSTEP).permute(1, 0, 2)
        else:
            cols_idx = (seg.unsqueeze(-1) * _KSTEP + lane).reshape(-1)
            f = padded[0][:, cols_idx].reshape(rows_p, n, 2, _KSTEP).permute(1, 2, 0, 3)
            st = state[0, cols_idx].reshape(n, 2, _KSTEP)
        # [slot, p, T, warp, g, row, t, j] -> [T, slot, warp, lane (g, t), field (p, j, row)]
        f = f.reshape(n, 2, n_tiles, 8, 8, 2, 4, 8).permute(2, 0, 3, 4, 6, 1, 7, 5).reshape(n_tiles, n, 8, 32, 32)
        tiles.append(_lane_words(f, rate).reshape(n_tiles, -1))
        # history rows -4..-1 = the last four fields of the incoming state, lanes (6, t), (7, t)
        tail = torch.stack([(st >> (k * rate)) & ((1 << rate) - 1) for k in (3, 2, 1, 0)], 2)   # [n, p, 4, 32]
        tail = tail.reshape(n, 2, 2, 2, 4, 8).permute(0, 2, 4, 1, 5, 3).reshape(n, 8, 32)
        history.append(_lane_words(tail, rate).reshape(-1))
        for s in range(n):
            history_offsets.append(hcursor + s * 8 * rate)
        unit_offsets.append((tcursor, n, rate))
        hcursor += n * 8 * rate
        tcursor += n * 8 * 32 * rate
        slot += n
    history_words, tile_words = hcursor, tcursor
    words = torch.cat([torch.cat(history), torch.cat(tiles, 1).reshape(-1)])
    base = word_offset + history_words
    units = torch.empty((n_tiles, len(sorted_rates), 8), dtype=torch.int64)
    slot = 0
    for start, n, rate in unit_offsets:
        within = start + (torch.arange(n) * 8 * 32 * rate)[:, None] + torch.arange(8)[None, :] * 32 * rate
        units[:, slot:slot + n] = base + torch.arange(n_tiles)[:, None, None] * tile_words + within
        slot += n
    return FragmentWire(
        words=words, perm=torch.tensor(permutation, dtype=torch.int16),
        rates=torch.tensor(sorted_rates, dtype=torch.int32),
        history_offsets=torch.tensor(history_offsets, dtype=torch.int64) + word_offset, unit_offsets=units,
        expert_offsets=torch.tensor([word_offset, word_offset + words.numel()], dtype=torch.int64),
        rows=rows, cols=cols, projection_group=projection_group,
    )


def repack_fragment(
    body_planes: tuple[bytes, ...] | bytes,
    rates: tuple[int, ...],
    *,
    rows: int,
    cols: int,
    projection_group: Literal["gate_up", "down"],
    initial_state: torch.Tensor | None = None,
    start_state: torch.Tensor | None = None,
    word_offset: int = 0,
) -> FragmentWire:
    """Repack disk BODY planes for one expert and one projection group.

    ``rows`` is the row count of each projection. Gate and up use two planes,
    in that order, with the same per-column ``rates``. Down uses one plane.
    Each 32-column group must have one rate, either R3 or R4.
    The permutation sorts groups by rate and keeps their order within a rate.
    Down pairs successive sorted groups of the same rate. Each rate must
    therefore have an even group count. Down has cols / 64 unit k-steps.

    ``initial_state`` gives each column's state before local row zero.
    ``start_state`` takes precedence when the caller supplies a TP-cut state.
    Both use original column order and shape [projections, cols].
    The caller must cut the BODY planes before this call.

    Each 128-row tile has eight units per k-step. A down unit holds the same
    16 rows of two 32-column groups. Gate/up holds the same 16 rows of two
    projections. Final tile padding contains zero codes. History precedes tile 0.
    ``word_offset`` is the expert's absolute base in a shared word buffer.
    """
    if projection_group not in ("gate_up", "down"):
        raise GrammarError(f"projection_group {projection_group!r} is not gate_up or down")
    groups = 1 if projection_group == "gate_up" else 2
    if rows <= 0 or cols <= 0 or cols % (_KSTEP * groups):
        raise GrammarError(f"fragment rows must be positive and cols must be a positive multiple of {32 * groups}")
    _step_rates(rates, cols, groups)
    projections = 2 if projection_group == "gate_up" else 1
    planes = (body_planes,) if isinstance(body_planes, bytes) else body_planes
    if len(planes) != projections:
        raise GrammarError(f"{projection_group} needs {projections} BODY planes")
    codes = torch.stack([unpack_body(plane, rates, rows, device="cpu") for plane in planes])
    return repack_codes(codes, rates, projection_group=projection_group,
                        start_state=start_state if start_state is not None else initial_state,
                        word_offset=word_offset)


def decode_fragment(fragment: FragmentWire, tables: torch.Tensor) -> torch.Tensor:
    """Decode fragment words to E4M3 bytes in original column order.

    ``tables`` holds native E4M3 bytes indexed by the 14-bit window state.
    Its shape is [2, 16384] for gate/up or [1, 16384] for down.
    The output shape is [projections * rows, cols], with gate before up.
    The decode reads its start state from history words, not a separate tensor.
    Row scales belong to the kernel epilogue and do not change these bytes.
    """
    projections = 2 if fragment.projection_group == "gate_up" else 1
    if tables.device.type != "cpu" or tables.dtype != torch.uint8:
        raise GrammarError("fragment tables need CPU uint8 E4M3 bytes")
    if tables.shape != (projections, 1 << _WINDOW):
        raise GrammarError(f"fragment tables need shape [{projections}, {1 << _WINDOW}]")
    words = fragment.words.numpy()
    base = int(fragment.expert_offsets[0])
    output = np.empty((projections, fragment.rows, fragment.cols), dtype=np.uint8)
    table = tables.numpy()
    groups = 1 if projections == 2 else 2
    originals = fragment.perm.reshape(-1, groups).tolist()
    for slot, pair in enumerate(originals):
        rate = int(fragment.rates[slot])
        offset = int(fragment.history_offsets[slot]) - base
        history = _unpack(words[offset:offset + 8 * rate], rate, 8)
        state = np.zeros((2, 32), dtype=np.int64)
        for row in range(4):
            state = (state << rate) | history[:, row]
        for tile in range(fragment.unit_offsets.shape[0]):
            for warp in range(fragment.unit_offsets.shape[2]):
                offset = int(fragment.unit_offsets[tile, slot, warp]) - base
                fields = _unpack(words[offset:offset + 32 * rate], rate, 32)
                first_row = tile * 128 + warp * 16
                for row in range(min(16, fragment.rows - first_row)):
                    state = ((state << rate) | fields[:, row]) & ((1 << _WINDOW) - 1)
                    for p in range(2):
                        projection = p if projections == 2 else 0
                        original = pair[0 if projections == 2 else p]
                        columns = slice(original * 32, (original + 1) * 32)
                        output[projection, first_row + row, columns] = table[projection, state[p]]
    return torch.from_numpy(output.reshape(projections * fragment.rows, fragment.cols))
