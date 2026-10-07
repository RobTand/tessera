"""The CPU fragment wire preserves routed E4M3 code bits and stream history."""
from __future__ import annotations

import json

import pytest
import torch

from tessera.decode import replay_window
from tessera.errors import GrammarError
from tessera.wire import pack_body


RATE_STEPS = ([(rate, rate) for rate in range(3, 9)]
              + [(ra + 1, ra, ra + 1, ra) for ra in range(3, 8)])


def _api():
    from tessera.fragment_wire import decode_fragment, repack_fragment

    return repack_fragment, decode_fragment


def _case(group, steps, rows=259, seed=71):
    rates = tuple(rate for rate in steps for _ in range(32))
    generator = torch.Generator().manual_seed(seed)
    count = 2 if group == "gate_up" else 1
    bodies = [torch.stack([torch.randint(1 << r, (rows,), generator=generator)
                           for r in rates], 1) for _ in range(count)]
    tables = torch.randint(256, (count, 1 << 14), generator=generator, dtype=torch.int64)
    tables[(tables & 127) == 127] = 126
    return bodies, rates, tables.to(torch.uint8)


def _reference(bodies, rates, tables, states=None):
    outputs = []
    for p, body in enumerate(bodies):
        out = torch.empty(body.shape, dtype=torch.uint8)
        for rate in sorted(set(rates)):
            columns = torch.tensor([j for j, r in enumerate(rates) if r == rate])
            initial = None if states is None else states[p, columns]
            state = replay_window(body[:, columns], 14, rate, initial)
            out[:, columns] = tables[p][state]
        outputs.append(out)
    return torch.cat(outputs)


@pytest.mark.parametrize("group", ["gate_up", "down"])
@pytest.mark.parametrize("steps", RATE_STEPS)
def test_fragment_bytes_equal_window_replay(group, steps):
    repack, decode = _api()
    bodies, rates, tables = _case(group, steps)
    packed = repack(tuple(pack_body(b, rates) for b in bodies), rates,
                    rows=bodies[0].shape[0], cols=len(rates), projection_group=group,
                    word_offset=3072)
    assert torch.equal(decode(packed, tables), _reference(bodies, rates, tables))
    expected_perm = sorted(range(len(steps)), key=steps.__getitem__)
    assert packed.perm.dtype == torch.int16
    assert packed.perm.tolist() == expected_perm
    assert packed.expert_offsets.tolist() == [3072, 3072 + packed.words.numel()]
    assert packed.history_offsets[0] == 3072
    groups = 1 if group == "gate_up" else 2
    assert packed.unit_offsets.shape == (3, len(steps) // groups, 8)
    assert packed.unit_offsets[0, 0, 0] == 3072 + sum(8 * r for r in steps) // groups


def _field(words, lane, bit, width, stride):
    word, offset = divmod(bit, 32)
    value = int(words[word * stride + lane]) & 0xFFFFFFFF
    if offset + width <= 32:
        return (value >> (32 - offset - width)) & ((1 << width) - 1)
    value = (value << 32) | (int(words[(word + 1) * stride + lane]) & 0xFFFFFFFF)
    return (value >> (64 - offset - width)) & ((1 << width) - 1)


@pytest.mark.parametrize("group", ["gate_up", "down"])
@pytest.mark.parametrize("rate", range(3, 9))
def test_fragment_pairs_follow_the_mma_lane_map(group, rate):
    repack, _decode = _api()
    steps = (rate,) if group == "gate_up" else (rate, rate)
    bodies, rates, _tables = _case(group, steps, rows=256)
    packed = repack(tuple(pack_body(b, rates) for b in bodies), rates,
                    rows=256, cols=len(rates), projection_group=group)
    warps = 8
    for tile in range(2):
        for warp in range(warps):
            words = packed.words[int(packed.unit_offsets[tile, 0, warp]):]
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for slot in range(2):
                    p = slot if group == "gate_up" else 0
                    row = 128 * tile + 16 * warp + 2 * g
                    for j in range(8):
                        column = (32 * slot if group == "down" else 0) + 8 * t + j
                        bit = (slot * 8 + j) * 2 * rate
                        pair = _field(words, lane, bit, 2 * rate, 32)
                        assert pair == (int(bodies[p][row, column]) << rate) | int(bodies[p][row + 1, column])


@pytest.mark.parametrize("group", ["gate_up", "down"])
@pytest.mark.parametrize("cut", [1, 129])
@pytest.mark.parametrize("steps", RATE_STEPS)
def test_tp_cut_history_preserves_nonzero_parent_state(group, cut, steps):
    repack, decode = _api()
    bodies, rates, tables = _case(group, steps, rows=cut + 131)
    initial = torch.full((len(bodies), len(rates)), (1 << 14) - 1, dtype=torch.int32)
    states = initial.clone()
    for p, body in enumerate(bodies):
        for rate in sorted(set(rates)):
            columns = torch.tensor([j for j, r in enumerate(rates) if r == rate])
            states[p, columns] = replay_window(body[:cut, columns], 14, rate, initial[p, columns])[-1].int()
    local = [body[cut:] for body in bodies]
    packed = repack(tuple(pack_body(b, rates) for b in local), rates,
                    rows=131, cols=len(rates), projection_group=group,
                    initial_state=initial, start_state=states)
    expected = _reference(bodies, rates, tables, initial).reshape(len(bodies), cut + 131, -1)[:, cut:].reshape(-1, len(rates))
    assert torch.equal(decode(packed, tables), expected)
    assert not hasattr(packed, "initial_state")
    groups = 1 if group == "gate_up" else 2
    originals = packed.perm.reshape(-1, groups).tolist()
    for s, columns in enumerate(originals):
        rate = rates[columns[0] * 32]
        words = packed.words[int(packed.history_offsets[s]):]
        for lane in range(8):
            g, t = lane >> 2, lane & 3
            for slot in range(2):
                p = slot if group == "gate_up" else 0
                original = columns[0 if group == "gate_up" else slot]
                for j in range(8):
                    pair = _field(words, lane, (slot * 8 + j) * 2 * rate, 2 * rate, 8)
                    state = int(states[p, original * 32 + 8 * t + j])
                    assert pair == (state >> ((1 - g) * 2 * rate)) & ((1 << (2 * rate)) - 1)


@pytest.mark.parametrize("bad_rates", [(3, 4) * 32, (2,) * 64, (3,) * 32 + (4,) * 32])
def test_fragment_refuses_unservable_rate_geometry(bad_rates):
    repack, _decode = _api()
    with pytest.raises(GrammarError, match="rate"):
        repack((pack_body(torch.zeros((128, 64), dtype=torch.int64), bad_rates),),
               bad_rates, rows=128, cols=64, projection_group="down")


@pytest.mark.parametrize("group", ["gate_up", "down"])
def test_real_glm53_export_matches_window_replay(group):
    repack, decode = _api()
    import box_artifacts
    from safetensors import safe_open
    from tessera.fused import parse_fused
    from tessera.planes import PlaneKind
    from tessera.unit_artifact import parse_unit_metadata
    from tessera.wire import unpack_body

    relative = "moe/glm53-a8-bf16menu-20260930/hf-staging-a8s"
    index = box_artifacts.skip_now("shared_runs", relative, "model.safetensors.index.json")
    mapping = json.loads(index.read_text())["weight_map"]
    projections = ("gate_proj", "up_proj") if group == "gate_up" else ("down_proj",)
    bodies, planes, tables, states = [], [], [], []
    for projection in projections:
        key = f"model.language_model.layers.3.mlp.experts.0.{projection}.wire"
        shard = box_artifacts.skip_now("shared_runs", relative, mapping[key])
        with safe_open(str(shard), framework="pt") as handle:
            blob = bytes(handle.get_tensor(key).tolist())
        member, = parse_fused(blob)
        metadata = parse_unit_metadata(member.blob)
        assert metadata.grid.name == "E4M3" and metadata.manifest.window_bits == 14
        rates = tuple(metadata.rates)
        body = unpack_body(metadata.chunks[PlaneKind.BODY], rates, metadata.rows)
        # The first 259 rows include two tile boundaries and a partial tile.
        bodies.append(body[:259])
        planes.append(pack_body(body[:259], rates))
        native = torch.tensor(metadata.grid.native, dtype=torch.uint8)
        from tessera.unit_artifact import _window_table

        codes = _window_table(metadata.chunks, metadata.grid, 14)
        tables.append(native[codes.long()])
        states.append(torch.zeros(metadata.columns, dtype=torch.int32) if metadata.shard_state is None else metadata.shard_state.int())
    tables, states = torch.stack(tables), torch.stack(states)
    packed = repack(tuple(planes), rates, rows=259, cols=len(rates),
                    projection_group=group, initial_state=states)
    assert torch.equal(decode(packed, tables), _reference(bodies, rates, tables, states))
