"""Class-local storage and device-prefix controls; CUDA parity lives separately."""
import dataclasses
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tessera import routed_fused as rf
from tessera.errors import GrammarError
from tessera.native_window_moe import PackedWindowMoeBundles
from tessera.window_gemm_grouped import PreparedGroupedWindowGemm


def descriptors(extents=((0, 2, 768), (2, 4, 1024))):
    return [{"start": start, "end": end, "q256": {"w13": [q, q], "w2": [q]}}
            for start, end, q in extents]


def projection(rates=(3, 3, 4, 4), *, family="value", cols=128, rows=128):
    widths = [16 * rate * cols * ((rows + 511) // 512) for rate in rates]
    offsets = [0]
    for width in widths:
        offsets.append(offsets[-1] + width)
    e = len(rates)
    return PreparedGroupedWindowGemm(
        words_all=torch.arange(offsets[-1], dtype=torch.int32),
        table_all=(torch.zeros(e, rf.TABLE_ENTRIES, dtype=torch.bfloat16)
                   if family == "value" else torch.empty(0, dtype=torch.bfloat16)),
        codes_all=(torch.empty(0, dtype=torch.uint8) if family == "value" else
                   torch.arange(rf.TABLE_ENTRIES, dtype=torch.int32).to(torch.uint8).expand(e, -1).clone()),
        native_all=(torch.empty(0, dtype=torch.uint8) if family == "value" else
                    torch.arange(256, dtype=torch.int32).to(torch.uint8).expand(e, -1).clone()),
        scale_all=torch.ones(e, rows),
        runs_all=torch.tensor([[r, 0, cols, 0] for r in rates], dtype=torch.int32),
        init_all=torch.zeros(e, cols, dtype=torch.int32), has_init=torch.zeros(e, dtype=torch.int32),
        word_off=torch.tensor(offsets[:-1], dtype=torch.int32),
        tile_words=torch.tensor([16 * r * cols for r in rates], dtype=torch.int32),
        total_words=torch.tensor(widths, dtype=torch.int32), run_off=torch.arange(e + 1, dtype=torch.int64),
        perm_all=torch.arange(cols, dtype=torch.int32).expand(e, -1).clone(),
        rows=rows, cols=cols, experts=e, window_bits=14, family=family,
        block_m=32, block_n=64, block_k=64, arithmetic="folded" if family == "value" else "epilogue")


def allow_cpu(monkeypatch):
    monkeypatch.setattr(rf, "fused_routed_window_supported", lambda *args: None)
    monkeypatch.setattr(rf, "_ext", lambda library: object())
    monkeypatch.setattr(rf, "_make_dispatch_resources", lambda device, kernel: rf._DispatchResources(
        (object(), object()), object(), (object(), object()),
        torch.empty(0, dtype=torch.float32, device=device), kernel))


def test_mixed_flat_storage_constructs_without_padding(monkeypatch):
    allow_cpu(monkeypatch)
    roles = [projection() for _ in range(3)]
    # The old whole-stack lane cannot view this exact, unequal-stride storage.
    with pytest.raises(GrammarError, match="equal parts|uniform"):
        rf.words_by_expert(roles[0])
    adapter = rf.FusedRoutedWindowMoE.from_bundles(*roles, expert_classes=descriptors())
    assert len(adapter.classes) == 2
    for index, cls in enumerate(adapter.classes):
        for role, original in zip((cls.gate, cls.up, cls.down), roles):
            assert role.experts == 2
            assert role.words_all.untyped_storage().data_ptr() == original.words_all.untyped_storage().data_ptr()
            assert role.words_all.numel() == sum(original.total_words[2*index:2*index+2].tolist())
            assert role.init_all.untyped_storage().data_ptr() == original.init_all.untyped_storage().data_ptr()
            assert role.scale_all.untyped_storage().data_ptr() == original.scale_all.untyped_storage().data_ptr()


def test_class_views_bound_every_plane_and_normalize_offsets():
    original = projection()
    view = rf.grouped_class_view(original, 2, 4)
    assert view.experts == 2
    assert view.words_all.shape == (2, 16 * 4 * 128)
    assert torch.equal(view.words_all.reshape(-1), original.words_all[original.word_off[2]:])
    assert view.word_off.tolist() == [0, 16 * 4 * 128]
    assert view.run_off.tolist() == [0, 1, 2]
    for field in ("words_all", "table_all", "runs_all", "scale_all", "init_all", "has_init", "perm_all"):
        a, b = getattr(original, field), getattr(view, field)
        assert a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr(), field
    assert torch.equal(view.runs_all, original.runs_all[2:4])


@pytest.mark.parametrize("start,end", [(-1, 2), (0, 5), (2, 2), (3, 1), (True, 2)])
def test_class_view_refuses_outside_or_empty_extent(start, end):
    with pytest.raises(GrammarError, match="class extent"):
        rf.grouped_class_view(projection(), start, end)


def test_class_view_refuses_unequal_or_overlapping_word_extents():
    with pytest.raises(GrammarError, match="stride|word"):
        rf.grouped_class_view(projection(), 0, 4)
    role = projection()
    bad = role.word_off.clone()
    bad[2] -= 1
    with pytest.raises(GrammarError, match="word"):
        rf.grouped_class_view(dataclasses.replace(role, word_off=bad), 2, 4)


def test_required_class_metadata_has_no_identity_fallback():
    with pytest.raises(TypeError, match="expert_classes"):
        PackedWindowMoeBundles(*[projection() for _ in range(3)], family="value")


def test_declared_profile_must_match_stored_schedule(monkeypatch):
    allow_cpu(monkeypatch)
    bad = descriptors(((0, 2, 512), (2, 4, 1024)))
    with pytest.raises(GrammarError, match="q256|schedule"):
        rf.FusedRoutedWindowMoE.from_bundles(*[projection() for _ in range(3)], expert_classes=bad)


def test_counter_start_reads_current_device_prefix_without_rebasing_routes():
    counter = torch.empty(1, dtype=torch.int32)
    prefix = torch.tensor([0, 2, 3, 5, 5], dtype=torch.int32)
    from tessera.routed_class_dispatch import initialize_class_counter
    initialize_class_counter(counter, prefix, 2, 9)
    assert counter.tolist() == [27]
    prefix[2] = 7
    initialize_class_counter(counter, prefix, 2, 9)
    assert counter.tolist() == [63]


def test_metadata_and_native_aliases_are_charged_once(monkeypatch):
    allow_cpu(monkeypatch)
    packed = PackedWindowMoeBundles(*[projection() for _ in range(3)], family="value",
                                   expert_classes=descriptors())
    adapter = packed.adapter()
    owner = packed.native_owner()
    assert owner.adapter() is adapter
    # Count the native planes directly from this fixture geometry. The raw
    # runs, offsets, permutations, codes, and native tables do not survive.
    experts, rows, cols = 4, 128, 128
    words = (2 * (16 * 3 * cols) + 2 * (16 * 4 * cols)) * 4
    tables = experts * (1 << 14) * 2
    scales = experts * rows * 4
    init = experts * cols * 4
    has_init = experts * 4
    pairs = experts * 8 * 4
    descriptors_bytes = experts * (cols // 32) * 12 * 4
    counters = 2 * 2 * 4
    assert owner.resident_bytes() == 3 * (words + tables + scales + init + has_init + pairs + descriptors_bytes) + counters
    names = dict(owner.named_tensors())
    assert names["routed_classes.counters"] is adapter.counters
    assert adapter.counters.numel() == 4
    for cls in adapter.classes:
        assert cls.table_gate.untyped_storage().data_ptr() == owner.gate.table_all.untyped_storage().data_ptr()


def test_factory_does_not_substitute_compact_on_native_failure(monkeypatch):
    allow_cpu(monkeypatch)
    packed = PackedWindowMoeBundles(*[projection() for _ in range(3)], family="value",
                                   expert_classes=descriptors())
    monkeypatch.setattr(rf, "_ext", lambda library: (_ for _ in ()).throw(RuntimeError("no toolchain")))
    with pytest.raises(RuntimeError, match="no toolchain"):
        packed.adapter()
    assert "_adapter" not in packed.__dict__


@pytest.mark.parametrize("reverse", [False, True])
def test_two_stream_events_and_absolute_prefix_reseed(monkeypatch, reverse):
    from contextlib import contextmanager
    make_resources = rf._make_dispatch_resources
    allow_cpu(monkeypatch)
    adapter = rf.FusedRoutedWindowMoE.from_bundles(*[projection() for _ in range(3)],
        expert_classes=descriptors())
    trace, launches, created_streams, created_events = [], [], [], []
    active = ["caller"]
    class Stream:
        def __init__(self, name=None, *, device=None):
            self.name = name if name is not None else f"side{len(created_streams)}"
            if name is None:
                created_streams.append(self)
        def wait_event(self, event):
            trace.append(("wait", self.name, event.recorded))
    class Event:
        def __init__(self):
            created_events.append(self)
        def record(self, stream):
            self.recorded = stream.name
            trace.append(("record", stream.name))
    caller = Stream("caller")
    @contextmanager
    def context(stream):
        prior = active[0]
        active[0] = stream.name
        yield
        active[0] = prior
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: caller)
    monkeypatch.setattr(torch.cuda, "Stream", Stream)
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "stream", context)
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda tensor, stream: None)
    monkeypatch.setattr(rf, "_sm_count", lambda index: 132)
    class Library:
        def routed_fused_forward(self, *args):
            launches.append((active[0], args[25].clone(), args[22], args[21].clone(), args[-2], args[3]))
    monkeypatch.setattr(rf, "_ext", lambda library: Library())
    resources = make_resources(torch.device("cpu"), rf._LutClassKernel("value", Library()))
    assert len(created_streams) == 2
    assert len(created_events) == 3
    assert resources.empty.numel() == 0
    assert all(hasattr(event, "recorded") for event in created_events)
    trace.clear()
    ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.int32)
    rw = torch.ones_like(ids, dtype=torch.float32)
    routing = rf._routing_tables(ids, rw, 4, torch.device("cpu"), None)
    out = torch.empty(4, 128, dtype=torch.bfloat16)
    x = torch.empty(2, 128, dtype=torch.bfloat16)
    order = (1, 0) if reverse else (0, 1)
    def no_resource_creation(*args, **kwargs):
        raise AssertionError("a projection must reuse its load-time resources")
    monkeypatch.setattr(torch.cuda, "Stream", no_resource_creation)
    monkeypatch.setattr(torch.cuda, "Event", no_resource_creation)
    monkeypatch.setattr(torch, "empty", no_resource_creation)
    for prefix in ([0, 2, 3, 5, 5], [0, 0, 0, 1, 3], [0, 1, 6, 7, 8]):
        routing.item_off.copy_(torch.tensor(prefix, dtype=torch.int32))
        adapter.counters.fill_(-99)
        for mode, n_blocks in ((0, 2), (2, 1)):
            from tessera.routed_class_dispatch import dispatch_class_projection
            dispatch_class_projection(mode, x if mode == 0 else out, None, routing,
                parameters=adapter.operands, starts=adapter.operands["starts"],
                ends=adapter.operands["ends"], issue_order=order, counters=adapter.counters,
                resources=resources, mul_weight=False, limit=float("inf"),
                a_row_mode=0 if mode == 0 else 1, out=out)
            for launch, c in zip(launches[-2:], order):
                stream, counter, index, offsets, grid, empty = launch
                assert stream == f"side{c % 2}"
                assert counter.tolist() == [prefix[2*c] * n_blocks]
                assert index is routing.flat_sorted
                assert offsets.tolist() == routing.offsets[2*c:2*c+3].tolist()
                assert grid == 132
                assert empty is resources.empty
    assert trace[:3] == [("record", "caller"), ("wait", "side0", "caller"), ("wait", "side1", "caller")]
    assert trace[-4:] == [("record", "side0"), ("wait", "caller", "side0"),
                         ("record", "side1"), ("wait", "caller", "side1")]


def test_dispatch_resources_have_one_adapter_owner(monkeypatch):
    import gc
    import weakref
    allow_cpu(monkeypatch)
    adapters = [rf.FusedRoutedWindowMoE.from_bundles(*[projection() for _ in range(3)],
                expert_classes=descriptors()) for _ in range(2)]
    first, second = adapters
    assert first.dispatch_resources is not second.dispatch_resources
    assert first.dispatch_resources.ready is not second.dispatch_resources.ready
    assert not any(a is b for a in first.dispatch_resources.finished for b in second.dispatch_resources.finished)
    key = first.resource_key
    assert rf.resolve_dispatch_resources(key) is first.dispatch_resources
    ref = weakref.ref(first.dispatch_resources)
    del first, adapters
    gc.collect()
    assert ref() is None
    with pytest.raises(GrammarError, match="owner no longer exists"):
        rf.resolve_dispatch_resources(key)
    assert rf.resolve_dispatch_resources(second.resource_key) is second.dispatch_resources

