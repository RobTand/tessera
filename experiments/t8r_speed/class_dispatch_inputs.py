"""Synthetic packed constants and saved pure-template comparators.

The production adapter owns dispatch. Raw launches here are reference controls only.
No exported artifact, recipe override, or rate qualification is produced.
"""
from __future__ import annotations

import torch

from bench_rates import build_projection, SWIGLU_LIMIT
from tessera import routed_fused as rf
from tessera.expert_classes import build_expert_metadata, inverse_expert_ids, normalize_expert_metadata
from tessera.native_window_moe import PackedWindowMoeBundles
from tessera.window_gemm_grouped import prepare_grouped_window_gemm_from_soa

SCHEDULES = {"q3_q4": (768, 1024), "three": (512, 768, 1024)}
SCHEDULES.update({f"uniform_q{q}": (q,)
                  for q in sorted({q for rates in SCHEDULES.values() for q in rates})})


def metadata(rates, experts):
    if experts % len(rates):
        raise ValueError("expert population must divide into equal storage classes")
    # Original ids alternate profiles. The metadata owner sorts each class.
    groups = {"w13": [[rates[e % len(rates)]] * 2 for e in range(experts)],
              "w2": [[rates[e % len(rates)]] for e in range(experts)]}
    result = build_expert_metadata(groups)
    storage = {name: [rows[e] for e in result["expert_ids"]] for name, rows in groups.items()}
    return normalize_expert_metadata(result["expert_ids"], result["expert_classes"], storage)


def packed_constants(rates, *, experts, hidden, inter, device, seed):
    """Reuse D41's seeded word builder, then the loader's exact flat SoA intake."""
    meta = metadata(rates, experts)
    bundles = []
    for role, rows, cols in ((0, inter, hidden), (1, inter, hidden), (2, hidden, inter)):
        pieces = []
        for desc in meta["expert_classes"]:
            q = desc["q256"]["w2" if role == 2 else "w13"][0]
            p = build_projection(rf, desc["end"] - desc["start"], rows, cols, q // 256, 0,
                                 seed + 100 * role + desc["start"], device, True)
            if p["window_bits"] != rf.WINDOW_BITS:
                raise ValueError(f"current recipe at q256={q} does not fit the native window owner")
            pieces.append(p)
        widths = torch.cat([torch.full((p["words"].shape[0],), p["words"].shape[1],
                                      dtype=torch.int32, device=device) for p in pieces])
        tile_words = torch.cat([torch.full((p["words"].shape[0],), p["tile_words"],
                                          dtype=torch.int32, device=device) for p in pieces])
        offsets = torch.cat((torch.zeros(1, dtype=torch.int32, device=device),
                             widths.cumsum(0, dtype=torch.int32)[:-1]))
        # Finite E4M3 byte codebook. Codes come from the existing D41 table builder.
        native = torch.arange(256, dtype=torch.int32, device=device)
        native[(native & 127) == 127] -= 1
        native = native.to(torch.uint8).expand(experts, 256).contiguous()
        empty = torch.empty(0, dtype=torch.uint8, device=device)
        bundles.append(prepare_grouped_window_gemm_from_soa(
            words_all=torch.cat([p["words"].flatten() for p in pieces]), table_all=empty,
            codes_all=torch.cat([p["table"] for p in pieces]), native_all=native,
            scale_all=torch.cat([p["scale"] for p in pieces]),
            runs_all=torch.cat([p["runs"][:, :4] for p in pieces]),
            init_all=torch.cat([p["init"] for p in pieces]),
            has_init=torch.cat([p["has_init"] for p in pieces]),
            word_off=offsets, tile_words=tile_words, total_words=widths,
            run_off=torch.arange(experts + 1, dtype=torch.int32, device=device),
            perm_all=torch.cat([p["perm"] for p in pieces]), rows=rows, cols=cols,
            experts=experts, window_bits=rf.WINDOW_BITS, family="e4m3"))
    packed = PackedWindowMoeBundles(*bundles, family="e4m3", expert_classes=meta["expert_classes"])
    inverse = torch.tensor(inverse_expert_ids(meta["expert_ids"]), dtype=torch.int32, device=device)
    return packed, meta, inverse


def routing_generations(meta, tokens, *, device, top_k=8):
    """Two fixed-shape generations change ids, weights and class populations."""
    classes = meta["expert_classes"]
    count = len(classes)
    splits = {1: ([8], [8]), 2: ([4, 4], [1, 7]), 3: ([3, 3, 2], [0, 1, 7])}[count]
    global_ids = torch.tensor(meta["expert_ids"], dtype=torch.int32, device=device)
    generations, populations = [], []
    for generation, split in enumerate(splits):
        columns = []
        for desc, n in zip(classes, split):
            width = desc["end"] - desc["start"]
            t = torch.arange(tokens, device=device).reshape(-1, 1)
            j = torch.arange(n, device=device).reshape(1, -1)
            columns.append(desc["start"] + (t * max(n, 1) + j + generation * 17) % width)
        storage_ids = torch.cat(columns, 1).to(torch.int32)
        if generation:
            storage_ids = storage_ids.flip(1)
        ids = global_ids.index_select(0, storage_ids.flatten().long()).view(tokens, top_k)
        weights = (torch.arange(tokens * top_k, device=device, dtype=torch.float32).view(tokens, top_k)
                   .remainder(11) + 1 + generation * 3)
        weights = weights / weights.sum(1, keepdim=True)
        generations.append((ids, weights))
        populations.append({"generation": generation, "class_routes": [tokens * n for n in split]})
    return generations, populations


def activations(tokens, hidden, inter, device, seed, *, top_k=8):
    gen = torch.Generator(device=device).manual_seed(seed + tokens)
    x = (torch.randn(tokens, hidden, generator=gen, device=device) * 0.05).to(torch.bfloat16)
    down = (torch.randn(tokens * top_k, inter, generator=gen, device=device) * 0.05).to(torch.bfloat16)
    return x, down


class PureControlWorkspace:
    """The old adapter owns two counters before any call or graph capture."""
    def __init__(self, device):
        self.counters = torch.zeros(2, dtype=torch.int32, device=device)

    def launch_seat(self, mode):
        # Match the old _launch, including its per-launch empty scale fallback.
        empty = self.counters.new_zeros(0, dtype=torch.float32)
        projection = 1 if mode == 2 else 0
        slot = self.counters[projection:projection + 1]
        slot.zero_()  # Captured device initialization, not a fresh counter.
        return slot, empty


def pure_launch(adapter, cls, mode, x, scale, routing, out, *, workspace, a_row_mode, mul_weight):
    """Old pure-template entry with its persistent, in-stream-reset counter seat."""
    down = mode == 2
    b0, b1 = (cls.down, cls.down) if down else (cls.gate, cls.up)
    suffix0, suffix1 = ("down", "down") if down else ("gate", "up")
    bm = rf.superblock_rows(adapter.library, mode, routing.tokens)
    counter, empty = workspace.launch_seat(mode)
    rf._ext(adapter.library).routed_fused_forward(
        mode, adapter.fp8, x, scale if scale is not None else empty,
        getattr(cls, "words_" + suffix0), getattr(cls, "words_" + suffix1),
        getattr(cls, "table_" + suffix0), getattr(cls, "table_" + suffix1),
        b0.init_all, b1.init_all, b0.has_init, b1.has_init, b0.scale_all, b1.scale_all,
        getattr(cls, "runs_" + suffix0), getattr(cls, "runs_" + suffix1),
        getattr(cls, "bdesc_" + suffix0), getattr(cls, "bdesc_" + suffix1),
        cls.tile_words_down if down else cls.tile_words_gate_up,
        cls.slot_words_down if down else cls.slot_words_gate_up, adapter.piece_major,
        routing.offsets, routing.flat_sorted, routing.rw_sorted, routing.superblocks(bm), counter,
        routing.top_k, a_row_mode, mul_weight, SWIGLU_LIMIT, out,
        torch.cuda.get_device_properties(x.device).multi_processor_count, bm)


def stitched_references(adapter, x, down_x, storage_ids, weights):
    """Save class-local pure outputs in original flat-route coordinates, then stitch.

    Gate/up is mode zero with the native SwiGLU epilogue. Standalone down takes
    original flat-route activations. The full comparator preserves the sorted
    activation boundary and the native fixed-order token sum.
    """
    hidden, inter = x.shape[1], down_x.shape[1]
    routes, top_k = storage_ids.numel(), storage_ids.shape[1]
    gate_flat = torch.empty(routes, inter, dtype=torch.bfloat16, device=x.device)
    down_flat = torch.empty(routes, hidden, dtype=torch.bfloat16, device=x.device)
    full_flat = torch.empty_like(down_flat)
    flat_ids = storage_ids.flatten()
    workspaces = [PureControlWorkspace(x.device) for _ in adapter.classes]
    for cls, workspace in zip(adapter.classes, workspaces):
        take = torch.where((flat_ids >= cls.start) & (flat_ids < cls.end))[0]
        if take.numel() == 0:
            continue
        local_ids = (flat_ids[take] - cls.start).reshape(-1, 1)
        rw = weights.flatten()[take].reshape(-1, 1)
        widths = tuple(dict.fromkeys(rf.superblock_rows(adapter.library, mode, take.numel())
                                     for mode in (0, 1, 2)))
        local = rf._routing_tables(local_ids, rw, cls.end - cls.start, x.device, widths)
        xq, scale = adapter._quantized(x[take // top_k], None, take.numel())
        act = torch.empty(take.numel(), inter, dtype=torch.bfloat16, device=x.device)
        pure_launch(adapter, cls, 0, xq, scale, local, act, workspace=workspace, a_row_mode=0, mul_weight=False)
        original = torch.empty_like(act)
        original[local.flat_sorted.long()] = act
        gate_flat[take] = original
        qdown, dscale = adapter._quantized(down_x[take], None, take.numel())
        output = torch.empty(take.numel(), hidden, dtype=torch.bfloat16, device=x.device)
        pure_launch(adapter, cls, 2, qdown, dscale, local, output, workspace=workspace, a_row_mode=2, mul_weight=True)
        down_flat[take] = output
        qa, ascale = adapter._quantized(act, None, take.numel())
        pure_launch(adapter, cls, 2, qa, ascale, local, output, workspace=workspace, a_row_mode=1, mul_weight=True)
        full_flat[take] = output
    full = torch.empty_like(x)
    rf._ext(adapter.library).token_sum(full_flat, full, top_k)
    routing = adapter._routing(storage_ids, weights)
    return {"gate_up": gate_flat[routing.flat_sorted.long()].clone(),
            "down": down_flat.clone(), "full": full.clone()}


class Invocation:
    """One static input seat; production routing is rebuilt on every invocation."""
    def __init__(self, adapter, inverse, x, down_x, generations, kind, *, pure=False):
        self.adapter, self.inverse = adapter, inverse
        self.x, self.down_x, self.generations, self.kind = x, down_x, generations, kind
        self.ids, self.weights = (t.clone() for t in generations[0])
        self.pure = pure
        self.pure_workspace = PureControlWorkspace(x.device) if pure else None
        routes = self.ids.numel()
        shape = (routes, down_x.shape[1] if kind == "gate_up" else x.shape[1])
        if kind == "full":
            shape = x.shape
        self.out = torch.empty(shape, dtype=torch.bfloat16, device=x.device)

    def select(self, generation):
        ids, weights = self.generations[generation]
        self.ids.copy_(ids)
        self.weights.copy_(weights)

    def __call__(self):
        adapter = self.adapter
        # Same inverse gather as the plugin boundary. No host route-count reads.
        storage = self.inverse.index_select(0, self.ids.flatten().long()).view_as(self.ids)
        if self.kind == "full" and not self.pure:
            self.out = adapter(self.x, storage, self.weights, swiglu_limit=SWIGLU_LIMIT)
            return self.out
        routing = adapter._routing(storage, self.weights)
        source = self.x if self.kind in ("gate_up", "full") else self.down_x
        xq, scale = adapter._quantized(source, None, source.shape[0])
        mode = 0 if self.kind in ("gate_up", "full") else 2
        row_mode = 0 if mode == 0 else 2
        if self.kind == "full":
            gate_out = torch.empty(self.ids.numel(), self.down_x.shape[1],
                                   dtype=torch.bfloat16, device=self.x.device)
        else:
            gate_out = self.out
        if self.pure:
            pure_launch(adapter, adapter.classes[0], mode, xq, scale, routing, gate_out,
                        workspace=self.pure_workspace, a_row_mode=row_mode, mul_weight=mode == 2)
        else:
            adapter._launch(mode, xq, scale, routing, a_row_mode=row_mode,
                            mul_weight=mode == 2, limit=SWIGLU_LIMIT, out=gate_out)
        if self.kind == "full":
            aq, a2 = adapter._quantized(gate_out, None, self.ids.numel())
            routed = torch.empty(self.ids.numel(), self.x.shape[1], dtype=torch.bfloat16, device=self.x.device)
            pure_launch(adapter, adapter.classes[0], 2, aq, a2, routing, routed,
                        workspace=self.pure_workspace, a_row_mode=1, mul_weight=True)
            rf._ext(adapter.library).token_sum(routed, self.out, self.ids.shape[1])
        return self.out
