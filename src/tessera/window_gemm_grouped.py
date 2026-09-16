"""The grouped window GEMM: many experts, one launch, device-side routing.

WHAT IT SERVES.  A stack of window units (BF16 value family or FP8 E4M3
family) with per-expert tables, rate runs and row-cut history, computed
against one activation batch under a token->expert routing.  Every expert's
weight tile is decoded inside the kernel from its own packed planes and kept
in registers/SMEM; no decoded expert weight ever reaches global memory.

DEVICE-SIDE ROUTING.  ``__call__`` builds the token->expert CSR with device
operations only: a fixed-size ``scatter_add_`` histogram (never ``bincount``,
whose CUDA output size is data-dependent and synchronises), a stable
``argsort`` of the flat route ids, and a cumsum over the fixed ``[E]``
histogram.  One grid over ``(expert, N block, M block)`` follows; a block
whose token range is empty returns before decoding anything, and every
program gathers its tokens by index.  There is no CPU loop over tokens or
experts and no synchronisation in the forward.

MODES (matching vLLM's ``TopKWeightAndReduce`` semantics).

* ``preserve=True`` -- route-preserving first projection: fp32 accumulation,
  one bf16 output per route, shape ``[T, top_k, rows]``.  With
  ``apply_router_weight_on_input=True`` the route weight multiplies this
  projection's fp32 accumulator -- vLLM's gemm1 ``MUL_ROUTED_WEIGHT``
  placement, which is the input-weight semantics up to rounding; otherwise
  the weight belongs to the reduction stage.
* ``preserve=False`` (default) -- reduction: fp32 accumulation, the route
  weight applied to the fp32 accumulator unless ``apply_router_weight_on_input``
  (the input then already carries it), summed into ``[T, rows]`` with atomics
  and cast once to bf16.  ``route_input=True`` indexes ``x`` by route instead
  of by token, so the down projection of a two-stage MoE consumes the
  per-route activations directly and still reduces over ``top_k``.
* Both modes apply the family epilogues: BF16 row scale only (the research
  folded rounding has no API here); FP8 ``y = acc * a_scale[t] * w_scale[e, n]``
  under vLLM's native per-token quantizer.  ``prepare_grouped_window_gemm``'s
  ``arithmetic="folded"`` selects the **research BF16 folded contract**
  instead -- one bf16 rounding of ``(value * row_scale)`` per weight in
  registers before the dot, exactly ``bf16_route.decode_folded``'s
  ``(values.float() * scale[:, :, None]).to(torch.bfloat16)``, with no scale
  in the epilogue.  It is a prepare-time property, not a call flag, and the
  FP8 family refuses it; the dense/default epilogue is unchanged.

``preserve`` + a per-route activation + ``reduce`` is the two-stage MoE shape
(gate/up -> activation -> down); the module itself never assumes that a sum
of linears can stand in for the activation.  The placement follows vLLM's own
fused path rather than an algebraic equivalent: in
``model_executor/layers/fused_moe/fused_moe.py`` gemm1 is dispatched with
``apply_router_weight_on_input`` and each kernel multiplies the fp32
accumulator by the route weight when ``MUL_ROUTED_WEIGHT`` is set -- before
the nonlinear activation and before the next A-side quant -- while gemm2 is
dispatched with ``not apply_router_weight_on_input``; and
``topk_weight_and_reduce.py`` weights the per-route outputs only when the
input did not already carry the weights, then sums.  A combination this
module does not serve fails closed (``preserve`` with ``route_input`` is two
different stages) rather than silently changing the placement.

VALIDATION BOUNDARY.  Constants are validated once in
``prepare_grouped_window_gemm``; the forward checks cheap metadata only
(rank, width, dtype, device, contiguity, ``T`` agreement) and never reads a
tensor back to the host.  Routing ids are per-forward: ``routing_ids_ok`` is
a device predicate the caller inspects on its own schedule; a bad id is not
silently dropped (``scatter_add_`` refuses it on device).
"""
from __future__ import annotations

import dataclasses
from typing import Sequence

import torch
import triton
import triton.language as tl

from .errors import GrammarError
from .kernel_window_gemv import TILE_ROWS, WindowGemvUnit
from .window_gemm import prepare_window_gemm

__all__ = ["prepare_grouped_window_gemm", "PreparedGroupedWindowGemm", "routing_ids_ok"]


@triton.jit
def _grouped_window_gemm_kernel(
    words_all, table_all, codes_all, native_all, x_ptr, out_ptr,
    scale_all, a_scale_ptr, rw_ptr, flat_ptr, offset_ptr,
    runs_all, init_all, has_init_ptr,
    word_off_ptr, tile_words_ptr, total_words_ptr, run_off_ptr, perm_ptr,
    rows, cols, E, P, top_k,
    L: tl.constexpr, TILE: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    FP8: tl.constexpr, PRESERVE: tl.constexpr,
    MUL_WEIGHT: tl.constexpr, ROUTE_INPUT: tl.constexpr, FOLDED: tl.constexpr,
):
    e = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_m = tl.program_id(2)

    start = tl.load(offset_ptr + e)
    end = tl.load(offset_ptr + e + 1)

    if pid_m * BM < end - start:
        pos = start + pid_m * BM + tl.arange(0, BM)
        live_tok = pos < end
        safe_pos = tl.minimum(pos, P - 1)
        flat = tl.load(flat_ptr + safe_pos, mask=live_tok, other=0)
        tok = flat // top_k
        rw = tl.load(rw_ptr + safe_pos, mask=live_tok, other=0.0)

        n0 = pid_n * BN
        g = n0 // TILE
        t = n0 - g * TILE
        offs_n = n0 + tl.arange(0, BN)
        live_n = offs_n < rows

        word_off = tl.load(word_off_ptr + e)
        tile_words = tl.load(tile_words_ptr + e)
        total_words = tl.load(total_words_ptr + e)
        run_off = tl.load(run_off_ptr + e)
        n_runs = tl.load(run_off_ptr + e + 1) - run_off
        has_init = tl.load(has_init_ptr + e)
        words = words_all + word_off
        wscale = tl.load(scale_all + e * rows + offs_n, mask=live_n, other=0.0)

        rows_v = tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for r in range(n_runs):
            rate = tl.load(runs_all + (run_off + r) * 4 + 0)
            col0 = tl.load(runs_all + (run_off + r) * 4 + 1)
            ncols = tl.load(runs_all + (run_off + r) * 4 + 2)
            word0 = tl.load(runs_all + (run_off + r) * 4 + 3)
            CHUNK = 16 * rate
            tile_base = g * tile_words + word0

            for c0 in range(0, ncols, BK):
                offs_k = c0 + tl.arange(0, BK)
                live_k = offs_k < ncols
                live_k2 = live_k[:, None]
                kglob = col0 + offs_k
                base = tile_base + offs_k * CHUNK

                qq = (t + 1 + rows_v) * rate
                length = tl.where(has_init != 0, L,
                                  tl.minimum(qq + g * TILE * rate, L))
                qb = qq - L
                neg = qb < 0
                wi = tl.where(neg, -1, qb >> 5)
                d1 = wi + 1
                shift = 64 - qq + 32 * wi

                prev_off = -tile_words + CHUNK - 1
                idx_prev = base[:, None] + prev_off
                idx_norm = base[:, None] + wi[None, :]
                idx1 = base[:, None] + d1[None, :]
                live_prev = live_k2 & (g > 0) & (idx_prev >= 0) & (idx_prev < total_words)
                live_norm = live_k2 & (idx_norm >= 0) & (idx_norm < total_words)
                live1 = live_k2 & (idx1 >= 0) & (idx1 < total_words)
                w0_prev = tl.load(words + idx_prev, mask=live_prev, other=0).to(tl.int64) & 0xFFFFFFFF
                w0_norm = tl.load(words + idx_norm, mask=live_norm, other=0).to(tl.int64) & 0xFFFFFFFF
                w0_init = tl.load(init_all + e * cols + kglob, mask=live_k, other=0).to(tl.int64) & 0xFFFFFFFF
                w0 = tl.where(neg[None, :] & (g == 0), w0_init[:, None],
                              tl.where(neg[None, :], w0_prev, w0_norm))
                w1 = tl.load(words + idx1, mask=live1, other=0).to(tl.int64) & 0xFFFFFFFF

                combined = (w0 << 32) | w1
                state = (combined >> shift[None, :]) & ((1 << length) - 1)[None, :]

                if FP8:
                    code = tl.load(codes_all + e * (1 << L) + state, mask=live_k2, other=0)
                    byte = tl.load(native_all + e * 256 + code.to(tl.int32), mask=live_k2, other=0)
                    val = byte.to(tl.float8e4nv, bitcast=True)
                else:
                    val = tl.load(table_all + e * (1 << L) + state, mask=live_k2, other=0.0)
                    if FOLDED:
                        # the research BF16 contract: one bf16 rounding of
                        # (value * row scale) in registers, before the dot
                        val = (val.to(tl.float32) * wscale[None, :]).to(tl.bfloat16)

                kcol = tl.load(perm_ptr + e * cols + kglob, mask=live_k, other=0)
                if ROUTE_INPUT:
                    xrow = flat
                else:
                    xrow = tok
                xk = tl.load(
                    x_ptr + xrow[:, None] * cols + kcol[None, :],
                    mask=live_tok[:, None] & live_k[None, :], other=0.0
                )
                acc += tl.dot(xk, val, out_dtype=tl.float32)

        contrib = acc if FOLDED else acc * wscale[None, :]
        if FP8:
            if ROUTE_INPUT:
                a_s = tl.load(a_scale_ptr + flat, mask=live_tok, other=0.0)
            else:
                a_s = tl.load(a_scale_ptr + tok, mask=live_tok, other=0.0)
            contrib = contrib * a_s[:, None]
        if MUL_WEIGHT:
            contrib = contrib * rw[:, None]      # vLLM multiplies the fp32 accumulator
        if PRESERVE:
            tl.store(
                out_ptr + flat[:, None] * rows + offs_n[None, :],
                contrib.to(tl.bfloat16), mask=live_tok[:, None] & live_n[None, :],
            )
        else:
            tl.atomic_add(
                out_ptr + tok[:, None] * rows + offs_n[None, :],
                contrib, mask=live_tok[:, None] & live_n[None, :],
            )

@dataclasses.dataclass(frozen=True)
class PreparedGroupedWindowGemm:
    """The frozen stack: SoA planes, per-expert headers, geometry."""

    words_all: torch.Tensor
    table_all: torch.Tensor
    codes_all: torch.Tensor
    native_all: torch.Tensor
    scale_all: torch.Tensor
    runs_all: torch.Tensor
    init_all: torch.Tensor
    has_init: torch.Tensor
    word_off: torch.Tensor
    tile_words: torch.Tensor
    total_words: torch.Tensor
    run_off: torch.Tensor
    perm_all: torch.Tensor
    rows: int
    cols: int
    experts: int
    window_bits: int
    family: str
    block_m: int
    block_n: int
    block_k: int
    quantizer: str = "native"
    arithmetic: str = "epilogue"

    @property
    def device(self) -> torch.device:
        return self.words_all.device

    def __call__(self, x: torch.Tensor,
                 expert_ids: torch.Tensor,
                 routing_weights: torch.Tensor,
                 a_scale: "torch.Tensor | None" = None,
                 out: "torch.Tensor | None" = None,
                 *,
                 preserve: bool = False,
                 apply_router_weight_on_input: bool = False,
                 route_input: bool = False) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.cols or x.device != self.device:
            raise GrammarError(
                f"x must be a [T, {self.cols}] tensor on {self.device}, got "
                f"{tuple(x.shape)} on {x.device}"
            )
        if not x.is_contiguous():
            raise GrammarError("x must be contiguous; the kernel indexes it with one stride")
        if expert_ids.dim() != 2 or routing_weights.shape != expert_ids.shape:
            raise GrammarError("expert_ids and routing_weights must share [T, top_k]")
        if expert_ids.device != self.device or routing_weights.device != self.device:
            raise GrammarError("routing tensors must live on the compute device")
        if routing_weights.dtype != torch.float32:
            routing_weights = routing_weights.float()
        t_tokens, top_k = expert_ids.shape
        if preserve and route_input:
            raise GrammarError(
                "preserve mode and route_input describe two different stages: a "
                "route-preserving projection takes token-indexed input"
            )
        expected_rows = t_tokens * top_k if route_input else t_tokens
        if x.shape[0] != expected_rows:
            raise GrammarError(
                f"x has {x.shape[0]} rows, the routing calls for {expected_rows} "
                f"({'route' if route_input else 'token'}-indexed input)"
            )
        if t_tokens == 0:
            shape = (0, top_k, self.rows) if preserve else (0, self.rows)
            return torch.empty(shape, dtype=torch.bfloat16, device=self.device)

        if self.family == "value":
            if x.dtype != torch.bfloat16 or a_scale is not None:
                raise GrammarError("the value family takes a bf16 x and no activation scale")
            xq = x
            fp8 = False
        else:
            if x.dtype == torch.float8_e4m3fn:
                if a_scale is None:
                    raise GrammarError(
                        "an fp8 x must carry its per-token activation scale; a bare fp8 "
                        "or bf16 activation would change the contract"
                    )
                a_scale = a_scale.reshape(-1)
                if a_scale.numel() != expected_rows or a_scale.dtype != torch.float32 \
                        or a_scale.device != self.device:
                    raise GrammarError(
                        f"a_scale must be fp32 [{expected_rows}] on {self.device} "
                        f"({'route' if route_input else 'token'}-indexed input)"
                    )
                if not a_scale.is_contiguous():
                    raise GrammarError(
                        "a_scale must be contiguous after flattening; the kernel reads it "
                        "with one stride"
                    )
                xq = x
            else:
                if x.dtype != torch.bfloat16:
                    raise GrammarError(f"the E4M3 family takes bf16 or fp8 x, got {x.dtype}")
                if a_scale is not None:
                    raise GrammarError("a_scale belongs with a prequantized fp8 x")
                if self.quantizer != "native":
                    raise GrammarError(
                        "this bundle was prepared without a quantizer; pass the "
                        "prequantized fp8 activation together with its per-token scale"
                    )
                from .serving.native_ops import native_fp8_quant
                xq, s = native_fp8_quant(x)
                a_scale = s.reshape(-1)
            fp8 = True

        ids = expert_ids.reshape(-1)
        if ids.dtype not in (torch.int32, torch.int64):
            raise GrammarError("expert_ids must be int32 or int64")
        ids = ids.to(torch.int64)
        rw = routing_weights.reshape(-1).contiguous()
        p = int(ids.numel())

        if preserve:
            result = (torch.empty(t_tokens, top_k, self.rows, dtype=torch.bfloat16,
                                  device=self.device) if out is None else out)
            if result.shape != (t_tokens, top_k, self.rows) \
                    or result.dtype != torch.bfloat16 or result.device != self.device:
                raise GrammarError(
                    f"out must be bf16 [{t_tokens}, {top_k}, {self.rows}] on {self.device}"
                )
            if not result.is_contiguous():
                raise GrammarError("out must be contiguous; the kernel writes row-major")
        else:
            result = (torch.zeros(t_tokens, self.rows, dtype=torch.float32, device=self.device)
                      if out is None else out)
            if result.shape != (t_tokens, self.rows) or result.dtype != torch.float32 \
                    or result.device != self.device:
                raise GrammarError(f"out must be fp32 [{t_tokens}, {self.rows}] on {self.device}")
            if not result.is_contiguous():
                raise GrammarError("out must be contiguous; the kernel accumulates row-major")
            result.zero_()          # overwrite semantics: the caller's buffer is consumed

        if p:
            counts = torch.zeros(self.experts, dtype=torch.int32, device=self.device)
            counts.scatter_add_(0, ids, torch.ones_like(ids, dtype=torch.int32))
            offsets = torch.zeros(self.experts + 1, dtype=torch.int32, device=self.device)
            offsets[1:] = torch.cumsum(counts, 0)
            order = torch.argsort(ids, stable=True)
            flat_sorted = order.to(torch.int32).contiguous()
            rw_sorted = rw[order].contiguous()
            grid = (self.experts, triton.cdiv(self.rows, self.block_n),
                    triton.cdiv(p, self.block_m))
            _grouped_window_gemm_kernel[grid](
                self.words_all, self.table_all, self.codes_all, self.native_all, xq, result,
                self.scale_all,
                a_scale if a_scale is not None else self.scale_all.new_zeros(1),
                rw_sorted, flat_sorted, offsets,
                self.runs_all, self.init_all, self.has_init,
                self.word_off, self.tile_words, self.total_words, self.run_off, self.perm_all,
                self.rows, self.cols, self.experts, p, top_k,
                L=self.window_bits, TILE=TILE_ROWS,
                BM=self.block_m, BN=self.block_n, BK=self.block_k,
                FP8=fp8, PRESERVE=preserve,
                MUL_WEIGHT=(apply_router_weight_on_input == preserve),
                ROUTE_INPUT=route_input,
                FOLDED=(self.arithmetic == "folded"),
                num_warps=8,
            )
        return result if preserve else result.to(torch.bfloat16)


def routing_ids_ok(expert_ids: torch.Tensor, experts: int) -> torch.Tensor:
    """A device predicate: every id in ``[0, experts)``.  No synchronisation;
    the caller reads it when it chooses (never on the hot path)."""
    return ((expert_ids >= 0) & (expert_ids < experts)).all()


def prepare_grouped_window_gemm(
    units: Sequence[WindowGemvUnit],
    *,
    initial_state: "torch.Tensor | None" = None,
    block_m: int = 64,
    block_n: int = 64,
    block_k: int = 64,
    quantizer: "str | None" = "native",
    arithmetic: str = "epilogue",
) -> PreparedGroupedWindowGemm:
    """Validate every unit once and freeze the SoA stack.

    ``initial_state`` may be ``[E, cols]`` int32 in original column order (one
    row per expert); a unit's own ``initial_state`` field is used where the
    stack does not carry a row.  Families, ``cols``, ``rows`` and
    ``window_bits`` must agree across the stack, and so must the block sizes.

    ``arithmetic`` names the weight-side contract, once and explicitly:

    * ``"epilogue"`` (default, dense): the row scale multiplies the fp32
      accumulator after the dot -- the dense BF16 route's contract;
    * ``"folded"``: the research BF16 contract, exactly
      ``bf16_route.decode_folded``'s ``(values.float() * scale[:, :, None])
      .to(torch.bfloat16)`` -- one bf16 rounding of (value * row scale) in
      registers, before ``tl.dot``, and no scale in the epilogue.  The FP8
      family has no folded form (its per-token A quant and row-scale epilogue
      are the published contract), so ``"folded"`` is refused there.
    """
    if arithmetic not in ("epilogue", "folded"):
        raise GrammarError(f"unknown weight arithmetic {arithmetic!r}")
    units = list(units)
    if not units:
        raise GrammarError("a grouped stack needs at least one expert")
    prepared = []
    for e, unit in enumerate(units):
        row = None if initial_state is None else initial_state[e]
        prepared.append(prepare_window_gemm(
            unit, initial_state=row, block_m=block_m, block_n=block_n, block_k=block_k,
            quantizer=quantizer,
        ))
    first = prepared[0]
    for e, p in enumerate(prepared[1:], start=1):
        for name in ("family", "cols", "rows", "window_bits"):
            if getattr(p, name) != getattr(first, name):
                raise GrammarError(
                    f"expert {e}: {name}={getattr(p, name)} differs from expert 0's "
                    f"{getattr(first, name)}; a grouped stack is homogeneous"
                )
    device = first.device
    if arithmetic == "folded" and first.family != "value":
        raise GrammarError(
            "the folded weight arithmetic is the research BF16 contract; the E4M3 "
            "family keeps the per-token A quant and the row-scale epilogue"
        )
    run_lengths = torch.tensor([p.runs.numel() // 4 for p in prepared], dtype=torch.int32)
    run_off = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(run_lengths, 0)])
    word_sizes = torch.tensor([p.words.numel() for p in prepared], dtype=torch.int32)
    word_off = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(word_sizes, 0)[:-1]])
    return PreparedGroupedWindowGemm(
        words_all=torch.cat([p.words for p in prepared]).contiguous(),
        table_all=torch.stack([p.table for p in prepared]).contiguous(),
        codes_all=torch.stack([p.codes for p in prepared]).contiguous(),
        native_all=torch.stack([p.native for p in prepared]).contiguous(),
        scale_all=torch.stack([p.scale for p in prepared]).contiguous(),
        runs_all=torch.cat([p.runs for p in prepared]).contiguous(),
        init_all=torch.stack([p.init_perm for p in prepared]).contiguous(),
        has_init=torch.tensor([1 if p.has_init else 0 for p in prepared],
                              dtype=torch.int32, device=device),
        word_off=word_off.to(device),
        tile_words=torch.tensor([p.tile_words for p in prepared], dtype=torch.int32, device=device),
        total_words=torch.tensor([p.total_words for p in prepared], dtype=torch.int32, device=device),
        run_off=run_off.to(device),
        perm_all=torch.stack([p.perm for p in prepared]).contiguous(),
        rows=first.rows,
        cols=first.cols,
        experts=len(prepared),
        window_bits=first.window_bits,
        family=first.family,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        quantizer=quantizer if first.family == "e4m3" else "native",
        arithmetic=arithmetic,
    )
