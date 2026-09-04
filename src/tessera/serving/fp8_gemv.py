"""The streamed FP8 route's decode-regime GEMV: the wire read once, never materialised.

The streamed route holds the packed window streams and, before this module,
decoded them into a fresh tile per forward and called ``torch._scaled_mm`` --
two passes over the data and a materialised tile.  The fused window-body GEMV
(``tessera.kernel_window_gemv``: its DECODED TILE bit-exact against the torch
decoder's on 196/196 reach units -- a claim about the bytes, never about the
GEMM over them; ~1.9x the resident FP8 lane per token at M=1 on the 4B shapes)
reads the wire once and never materialises it; this module wires it into the forward for the decode
regime (M <= 8) and keeps the materialised path for prefill.

THE ACTIVATION CONTRACT DOES NOT CHANGE.  The GEMV is a W4A16 kernel (bf16
``x`` in, fp32 accumulation), while this route is W8A8 (per-token-dynamic FP8
``x``).  The dispatch therefore runs the route's own FP8 quantiser first and
hands the GEMV the quantised E4M3 **codes** -- every legal E4M3 byte is exact
in bf16 (``test_every_legal_e4m3_byte_is_exact_in_bf16``) -- with the
per-token ``a_scale`` applied to the kernel's fp32 output.  Feeding it raw
bf16 activations would be a silent contract change (less quant noise, a
different KL) and is refused by construction here: the quantiser runs on every
path, and what the kernel multiplies is its output.

THE SCALE IS APPLIED ON THE OUTPUT, NEVER FOLDED INTO A BF16 OPERAND -- the
rule ``bf16_route`` states for the weight side, held here for the activation
side.  ``a_q * a_scale`` is NOT exact in bf16: a code carries four significant
bits and an fp32 scale twenty-four, so their product needs up to twenty-eight
and bf16 keeps eight.  Folding it in cost 1.6e-03 relative rms on EVERY
activation element -- some 800x the fp32 accumulation floor -- where
``_scaled_mm`` multiplies the codes and scales in its epilogue.  Applying
``a_scale`` on the fp32 output leaves fp32 summation order as the only
difference this module can name, which is what the docstring always claimed --
and that order is MEASURED, not asserted: the lane run twice on identical
input differs by 1.6e-07 relative and changes no bf16 output word at any M
(``experiments/gemv_a_side_precision.py``).

THIS DOES NOT CLOSE #110.  The lane and its published fallback disagreed as
served at M = 1 -- mutual KL 0.012111, top-1 91.02%, byte-identical bytes (one
inode).  Reading the artefacts, this branch is the ONLY place that difference
can live: prefill, where both arms take ``_materialised_path``, read exactly
0.000000 over 4088 positions, and the two serve logs differ in nothing but the
112 intended refusals.  Reading a propagation screen, the fold is only ~1/40 of
what was measured (KL 3.2e-04 at the served position set).  Those two readings
disagree by a factor of forty and this session could not run the one
measurement that decides it -- the served re-run after the fix.  Do not read
the fix below as an explanation of the served number, and do not read the
screen as proof that a second term exists;
``docs/measurements/tessera-gemv-a-side-2026-09-04.md`` holds both.

THE DISPATCH LIVES INSIDE A FUNCTIONAL CUSTOM OP.  The token count is
symbolic under vLLM's compiled forward, so a Python branch on it would
specialise the graph per batch (or fail the trace on an unbacked size); the
refusals, the M tile and the prefill/materialised fallback all happen inside
``tessera::fp8_streamed_apply``, where ``x.shape[0]`` is a concrete integer --
exactly the shape ``tessera_window_gemv::gemv`` already takes for the same
reason.  The op owns every tensor it returns; no per-device pool is mutated,
no ``data_ptr`` is compared, no ``int()`` on the token dim executes in the
trace.  The extension is resolved at preparation (``prepare_fp8_gemv`` builds
or finds it), so the first call -- under a compiled forward, the trace itself
-- never builds.

FALLBACK, NAMED.  ``prepare_fp8_gemv`` raises where the lane cannot serve: a
rate-3 column, L != 14, a non-WINDOW body or non-CHANNEL plane, a shard start
state, or an extension that cannot build.  The route serves those units
through the torch window decode exactly as before (``ext.
substitutes_when_unavailable`` with this lane's entry says both residencies
substitute ``torch_window``).  At call time the op routes M > 8 and the
rate-1-at-M>=4 corner (no 8-row lane) to the materialised path itself, so a
served unit is never refused for its batch shape.
"""
from __future__ import annotations

from typing import List, Sequence

import torch

from .ext import WINDOW_GEMV_MODULE_NAME
from .scheme import ROUTES, TESSERA_FP8, WINDOW_GEMV_SYMBOL, launch_pairs
from .telemetry import DECODER_TORCH_WINDOW, DECODER_WINDOW_GEMV

__all__ = [
    "GEMV_MODULE_NAME",
    "GEMV_SYMBOL",
    "GEMV_MAX_M",
    "m_tile",
    "COMPILED_SYMBOL",
    "STREAMED_APPLY_OP",
    "COMPILED_DECODER",
    "PreparedFp8Gemv",
    "prepare_fp8_gemv",
    "decode_is_gemv",
    "streamed_apply",
    "holder_decode",
    "reference_bytes_for_test",
    "census_expected",
]

#: The JIT module name the load path asks for -- ``ext``'s constant, so the
#: contract table and the load call cannot drift.
GEMV_MODULE_NAME = WINDOW_GEMV_MODULE_NAME

#: Stamped on a route record whose launch was the window GEMV: the custom op
#: actually invoked, the honest spelling of "which GEMM ran".  Read off
#: ``scheme.WINDOW_GEMV_SYMBOL`` rather than spelled here, for the reason
#: ``GEMM_SYMBOL`` below is read off ``ROUTES``: the contract validator and
#: this dispatch must name one op, and this module imports torch so the
#: torch-free side cannot import it.
GEMV_SYMBOL = WINDOW_GEMV_SYMBOL

#: The route-table symbol of the materialised path, read off ROUTES rather
#: than restated: the census compares the record against this table.
GEMM_SYMBOL = ROUTES[TESSERA_FP8]["gemm_symbol"]
ACTIVATION_CONTRACT = ROUTES[TESSERA_FP8]["activation_contract"]

#: What a compiled record stamps.  One graph serves every M, so no single
#: path's symbol is true of every launch through it; the honest static answer
#: is the pair, in one string each owned here and read by the census.
COMPILED_SYMBOL = f"{GEMM_SYMBOL}+{GEMV_SYMBOL}"
COMPILED_DECODER = f"{DECODER_TORCH_WINDOW}+{DECODER_WINDOW_GEMV}"

#: The op the streamed GEMV lane dispatches through -- the node a compiled
#: forward contains when the holder prepared, and nothing at all when it did
#: not.  A constant because the compile-cache identity declares it
#: (``compile_identity.note_traced_dispatch``) and the custom-op registration
#: below reads it: the string a key is built from and the string torch
#: dispatches on must be one string.
STREAMED_APPLY_OP = "tessera::fp8_streamed_apply"

#: Tensors per role in the flattened op arguments, in order.
_ROLE_TENSORS = ("words", "items_1", "items_4", "perm", "table", "scale",
                 "runs", "codes", "native")
#: Ints per role in the flattened op metadata, in order.
_ROLE_INTS = ("tile_words", "rows", "window_bits", "rpl", "warps", "blocks",
              "max_cols_1", "max_cols_4", "rate_one", "uniform", "n_tiles")


def _m_tile(M: int) -> int:
    """The kernel build an M-wide batch runs on, off the lane's own rule."""
    from tessera.kernel_window_gemv import _m_tile as _kernel_m_tile

    return _kernel_m_tile(M)


def m_tile(M: int) -> int:
    """Public spelling of the routing rule above, for the route's telemetry."""
    return _m_tile(M)


def _gemv_max_m() -> int:
    from tessera import kernel_window_gemv as kg

    return kg.GEMV_MAX_M


#: The widest batch the GEMV serves; wider is the materialised path.  Read off
#: the lane, never restated, so a wider kernel widens this route with no edit.
GEMV_MAX_M = _gemv_max_m()


class _RoleGemv:
    """One role's GEMV lane: the repacked wire plus its two item tables."""

    __slots__ = ("name", "row_offset", "tensors", "meta")

    def __init__(self, name: str, row_offset: int, tensors: Sequence[torch.Tensor],
                 meta: Sequence[int]):
        self.name = str(name)
        self.row_offset = int(row_offset)
        self.tensors = tuple(tensors)
        self.meta = tuple(int(v) for v in meta)
        assert len(self.tensors) == len(_ROLE_TENSORS)
        assert len(self.meta) == len(_ROLE_INTS)

    def field(self, key: str):
        return self.tensors[_ROLE_TENSORS.index(key)]

    def scalar(self, key: str) -> int:
        return self.meta[_ROLE_INTS.index(key)]


class PreparedFp8Gemv:
    """A streamed module's decode-regime lane, prepared once on a device."""

    __slots__ = ("__roles", "__rows", "__columns", "__device")

    def __init__(self, roles: Sequence[_RoleGemv], *, rows: int, columns: int,
                 device: torch.device):
        self.__roles = tuple(roles)
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__device = device
        if sum(r.scalar("rows") for r in self.__roles) != self.__rows:
            raise ValueError("prepared GEMV roles do not stack to the module's rows")

    @property
    def rows(self): return self.__rows
    @property
    def columns(self): return self.__columns
    @property
    def device(self): return self.__device
    @property
    def role_names(self): return tuple(r.name for r in self.__roles)
    @property
    def rate_one(self) -> bool:
        """Whether any role carries a rate-1 column (no 8-row lane at M >= 4)."""
        return any(bool(r.scalar("rate_one")) for r in self.__roles)

    def resident_bytes(self) -> int:
        """Device bytes the repacked wire and its tables occupy."""
        return sum(t.numel() * t.element_size()
                   for r in self.__roles for t in r.tensors)

    def op_args(self):
        """The holder as one custom-op call: flat tensors, flat ints, geometry."""
        tensors: List[torch.Tensor] = []
        meta: List[int] = []
        for r in self.__roles:
            tensors.extend(r.tensors)
            meta.extend(r.meta)
        return tensors, meta, self.__rows, self.__columns


def prepare_fp8_gemv(parsed_roles, device=None, *, expected) -> PreparedFp8Gemv:
    """``[(role, ParsedUnit)]`` in stacking order -> the GEMV lane.

    Every role is repacked through ``kernel_window_gemv.prepare_from_parsed``
    (which resolves the extension, so the first forward never builds).  The
    lane's kernel decode of the whole module is then compared byte for byte
    against ``expected`` -- ``(bytes uint8 [rows, cols], scale fp32 [rows])``,
    the bytes the route's verified torch decoder produced for the same module,
    so this check ties the two serving decoders together rather than re-reading
    the reference.  A refusal (rate-3, L != 14, a shard start state, no
    toolchain) or a disagreement propagates to the caller, which serves the
    unit through the torch path.
    """
    from tessera import kernel_window_gemv as kg

    device = torch.device("cuda" if device is None else device)
    if not parsed_roles:
        raise ValueError("a Tessera GEMV module needs at least one role")
    roles = []
    offset = 0
    columns = None
    for name, parsed in parsed_roles:
        unit = kg.prepare_from_parsed(parsed)
        if columns is None:
            columns = unit.cols
        elif unit.cols != columns:
            raise ValueError(f"role {name!r} has {unit.cols} input columns, the module {columns}")
        (words, items_1, items_4, perm, table, scale,
         tile_words, unit_rows, window_bits, rpl, warps, blocks,
         max_cols_1, max_cols_4, rate_one, uniform) = kg._op_args(unit)
        # ``runs`` is read on the host only (``decode_typed`` takes its fields
        # through ``.item()`` and never passes the tensor to the device), so it
        # is kept on CPU: a CUDA ``.item()`` synchronises the device on every
        # prefill decode, and the op's other inputs stay graph-captured device
        # tensors either way.
        tensors = (words, items_1, items_4, perm, table, scale,
                   unit.rep.runs.cpu(), unit.codes_of_state, unit.native)
        meta = (tile_words, unit_rows, window_bits, rpl, warps, blocks,
                max_cols_1, max_cols_4, rate_one, uniform, unit.rep.n_tiles)
        roles.append(_RoleGemv(name, offset, tensors, meta))
        offset += unit_rows
    holder = PreparedFp8Gemv(roles, rows=offset, columns=columns, device=device)
    exp_bytes, exp_scale = expected
    got_bytes, got_scale = holder_decode(holder)
    if not torch.equal(got_bytes, exp_bytes.to(got_bytes.device)):
        wrong = int((got_bytes != exp_bytes.to(got_bytes.device)).sum())
        raise RuntimeError(
            f"the GEMV repack disagrees with the torch window decoder on {wrong} of "
            f"{exp_bytes.numel()} bytes; refusing to serve bytes the verified decoder "
            "would not produce")
    if not torch.equal(got_scale, exp_scale.to(got_scale.device)):
        raise RuntimeError("the GEMV row scale disagrees with the torch decoder's; refusing")
    return holder


def decode_is_gemv(holder: PreparedFp8Gemv, M: int) -> bool:
    """The dispatch rule, in one place: the op and the telemetry read it.

    M past the lane's max is prefill (the materialised path); M >= 4 over a
    rate-1 column has no 8-row lane (the materialised path serves the batch).
    Everything else in the decode regime reads the wire directly.
    """
    M = int(M)
    if M > GEMV_MAX_M:
        return False
    if _m_tile(M) >= 4 and holder.rate_one:
        return False
    return True


def _role_view(tensors: List[torch.Tensor], meta: List[int], i: int):
    """The i-th role's slice of the flattened op arguments."""
    nt, ni = len(_ROLE_TENSORS), len(_ROLE_INTS)
    return tensors[nt * i:nt * (i + 1)], meta[ni * i:ni * (i + 1)]


def _gemv_path(a_val: torch.Tensor, tensors: List[torch.Tensor], meta: List[int]) -> torch.Tensor:
    """``a_val [M, K]`` bf16 (the dequantised FP8 activations) -> fp32 ``[M, N]``."""
    from tessera import kernel_window_gemv as kg

    outs = []
    for i in range(len(meta) // len(_ROLE_INTS)):
        (words, items_1, items_4, perm, table, scale, _runs, _codes, _native), m = \
            _role_view(tensors, meta, i)
        (tile_words, rows, window_bits, rpl, warps, blocks,
         max_cols_1, max_cols_4, rate_one, uniform, _n_tiles) = m
        outs.append(kg._gemv_concrete(
            a_val, words, items_1, items_4, perm, table, scale,
            int(tile_words), int(rows), int(window_bits), int(rpl), int(warps),
            int(blocks), int(max_cols_1), int(max_cols_4),
            bool(rate_one), bool(uniform), 0))
    return torch.cat(outs, 1)


def _materialised_path(a_q: torch.Tensor, a_scale: torch.Tensor, scale_b: torch.Tensor,
                       tensors: List[torch.Tensor], meta: List[int], cols: int) -> torch.Tensor:
    """Kernel-decode the tile and run the route's own ``_scaled_mm``."""
    from tessera import kernel_window_gemv as kg

    ext = kg._ext()
    parts = []
    for i in range(len(meta) // len(_ROLE_INTS)):
        (words, _i1, _i4, perm, _table, _scale, runs, codes, native), m = \
            _role_view(tensors, meta, i)
        (tile_words, rows, window_bits, _rpl, _warps, _blocks,
         _mc1, _mc4, _ro, _uni, n_tiles) = m
        out = torch.empty(int(rows), cols, dtype=torch.uint8, device=a_q.device)
        ext.window_decode(words, int(tile_words), int(n_tiles), runs, perm, codes,
                          int(window_bits), out)
        parts.append(native[out.long()].view(torch.float8_e4m3fn))
    b = torch.cat(parts, 0)
    return torch._scaled_mm(a_q, b.t(), scale_a=a_scale, scale_b=scale_b,
                            out_dtype=torch.bfloat16)


# The forward's dispatch is FUNCTIONAL: the op owns the tensor it returns, so
# a compiled forward traces it as one opaque node -- no branch on the token
# dim, no mutation of an aliased pool (the failure ``ops`` documents at
# length), no data-pointer comparison.  The ``(tensors, meta)`` flattening is
# the same shape as ``ops._nvfp4_decode_module``'s ``(planes, scalars)``.
@torch.library.custom_op(STREAMED_APPLY_OP, mutates_args=())
def streamed_apply(a_q: torch.Tensor, a_scale: torch.Tensor, scale_b: torch.Tensor,
                   tensors: List[torch.Tensor], meta: List[int],
                   rows: int, cols: int) -> torch.Tensor:
    M = a_q.shape[0]
    nroles = len(meta) // len(_ROLE_INTS)
    rate_one = any(bool(meta[len(_ROLE_INTS) * i + _ROLE_INTS.index("rate_one")])
                   for i in range(nroles))
    # The tile is read only on the GEMV branch: past the lane's max the batch
    # is prefill and ``_m_tile`` refuses it by name.
    if M <= GEMV_MAX_M and not (_m_tile(M) >= 4 and rate_one):
        # The CODES, exact in bf16; ``a_scale`` multiplies the fp32 output.
        # Folding it into the operand instead rounds every element to bf16 and
        # is the whole of #110 -- see the module docstring.
        a_val = a_q.float().to(torch.bfloat16).contiguous()
        return (_gemv_path(a_val, tensors, meta) * a_scale).to(torch.bfloat16)
    return _materialised_path(a_q, a_scale, scale_b, tensors, meta, cols)


@streamed_apply.register_fake
def _streamed_apply_fake(a_q, a_scale, scale_b, tensors, meta, rows, cols):
    return a_q.new_empty((a_q.shape[0], rows), dtype=torch.bfloat16)


def holder_decode(holder: PreparedFp8Gemv):
    """Debug/test: ``(bytes uint8 [rows, cols], scale fp32 [rows])`` -- what the
    torch decoder verified at load, from the kernel's own decode.  Never a
    serving path (the serve never materialises)."""
    from tessera import kernel_window_gemv as kg

    ext = kg._ext()
    tensors, meta, _rows, cols = holder.op_args()
    byte_parts, scale_parts = [], []
    for i in range(len(meta) // len(_ROLE_INTS)):
        (words, _i1, _i4, perm, _table, scale, runs, codes, native), m = \
            _role_view(tensors, meta, i)
        (tile_words, rows, window_bits, _rpl, _warps, _blocks,
         _mc1, _mc4, _ro, _uni, n_tiles) = m
        out = torch.empty(int(rows), cols, dtype=torch.uint8, device=words.device)
        ext.window_decode(words, int(tile_words), int(n_tiles), runs, perm, codes,
                          int(window_bits), out)
        byte_parts.append(native[out.long()])
        scale_parts.append(scale)
    return torch.cat(byte_parts, 0), torch.cat(scale_parts, 0)


def reference_bytes_for_test(parsed):
    """The definition, for synthetic test units: grid codes from the state
    recursion (``kernel_window_gemv.reference_states``), native bytes through
    the grid map, the row scale in the reader's fp32 expression."""
    from tessera import kernel_window_gemv as kg

    unit, grid = parsed.unit, parsed.grid
    body = unit.body_bits.cuda()
    rates = tuple(int(r) for r in unit.rates)
    states = kg.reference_states(body, rates, int(unit.window_bits))
    native = torch.tensor(grid.native, dtype=torch.uint8, device=body.device)
    codes = unit.window_codes.to(body.device)
    scale = (unit.scale_rows.to(body.device).float() * float(unit.scale_global)).reshape(-1)
    return native[codes[states].long()], scale


def census_expected(*, compiled: bool):
    """The ``(symbol, decoder)`` pairs an FP8 module may report, by regime.

    Owned here -- the dispatch lives here -- and read by the route census, so
    a new path updates the expectation where the path was added rather than in
    a second spelling in the tool.  The GEMV lane's prefill decodes through
    the lane's kernel decode (never materialised on the serve, but a tile all
    the same), so a materialised launch on a GEMV-prepared module stamps the
    lane's decoder, not the torch window's.  Decode admits every pair the
    dispatch can take (rate-1 units and extension-less boxes serve
    materialised inside the decode regime); a compiled record covers both
    regimes in one graph and stamps the combined pair (plus the torch pair
    where no GEMV lane was prepared).
    """
    decode = launch_pairs(TESSERA_FP8, regime="decode")
    batch = launch_pairs(TESSERA_FP8, regime="batch")
    if compiled:
        combined = {(COMPILED_SYMBOL, COMPILED_DECODER)}
        return {"decode": combined | batch, "batch": combined | batch}
    return {"decode": decode, "batch": batch}
