"""The Tessera 16-bit W16A16 dense route: a BF16 wire served as a bf16 tile.

WHAT IT SERVES.  Tessera's BF16 wire -- the window body over the CHANNEL scale
plane, the *identical* body and plane the E4M3 route ships, with only the
alphabet the 2^L table snaps to changed -- decoded to an ordinary
``torch.bfloat16`` tensor and multiplied by the BF16 GEMM the runtime already
has.  There is no weight-side hardware format to satisfy and nothing to pack:
on this grid a code IS a bf16 bit pattern, so the table gather yields the tile.

WHY THE FAMILY EXISTS.  The window body's error over the E4M3 alphabet
saturates at ~0.022 out-space from R = 6 upward -- the floor is the
*alphabet's* resolution, not the trellis's -- while the identical body over
bf16 keeps halving at ~1.93x per bit through R = 7
(``docs/measurements/tessera16-alphabet-floor-2026-09-02.md``).  Above ~6 bpp
an 8-bit tile has nothing left to buy, so the route that lets an allocator
spend 7 bits usefully is the one whose alphabet is not the constraint.

THE ROW SCALE IS NEVER FOLDED INTO THE TILE.  This is the rule the family is
for, and it is a rule about *where* a factor is applied and not a trick.  A
CHANNEL scale is one factor per **output row**, and an output-row factor
commutes with the matmul: ``x (s * W)^T = (x W^T) * s``.  So the route runs the
GEMM on the raw table values -- exact, since every table entry is already a
bf16 word -- and applies ``s`` in fp32 to the GEMM's fp32 output, one rounding
at the end.  Folding instead costs 0.0011-0.0022 in absolute relative output
error at *any* rate, because it is a property of bf16's 7-bit mantissa rather
than of the coder: 2% of the error at R = 4 and **16% at R = 7**
(``tessera16-alphabet-floor`` B).  ``tessera.decode.materialize_bf16`` returns
the pair for exactly this reason and the encoder refuses to fold; this route
matches.  ``materialize_bf16_folded`` is the *twin's* rendering -- what a
one-tensor BF16 checkpoint has no choice but to ship -- and it is not reachable
from here.

The GEMM is therefore ``torch.mm(..., out_dtype=torch.float32)``: the stock
bf16 mainloop with its fp32 accumulator handed out unrounded, so the row scale
multiplies the accumulator and not a bf16 truncation of it.  That is the same
epilogue ``kernel_window_gemv`` applies on its accumulated fp32 output
(``y_i = s_i * sum_k t_ik x_k``) and the same one ``lane_planes`` builds for
this plane on the kernel lane, so the three agree on what the wire MEANS
whatever runs it.

WHAT IT REUSES.  Blob parsing is Tessera's reader (``tessera.unit_artifact``,
``tessera.fused``), the packing is the wire's own
(``tessera.lane_planes.pack_window_planes``, through ``window``) and the
reference decode is ``tessera.decode.materialize_bf16``.  At preparation every
role is decoded through the in-forward decoder AND through the reference
decoder and the two are compared element for element; a disagreement refuses
the module.  Both residency modes then hold values the reference produced.
No CUDA extension: the window decoder is pure torch.

RESIDENCY.  ``resident`` decodes once at load and holds the bf16 tile plus one
fp32 per row.  **As a size claim that is nothing at all** -- 16 bits a weight
is the source precision -- and it is not offered as one: it is the correctness
path, the tile a stock GEMM consumes with no decoder in the serve.
``streamed`` holds the packed window planes at the artifact's own 4-8 bpp and
decodes each forward into a transient tile the op owns.  That is the mode the
family is a product in.

WHAT IS NOT ATTESTED.  No ``lane_eligibility`` cell in this package's
``runtime_contract.json`` names ``TESSERA_BF16``, because no container receipt
covers it yet: there is no served census and no served KL against the twin the
exporter writes.  Absence resolves ``unattested``, which is the honest status
and not a refusal (principle 14).  A cell is added when a receipt exists, not
when this module does.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from .lane import MODE_RESIDENT, MODE_STREAMED, MODES
from .scheme import ROUTES, TESSERA_BF16, parse_tessera_blob_for_scheme, validate_tessera_scheme
from .sharding import plan_shard, require_axis_supported, shard_parsed_roles
from .telemetry import DECODER_TORCH_WINDOW, emit_route, route_shape
from .window import PreparedWindow, prepare_window

__all__ = [
    "ACTIVATION_CONTRACT",
    "PreparedTesseraBf16Module",
    "prepare_tessera_bf16_module",
    "build_tessera_bf16_method",
]

ACTIVATION_CONTRACT = ROUTES[TESSERA_BF16]["activation_contract"]
_GRID = "BF16"


class _Bf16Role:
    __slots__ = ("name", "row_offset", "rows", "window")

    def __init__(self, name: str, row_offset: int, rows: int, window: PreparedWindow):
        self.name = str(name)
        self.row_offset = int(row_offset)
        self.rows = int(rows)
        self.window = window


class PreparedTesseraBf16Module:
    """Private, once-prepared device owner for one vLLM module's BF16 roles."""

    __slots__ = ("__roles", "__rows", "__columns", "__scale", "__device")

    def __init__(self, roles: Sequence[_Bf16Role], *, rows: int, columns: int,
                 scale: torch.Tensor, device: torch.device):
        self.__roles = tuple(roles)
        self.__rows = int(rows)
        self.__columns = int(columns)
        self.__scale = scale
        self.__device = device
        if sum(r.rows for r in self.__roles) != self.__rows:
            raise ValueError("prepared roles do not stack to the module's rows")
        if any(r.window.cols != self.__columns for r in self.__roles):
            raise ValueError("every role of a module shares its input width")
        if tuple(scale.shape) != (self.__rows,) or scale.dtype != torch.float32:
            raise ValueError("the row scale is one fp32 per module row")

    @property
    def rows(self): return self.__rows
    @property
    def columns(self): return self.__columns
    @property
    def device(self): return self.__device
    @property
    def decoder(self): return DECODER_TORCH_WINDOW
    @property
    def role_names(self): return tuple(r.name for r in self.__roles)

    def row_scale(self) -> torch.Tensor:
        """A copy of the per-row fp32 scale (the reader's expression), ``[rows]``."""
        return self.__scale.clone()

    def wire_bytes_resident(self) -> int:
        """Device bytes the prepared planes occupy (the streamed footprint's wire half)."""
        return sum(r.window.resident_bytes() for r in self.__roles)

    def decode(self) -> torch.Tensor:
        """A fresh ``bfloat16 [rows, columns]`` of table VALUES: the forward's entry.

        The row scale is deliberately absent -- see the module docstring.  What
        this returns is the tile the GEMM multiplies, not the weight the
        encoder scored; ``value * scale[:, None]`` is that, and nothing in the
        serve ever forms it.
        """
        if len(self.__roles) == 1:
            return self.__roles[0].window.decode()
        return torch.cat([r.window.decode() for r in self.__roles], 0)


def prepare_tessera_bf16_module(parsed_roles, device=None) -> PreparedTesseraBf16Module:
    """``[(role, ParsedUnit)]`` in stacking order -> a prepared module.

    Every role must be the scalar BF16 grid, window body, CHANNEL plane.  Each
    is packed for the in-forward decoder and decoded once through it and once
    through ``tessera.decode.materialize_bf16``; the two must agree element for
    element or the module is refused.  The per-row scale is the reference
    decoder's (``scale_rows * global`` in fp32), and it stays out of the tile.
    """
    from tessera.decode import materialize_bf16
    from tessera.manifest import ScalePlaneKind

    device = torch.device("cuda" if device is None else device)
    if not parsed_roles:
        raise ValueError("a Tessera module needs at least one role")
    roles = []
    scales = []
    offset = 0
    columns = None
    for name, parsed in parsed_roles:
        unit, grid = parsed.unit, parsed.grid
        if grid.name != _GRID or grid.arity != 1:
            raise ValueError(
                f"role {name!r}: the 16-bit route decodes the scalar BF16 grid, not {grid.name}")
        if parsed.body.name != "WINDOW":
            raise ValueError(
                f"role {name!r}: the 16-bit route decodes the window body (Tessera's BF16 "
                f"recipe at every rung); this unit carries {parsed.body.name}, which has no "
                "in-forward decoder here")
        if getattr(unit, "scale_plane", None) is not ScalePlaneKind.CHANNEL:
            raise ValueError(
                f"role {name!r}: the row scale this route applies on the GEMM output is the "
                f"CHANNEL plane's; this unit carries "
                f"{getattr(getattr(unit, 'scale_plane', None), 'name', None)}")
        steps, cols = unit.body_bits.shape
        if columns is None:
            columns = int(cols)
        elif int(cols) != columns:
            raise ValueError(f"role {name!r} has {cols} input columns, the module {columns}")
        # The table holds VALUES, not codes, so no ``code_map``: ``window``
        # keeps a floating table at its own dtype and the gather yields the
        # bf16 tile directly.  ``window_table_values`` is the definition
        # (a gather through ``grid_vector_table``); the equality with the
        # ALPHABET plane's own uint16 words viewed as bf16 is what
        # ``test_bf16_route`` pins, and it is what entitles a kernel to take
        # the view instead.
        table = _window_table_values(parsed)
        # A ROW shard's first surviving step does not start from the pinned
        # zero register, and the window body's L-bit pad IS that start state
        # (``lane_planes.pack_window_planes``).  A whole unit carries None and
        # takes exactly the path it always did.  The equality check below is
        # against ``materialize_bf16``, which reads the same field, so a
        # threading error cannot pass here.
        window = prepare_window(unit.body_bits, unit.rates, unit.window_bits, table,
                                device, initial_state=getattr(unit, "initial_state", None))
        reference, scale = materialize_bf16(unit, parsed.forests, parsed.code)
        reference = reference.to(device)
        decoded = window.decode()
        if decoded.dtype != torch.bfloat16:
            raise RuntimeError(
                f"role {name!r}: the packed-window decoder produced {decoded.dtype}, not "
                "bfloat16; a BF16 unit's table holds values and must survive the decode")
        if not torch.equal(decoded, reference):
            wrong = int((decoded != reference).sum())
            raise RuntimeError(
                f"role {name!r}: the packed-window decoder disagrees with tessera.decode."
                f"materialize_bf16 on {wrong} of {reference.numel()} values; refusing to serve "
                "a tile the reference decoder would not produce")
        scales.append(scale.to(device, torch.float32).reshape(-1))
        roles.append(_Bf16Role(name, offset, steps, window))
        offset += int(steps)
    return PreparedTesseraBf16Module(roles, rows=offset, columns=columns,
                                     scale=torch.cat(scales).contiguous(), device=device)


def _window_table_values(parsed) -> torch.Tensor:
    """The unit's ``2^L`` table as bf16 values, off Tessera's own definition."""
    from tessera.bf16_route import window_table_values

    if parsed.unit.window_codes is None:
        raise ValueError("a window body needs the unit's table")
    return window_table_values(parsed.unit.window_codes, parsed.grid)


def build_tessera_bf16_method(scheme, prefix: str, mode: str):
    """Construct the vLLM linear method serving a Tessera BF16 module.

    Reached through ``lane.build_tessera_method``, which owns the residency
    mode; this builder takes the resolved ``mode`` and validates the scheme so
    an unserveable geometry is refused at method construction.
    """
    resolved = mode
    if resolved not in MODES:
        raise ValueError(f"unknown residency mode {resolved!r}")
    declared = validate_tessera_scheme(scheme, prefix)
    if declared["family"] != TESSERA_BF16:
        raise ValueError(
            f"{prefix}: the 16-bit route serves {TESSERA_BF16}, not {declared['family']}")
    rows, columns, wire_bytes = declared["rows"], declared["columns"], declared["wire_bytes"]

    from vllm.model_executor.layers.linear import LinearMethodBase
    from vllm.model_executor.parameter import BasevLLMParameter

    class TesseraBf16LinearMethod(LinearMethodBase):
        """W16A16 Tessera linear (the 16-bit route)."""

        def __init__(self, mode: str) -> None:
            self._mode = mode

        # -- load -------------------------------------------------------
        def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                           input_size, output_size, params_dtype, **extra_weight_attrs):
            out_size = int(sum(output_partition_sizes))
            in_size = int(input_size_per_partition)
            # See ``sharding``: the plan is the whole module at TP=1 and is the
            # shape check it replaces; at TP>1 it names the axis to cut on.
            plan = plan_shard(prefix, rows=rows, columns=columns,
                              out_size=out_size, in_size=in_size)
            # And the per-axis answer, asked here because here is where the
            # axis is known.  This route cuts both, for the reason
            # ``ROUTE_TP_AXES`` records; the call is what keeps the published
            # ``loader_axes`` a statement about the loader rather than about
            # the table.
            require_axis_supported(TESSERA_BF16, plan)
            weight_loader = extra_weight_attrs.get("weight_loader")
            # The whole container as one opaque blob: a blob has no output axis
            # to split.  No activation scale of any kind: the A side is bf16
            # and unquantised, which is the whole of this route's A contract.
            layer.register_parameter("wire_bytes", BasevLLMParameter(
                data=torch.empty(wire_bytes, dtype=torch.uint8), weight_loader=weight_loader))
            layer.tessera_shard_plan = plan
            layer.tessera_rows = plan.shard_rows
            layer.tessera_columns = plan.shard_columns
            layer.tessera_mode = self._mode
            layer.tessera_family = TESSERA_BF16
            layer.tessera_activation_contract = ACTIVATION_CONTRACT

        def process_weights_after_loading(self, layer) -> None:
            """Parse the container, prepare and verify every role, decode or keep."""
            blob = layer.wire_bytes.data
            if blob.device.type != "cpu":
                blob = blob.cpu()
            roles = parse_tessera_blob_for_scheme(blob.contiguous().numpy().tobytes(), scheme, prefix)
            # This rank's slice of every role; identity at TP=1.
            roles = shard_parsed_roles(roles, layer.tessera_shard_plan)
            device = layer.wire_bytes.device
            if device.type != "cuda":
                device = torch.device("cuda")
            prepared = prepare_tessera_bf16_module(roles, device=device)
            layer.tessera_prepared = prepared
            layer.tessera_decoder = prepared.decoder
            layer.tessera_roles = prepared.role_names
            # The epilogue factor, kept as ``[rows]`` fp32 and broadcast over
            # the GEMM's fp32 output rows.  NOT folded into the tile, and not
            # narrowed: see the module docstring.
            layer.register_buffer("row_scale", prepared.row_scale().contiguous(),
                                  persistent=False)
            del layer.wire_bytes
            if self._mode == MODE_RESIDENT:
                layer.register_buffer("weight_bf16", prepared.decode(), persistent=False)
                layer.tessera_prepared = None
            # streamed: the prepared planes stay; the tile is decoded per forward

        # -- forward ----------------------------------------------------
        def apply(self, layer, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
            orig = x.shape
            x2 = x.reshape(-1, orig[-1])
            if x2.dtype != torch.bfloat16:
                x2 = x2.to(torch.bfloat16)
            if layer.tessera_mode == MODE_RESIDENT:
                b = layer.weight_bf16
            else:
                b = layer.tessera_prepared.decode()
            # ``out_dtype=torch.float32`` hands out the bf16 mainloop's own
            # accumulator instead of a bf16 truncation of it, so the row scale
            # multiplies the sum and the result is rounded once.  Rounding to
            # bf16 first and scaling after would put a second rounding between
            # the GEMM and the answer -- which is most of what not folding the
            # scale into the tile was bought to avoid.
            y = torch.mm(x2.contiguous(), b.t(), out_dtype=torch.float32)
            y = (y * layer.row_scale).to(torch.bfloat16)
            try:
                emit_route(
                    layer, kind="dense", policy=f"{TESSERA_BF16}:{layer.tessera_mode}",
                    symbol="torch.mm", tile_m=0,
                    shape=route_shape(x2, layer.tessera_rows, layer.tessera_columns),
                    contract=layer.tessera_activation_contract, state="served", reason=None,
                    decoder=layer.tessera_decoder,
                )
            except Exception:  # noqa: BLE001 -- telemetry never breaks a request
                pass
            if bias is not None:
                y = y + bias
            return y.reshape(*orig[:-1], layer.tessera_rows)

    return TesseraBf16LinearMethod(resolved)
