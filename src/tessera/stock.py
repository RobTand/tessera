"""A Tessera unit as the stock ``compressed-tensors`` tensors vanilla vLLM serves.

The wire's 4.0 bits per weight exist only on the kernel lane.  What a stock
runtime serves is the **materialised** unit: every Tessera code on an E2M1
grid is a legal NVFP4 nibble and every block-plane scale is one E4M3 byte
behind a power-of-two global, so a K1 or K2 unit over an S6b or LUT plane
*is* an NVFP4 tensor at 4.5 bits resident; an E4M3 unit over the CHANNEL
plane *is* a per-channel FP8 tensor at 8 bits resident.  Neither
materialisation rounds -- the served bytes decode to the reader's
reconstruction bit for bit -- and ``stock_dequant`` states that decode in
the runtime's own arithmetic (vLLM's E2M1 table, its ``1 / divisor`` global,
its per-row FP8 scale) so the identity is checked against what the kernel
computes and not against Tessera's own reader twice.

Two honesties travel with the tensors.  The resident bytes are the stock
format's (``stock_bytes``: 4.5 or 8.0 bpp), never the wire's; an exporter
records both and a card that quotes the wire's rate for a stock checkpoint is
lying.  And a fused group -- q/k/v, gate/up -- is one vLLM Linear carrying
one ``weight_global_scale``: vLLM takes the largest divisor across the shards,
warns, and dequantises every shard with it, so shards that were encoded under
different globals would decode to the wrong weights silently.  ``share_global``
rewrites a group onto one power of two by an integer binade shift of the E4M3
scale bytes, exactly or not at all.
"""

from __future__ import annotations

import math

import torch

from .alphabet import E2M1_GRID, PayloadGrid
from .decode import _grid_and_forests, decode_codes_mixed, materialize_fp8
from .errors import GrammarError
from .manifest import ScalePlaneKind
from .wire import nvfp4_scale_bytes, nvfp4_scale_bytes_lut

__all__ = [
    "E2M1_MAGNITUDES",
    "e2m1_nibbles",
    "materialize_stock",
    "share_global",
    "stock_bytes",
    "stock_dequant",
    "stock_kind",
]

#: vLLM's ``kE2M1ToFloat`` table: nibble bit 3 is the sign, bits 2..0 index
#: this.  Stated here, not imported from the alphabet, because ``stock_dequant``
#: is a statement of what the runtime does; the module-level check below is
#: where the two are proven the same table.
E2M1_MAGNITUDES: tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: E4M3FN's largest finite value; a scale byte shifted past it is NaN.
E4M3_MAX = 448.0

NVFP4_KEYS = ("weight_packed", "weight_scale", "weight_global_scale")
FP8_KEYS = ("weight", "weight_scale")


def _nvfp4_values(nibbles: torch.Tensor) -> torch.Tensor:
    """E2M1 nibbles -> values, exactly as ``nvfp4_emulation_utils`` decodes them."""
    table = torch.tensor(E2M1_MAGNITUDES, dtype=torch.float32, device=nibbles.device)
    codes = nibbles.long()
    return table[codes & 7] * torch.where(codes >> 3 == 1, -1.0, 1.0)


_alphabet_values = tuple(E2M1_GRID.values)
_runtime_values = tuple(float(v) for v in _nvfp4_values(torch.arange(16)))
if _alphabet_values != _runtime_values:   # pragma: no cover - a grid change, not a runtime one
    raise GrammarError(
        "the alphabet's E2M1 code order is not vLLM's: a Tessera nibble would "
        f"decode to a different value on the served kernel ({_alphabet_values} "
        f"vs {_runtime_values})"
    )


def e2m1_nibbles(codes: torch.Tensor, grid: PayloadGrid) -> torch.Tensor:
    """Per-position E2M1 nibbles ``[rows, cols]`` from per-code indices ``[steps, cols]``.

    A k-tuple code ``i`` reconstructs base codes ``i // G^(k-1), ..., i % G``
    onto ``k`` consecutive rows (``tuple_grid``); at arity 1 the code is the
    nibble.  The layout is ``dequantize``'s, which is the layout the reader's
    reconstruction has, so a nibble lands on the row its weight occupies.
    """
    if grid.arity == 1:
        if grid.size != 16:
            raise GrammarError(f"{grid.name} is not an E2M1 grid; its codes are not nibbles")
        return codes.to(torch.uint8)
    base = round(grid.size ** (1.0 / grid.arity))
    if base**grid.arity != grid.size or base != 16:
        raise GrammarError(
            f"{grid.name} ({grid.size} codes at arity {grid.arity}) is not a tuple of E2M1"
        )
    steps, cols = codes.shape
    rest = codes.long()
    digits = []
    for _ in range(grid.arity):
        digits.append(rest % base)
        rest = rest // base
    digits.reverse()                                   # slowest digit first: row 0 of the tuple
    return torch.stack(digits, dim=1).reshape(steps * grid.arity, cols).to(torch.uint8)


def materialize_stock(unit, forest, code) -> dict[str, torch.Tensor]:
    """The compressed-tensors tensors for one unit, by its grid and plane.

    * E2M1 / E2M1x2 over an S6b or LUT plane -> the NVFP4 triple:
      ``weight_packed`` uint8 ``[rows, cols/2]`` (low nibble = even column),
      ``weight_scale`` float8_e4m3fn ``[rows, cols/16]``, and
      ``weight_global_scale`` fp32 ``[1]`` holding the **divisor** ``1/global``
      (the convention vLLM inverts on load).
    * E4M3 over the CHANNEL plane -> the per-channel FP8 pair: ``weight``
      float8_e4m3fn ``[rows, cols]`` and ``weight_scale`` fp32 ``[rows, 1]``.
    * BF16 over the CHANNEL plane -> ``weight`` bfloat16 ``[rows, cols]`` and
      nothing else.  A checkpoint ships one tensor and no scale, so this is
      the one rendering in the 16-bit route that **folds** the row scale into
      the value (``materialize_bf16_folded``) -- and the fold costs a
      rate-independent ~0.0015 of relative output error that a route holding
      the wire does not pay (``decode.materialize_bf16``).  It buys the thing
      only a stock tensor can buy: a checkpoint with no quantization config
      at all, servable by a runtime that has never heard of Tessera *or* of
      compressed-tensors.  Read it as the twin's price, not the format's.

    Every other combination is refused: it has no stock tensor, and the
    kernel lane is where it serves.
    """
    grid, _forests = _grid_and_forests(forest)
    plane = getattr(unit, "scale_plane", ScalePlaneKind.S6B)
    if plane is ScalePlaneKind.CHANNEL and grid.name == "BF16":
        from .decode import materialize_bf16_folded

        return {"weight": materialize_bf16_folded(unit, forest, code).contiguous()}
    if plane is ScalePlaneKind.CHANNEL:
        native, scale = materialize_fp8(unit, forest, code)
        return {
            "weight": native.contiguous().view(torch.float8_e4m3fn),
            "weight_scale": scale.reshape(-1, 1).to(torch.float32).contiguous(),
        }
    codes = decode_codes_mixed(unit, forest, code)
    nibbles = e2m1_nibbles(codes, grid)
    rows, cols = nibbles.shape
    if cols % unit.half or cols % 2:
        raise GrammarError(f"{cols} columns do not pack to nibble pairs and per-{unit.half} scales")
    packed = (nibbles[:, 0::2] & 0xF) | ((nibbles[:, 1::2] & 0xF) << 4)
    if plane is ScalePlaneKind.LUT:
        if unit.scale_lut is None:
            raise GrammarError("a LUT scale plane needs the unit's table")
        e4m3, global_scale = nvfp4_scale_bytes_lut(unit.scale_refine, unit.scale_lut, unit.scale_global)
    else:
        e4m3, global_scale = nvfp4_scale_bytes(unit.scale_base, unit.scale_refine, unit.group, unit.half)
    _po2(global_scale, "the unit's global scale")
    return {
        "weight_packed": packed.contiguous(),
        "weight_scale": e4m3.reshape(rows, cols // unit.half).contiguous().view(torch.float8_e4m3fn),
        "weight_global_scale": torch.tensor([1.0 / global_scale], dtype=torch.float32),
    }


def stock_kind(tensors: dict[str, torch.Tensor]) -> str:
    """``"nvfp4"`` or ``"fp8"`` by the tensors' names, refusing anything else."""
    keys = set(tensors)
    if keys == set(NVFP4_KEYS):
        return "nvfp4"
    if keys == set(FP8_KEYS):
        return "fp8"
    raise GrammarError(f"not a stock materialisation: {sorted(keys)}")


def stock_dequant(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    """The fp32 weights the runtime computes from the stock tensors.

    NVFP4: ``e2m1[nibble] * fp8(scale) * (1 / weight_global_scale)`` -- vLLM
    stores the global as a divisor, inverts it on load and folds it into the
    kernel's alpha.  FP8 per-channel: ``fp8(weight) * weight_scale[row]``.
    Every factor here has at most eleven significant bits and the globals are
    powers of two, so the products are exact in fp32 in any order, which is
    what lets the identity test ask for ``torch.equal``.
    """
    kind = stock_kind(tensors)
    if kind == "fp8":
        return tensors["weight"].float() * tensors["weight_scale"].float().reshape(-1, 1)
    packed = tensors["weight_packed"]
    rows, half_cols = packed.shape
    nibbles = torch.empty(rows, half_cols * 2, dtype=torch.uint8, device=packed.device)
    nibbles[:, 0::2] = packed & 0xF
    nibbles[:, 1::2] = packed >> 4
    values = _nvfp4_values(nibbles)
    scale = tensors["weight_scale"].float()
    if scale.shape[0] != rows or (half_cols * 2) % scale.shape[1]:
        raise GrammarError(f"scale {tuple(scale.shape)} does not tile weights {rows}x{half_cols * 2}")
    group = (half_cols * 2) // scale.shape[1]
    field = torch.repeat_interleave(scale, group, dim=1)
    divisor = float(tensors["weight_global_scale"].reshape(-1)[0])
    return values * field * (1.0 / divisor)


def stock_bytes(tensors: dict[str, torch.Tensor]) -> int:
    """Resident bytes: what the checkpoint carries and the runtime holds."""
    return sum(int(t.numel()) * t.element_size() for t in tensors.values())


def _po2(value: float, what: str) -> int:
    if not (value > 0.0) or not math.isfinite(value):
        raise GrammarError(f"{what} is {value!r}, not a positive power of two")
    exponent = math.log2(value)
    if exponent != int(exponent):
        raise GrammarError(f"{what} is {value!r}, not a power of two")
    return int(exponent)


def share_global(
    group: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict[str, dict[str, torch.Tensor]], float]:
    """A fused group's NVFP4 members rewritten onto one ``weight_global_scale``.

    Candidates are tried largest divisor first -- the choice vLLM itself makes
    when shards disagree -- so members shift *up* the E4M3 range before any
    shifts down.  A member moves by ``ratio = shared / own``, a power of two:
    its scale bytes are exact after the move when every entry re-snaps to
    itself as float8_e4m3fn and none became NaN (past 448).  A group no
    candidate carries exactly is refused with the members named.  Returns the
    rewritten group and the shared divisor; a group already on one divisor
    comes back untouched.
    """
    if not group:
        raise GrammarError("share_global needs at least one member")
    divisors = {}
    for name, tensors in group.items():
        if stock_kind(tensors) != "nvfp4":
            raise GrammarError(f"{name}: only NVFP4 members carry a global scale to share")
        divisors[name] = float(tensors["weight_global_scale"].reshape(-1)[0])
        _po2(divisors[name], f"{name}'s weight_global_scale")
    distinct = sorted(set(divisors.values()), reverse=True)
    if len(distinct) == 1:
        return group, distinct[0]
    lo, hi = _po2(distinct[-1], "divisor"), _po2(distinct[0], "divisor")
    candidates = [float(2.0**e) for e in range(hi, lo - 1, -1)]
    failures = {}
    for shared in candidates:
        rewritten = {}
        for name, tensors in group.items():
            ratio = shared / divisors[name]
            scale = tensors["weight_scale"].float()
            moved = scale * ratio
            snapped = moved.to(torch.float8_e4m3fn)
            back = snapped.float()
            if not bool(torch.isfinite(back).all()) or not torch.equal(back, moved):
                failures[shared] = name
                break
            rewritten[name] = {
                **tensors,
                "weight_scale": snapped.contiguous(),
                "weight_global_scale": torch.tensor([shared], dtype=torch.float32),
            }
        else:
            return rewritten, shared
    raise GrammarError(
        "no single weight_global_scale carries this fused group exactly: "
        + ", ".join(f"{name} 1/{d:g}" for name, d in divisors.items())
        + "; first entry to leave E4M3 per candidate: "
        + ", ".join(f"1/{d:g} -> {name}" for d, name in failures.items())
        + ". Encode the group under one pinned global instead of shifting it."
    )


# --- what the checkpoint DECLARES, and what the runtime does with it ----------
#
# A compressed-tensors ``quantization_config`` carries a top-level ``format``
# beside the per-group ones.  Ours was the constant ``"mixed-precision"``
# whatever the artifact turned out to be, and the stock NVFP4 comparator twin is
# not mixed: it is one group, ``nvfp4-pack-quantized``, over every target.  The
# constant is not cosmetic.  In the pinned runtime the field is read twice, and
# the two readings differ in kind:
#
#   1. As a per-group DEFAULT.  ``CompressedTensorsConfig.from_config`` keeps it
#      as ``self.quant_format``
#      (``model_executor/layers/quantization/compressed_tensors/
#      compressed_tensors.py:250``), and ``get_scheme_dict`` substitutes it only
#      for a group that declared none (``:963-964``:
#      ``if scheme_dict.get("format") is None: scheme_dict["format"] =
#      self.quant_format``).  Every group both exporters write declares its own
#      ``format``, so the substitution never fires and the top-level string does
#      not choose a scheme or change a byte in memory.
#
#   2. As the FP4-model predicate, where it is the whole answer.
#      ``ModelConfig.is_nvfp4_quantized`` (``config/model.py:2096-2108``) is a
#      substring test on this exact field, and ``"mixed-precision"`` does not
#      contain ``"nvfp4"``.  A uniform-NVFP4 checkpoint from anyone else
#      answers True and ours answered False, so the two are not the same
#      compiled graph and no receipt of ours said so.
#
# ``declared_format`` derives the field from the groups that were actually
# written; ``vllm_fp4_predicate`` states the consequence, with the attestation
# that makes it a reading of a runtime rather than a claim about one (AGENTS
# principle 14: vLLM publishes no machine-readable table for this predicate, so
# the recorded value is honest only with the image and version it was read in).

#: Per-group ``format`` strings the two exporters write.
NVFP4_PACK_QUANTIZED = "nvfp4-pack-quantized"
FLOAT_QUANTIZED = "float-quantized"
#: The top-level label for an artifact whose groups do not agree on one format.
MIXED_PRECISION = "mixed-precision"

#: Where the predicate below was read.  Quoted, not paraphrased, because the
#: value it resolves to is stamped onto artifacts: an image that moves the check
#: makes a stamped record stale, and the version is what lets someone see that.
VLLM_FP4_PREDICATE_ATTESTATION = {
    "image": "vllm/vllm-openai:latest",
    # The tag floats and the two boxes do not hold the same bytes under it
    # (tessera#100), so the id is part of the reading, not decoration.
    "image_id": "sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14",
    "box": "sparky",
    "version": "0.28.0",
    "read": "2026-09-03",
    "predicate": {
        "path": "vllm/config/model.py",
        "lines": "2096-2108",
        "source": (
            'self.quantization == "compressed-tensors" and quant_config is not None '
            'and "nvfp4" in quant_config.get("format", "").lower()'
        ),
    },
    "consumer": {
        "path": "vllm/config/vllm.py",
        "lines": "134-144",
        "source": (
            'is_custom_op_enabled("silu_and_mul") or is_custom_op_enabled("quant_fp8") '
            "or model_config.is_nvfp4_quantized()"
        ),
        "note": (
            "the predicate's ONE consumer in 0.28.0, the pass-config entry "
            "fuse_act_quant at O1/O2/O3 (default optimization_level is O2, "
            "config/vllm.py:401). fuse_attn_quant beside it is the hard constant "
            "IS_QUANTIZED = False (config/vllm.py:113-120, pending vllm#25689), so "
            "no attention+quant fusion is at stake either way."
        ),
    },
    "scope": (
        "a COMPILED serve only.  --enforce-eager sets compilation mode NONE "
        "(config/vllm.py:1284-1290), so no fusion pass runs on either arm; the KL "
        "harnesses serve eager by default (serve_and_dump_kl.sh, "
        "tessera_plugin_served.sh) and this predicate cannot have moved a number "
        "taken through them.  The default serve is where it bites: "
        "optimization_level defaults to O2 (config/vllm.py:401), whose pass config "
        "sets fuse_act_quant = enable_act_fusion, and under the default compiled "
        "backend custom_ops resolves to ['none'] (config/vllm.py:1392-1399) so "
        "neither silu_and_mul nor quant_fp8 is enabled -- +quant_fp8 is appended "
        "only for blocked weights -- BOTH of its two append sites "
        "(config/vllm.py:1368-1375 and 1803-1810) are guarded by the same "
        "has_blocked_weights() call, which tests for "
        "QuantizationStrategy.BLOCK (compressed_tensors.py:969-977), and the groups "
        "these exporters write declare strategy 'tensor_group' (NVFP4) and 'channel' "
        "(FP8), neither of which is BLOCK.  There is no default append of "
        "+silu_and_mul anywhere: grep over config/*.py and platforms/*.py finds the "
        "name only at the is_custom_op_enabled read itself (config/vllm.py:141).  "
        "So on a default compiled serve this "
        "predicate is the ONLY thing switching "
        "fuse_act_quant, and it switched it off for us and on for a uniform-NVFP4 "
        "checkpoint from anyone else."
    ),
    "nvfp4_pattern_built": (
        "yes.  SiluMulNvfp4QuantPattern is registered only when "
        "silu_and_mul_nvfp4_quant_supported "
        "(compilation/passes/fusion/act_quant_fusion.py:36-40,298-299), i.e. when the "
        "op is in this build's torch.ops._C, and it is: the schema "
        "'silu_and_mul_nvfp4_quant(Tensor! result, Tensor! result_block_scale, "
        "Tensor input, Tensor input_global_scale) -> ()' is in "
        "vllm/_C_stable_libtorch.abi3.so beside the cutlass_scaled_fp4_mm schema "
        "that serves NVFP4 on this image.  First read out of the binary, because "
        "the extension needs libcuda and returns a uniform False without a GPU -- "
        "hasattr(torch.ops._C, 'cutlass_scaled_fp4_mm') is False too on a CPU "
        "container of this image, which demonstrably serves NVFP4 with it, so that "
        "reading cannot tell a missing op from a missing driver.  Then confirmed on "
        "the serving hardware: with --gpus all and vllm._custom_ops imported, "
        "torch.ops._C carries 20 names, silu_and_mul_nvfp4_quant / "
        "cutlass_scaled_fp4_mm / silu_and_mul are all present, a nonsense control "
        "name is absent, and silu_and_mul_nvfp4_quant_supported is True.  "
        "The per-SM guard is answered too, by calling the op rather than reading "
        "it: on sm121 (compute capability 12.1) "
        "silu_and_mul_nvfp4_quant(out, block_scale, x, global_scale) returns "
        "without raising and writes nonzero packed output, so the "
        "'No compiled silu_and_mul nvfp4 quantization kernel for SM ' TORCH_CHECK "
        "in the binary does not fire on this target.  The pattern registers and "
        "its kernel runs here."
    ),
    "gpu_reading": {
        "box": "sparky",
        "compute_capability": "12.1",
        "read": "2026-09-03",
        "torch_ops_C_names": 20,
        "hasattr": {
            "silu_and_mul_nvfp4_quant": True,
            "cutlass_scaled_fp4_mm": True,
            "silu_and_mul": True,
            "no_such_op_xyzzy": False,
        },
        "silu_and_mul_nvfp4_quant_supported": True,
        "op_call_on_sm121": "ok, nonzero output",
    },
}


def declared_format(config_groups) -> str:
    """The top-level ``format`` for a config whose groups are ``config_groups``.

    Groups that agree on one format make an artifact OF that format and it says
    so; groups that disagree make a mixed artifact and it says that.  Keying on
    the distinct formats rather than on the number of groups is the honest
    reading of both consumers above: vLLM's substring predicate asks what the
    weights are, not how many ``config_groups`` keys the exporter chose to use.

    An empty mapping has no format to declare and is not a quantized checkpoint;
    the caller writes no ``quantization_config`` at all rather than one that
    tells a runtime to look for compressed tensors that do not exist.
    """
    if not config_groups:
        raise ValueError(
            "no config groups: a checkpoint with nothing quantized declares no "
            "quantization_config, not a format")
    formats = {group["format"] for group in config_groups.values()}
    return formats.pop() if len(formats) == 1 else MIXED_PRECISION


def vllm_fp4_predicate(quant_method: str, declared: str) -> dict:
    """``ModelConfig.is_nvfp4_quantized`` for this declaration, and why.

    Recorded on the artifact so a comparison against someone else's uniform
    NVFP4 checkpoint can see whether the two arms compile the same graph,
    instead of the answer being an unrecorded consequence of a constant.
    """
    compressed = quant_method == "compressed-tensors"
    nvfp4 = "nvfp4" in declared.lower()
    if not compressed:
        reason = (
            f"quant_method is {quant_method!r}, not 'compressed-tensors': the "
            "predicate's first conjunct fails and the top-level format is not read "
            "by it at all")
    elif nvfp4:
        reason = f"format {declared!r} contains 'nvfp4'"
    else:
        reason = f"format {declared!r} does not contain 'nvfp4'"
    return {
        "quant_method": quant_method,
        "format": declared,
        "vllm_is_nvfp4_quantized": compressed and nvfp4,
        "reason": reason,
        "consequence": (
            "one of the three disjuncts of enable_act_fusion, i.e. of the "
            "fuse_act_quant pass-config entry, on a compiled serve"),
        "attested": VLLM_FP4_PREDICATE_ATTESTATION,
    }
