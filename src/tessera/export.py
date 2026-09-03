"""Write a Tessera checkpoint, and read one back.

``build_unit_artifact`` already turns one encoded Linear into bytes that
``read_unit_artifact`` inverts exactly.  This module is the model-level walk
around that seam: encode the Linears a plan names, copy everything else
verbatim, and record the identity a reader needs to accept the result.

**Rungs are addressed by ``(grid, q256)``, not by name.**  A rung *name* like
``TESSERA_E2M1_K2_R896`` is PrismaQuant's label for an allocator candidate; the
thing the wire commits to is ``encoder_profile_id``, which hashes the code, the
forest construction, the rate set and the grid digest.  Keeping the parser on
the producer side and the identity on the wire means a mislabelled artifact is
still *unambiguous* -- the reader rebuilds the grid from the profile id and
refuses anything it cannot reproduce.  Two spellings of one spec is the failure
that identity discipline exists to prevent, so there is deliberately only one
place the grid is decided.

**Rendering identity is asserted, not assumed** (principle 8).  Every unit is
read back off its own bytes and compared to the encoder's reconstruction before
it is written.  The surrogate that priced the Linear, the KL that validated it,
and the bytes that ship are then the same tensor by construction rather than by
three code paths agreeing.

**The artifact declares itself unbacked** (principle 9).  No serving runtime
decodes this container today.  ``route_status`` says so in a field a gate can
read, so nothing downstream can mistake "exportable" for "servable".
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import torch

from .alphabet import GAUSSIAN_SOURCE, PayloadGrid, build_forest, grid_digest
from .decode import reconstruct_unit
from .encode import EncodedUnit, encode_unit
from .errors import GrammarError
from .grammar import bresenham_rate_schedule
from .manifest import BodyKind, RotationState, ScalePlaneKind
from .trellis import ConvCode
from .unit_artifact import build_unit_artifact, read_unit_artifact

__all__ = [
    "CONTAINER_VERSION",
    "DEFAULT_SPAN",
    "DEFAULT_SCALE_PLANE",
    "ExportReport",
    "ExportedUnit",
    "WireRecipe",
    "wire_recipe",
    "DEFAULT_LDLQ_SIGMA",
    "DEFAULT_LDLQ_BLOCK",
    "DEFAULT_REFIT_OBJECTIVE",
    "HESSIAN_IDENTITY",
    "ActivationSource",
    "E4M3_RECIPE",
    "BF16_RECIPE",
    "E2M1X2_SUBCAP_RECIPE",
    "E4M3_WINDOW_BITS",
    "BF16_WINDOW_BITS",
    "BF16_CHANNEL_SIGMA",
    "E2M1X2_SUBCAP_WINDOW_BITS",
    "tcq_cap_q256",
    "RecipeRange",
    "recipe_table",
    "recipe_at",
    "rung_ceiling",
    "PER_RUNG",
    "plan_for",
    "encode_linear",
    "encode_settings_from_config",
    "export_checkpoint",
    "export_checkpoint_streaming",
    "grid_from_config",
    "load_tessera_weight",
    "read_checkpoint_config",
]

#: Bumped when the on-disk *container* changes shape.  The per-unit wire has
#: its own identity (``encoder_profile_id``); this versions the checkpoint
#: layout around it -- the tensor suffix, the config schema, the plan encoding.
CONTAINER_VERSION = 1

#: The suffix a quantized Linear's bytes are stored under.  A reader that does
#: not know Tessera sees an opaque uint8 tensor under a name that is not the
#: original, so it cannot silently load the blob *as* a weight.
BLOB_SUFFIX = ".tessera"

DEFAULT_CODE = ConvCode(memory=6)
DEFAULT_GROUP = 32
DEFAULT_HALF = 16
#: Scale-plane refits per unit (``encode_unit``).  An encoder setting, not
#: wire: the bytes decode identically at any value.  Recorded in the config so
#: a merge can refuse parts built at different settings.
DEFAULT_SCALE_REFIT = 4
#: Branch-metric weighting for the Viterbi (``encode_unit``).  ``"scale"``
#: weights every position by its half's scale squared so the path minimises
#: the true squared error rather than the per-half normalised one -- the
#: objective the plane refit already descends.  Measured 1.0077x at the
#: default wire on six GLM experts (every tensor 1.005-1.010x), 1.014x on the
#: span-1/S6b/refit-0 wire, at no encode cost
#: (``experiments/results/tessera_trellis_weighting_check.json``).  An encoder
#: setting, not wire; recorded in the config for the merge guard.
DEFAULT_TRELLIS_WEIGHTING = "scale"
#: The shipping wire since 2026-09-01 (schema minor 1): a span-2 trellis --
#: one select bit per two positions, 3.75 b/wt at the E2M1x2 cap -- over a
#: LUT scale plane, a 4-bit index per 16 weights into a per-unit table of
#: sixteen E4M3 scales at 0.25 bpp.  Together 4.0 bpp, the same size as the
#: span-1 trellis over the S6b plane it replaces, measured 1.111x better on
#: the output-space weight leg over six GLM experts
#: (``docs/measurements/tessera-index-plane-2026-09-01.md``).  Both are wire:
#: they travel in the manifest and are bound into the encoder profile id, and
#: ``span=1, scale_plane=S6B`` reproduces every earlier artifact byte for byte.
DEFAULT_SPAN = 2
DEFAULT_SCALE_PLANE = ScalePlaneKind.LUT
#: The BODY kind (schema minor 2).  ``WINDOW`` is the bitshift trellis over a
#: ``2^window_bits`` table on the ALPHABET plane (``encode.window_table``,
#: ``encode.viterbi_window``): measured on six GLM experts at 4.0 bpp over a
#: per-channel E4M3 plane, L=14 is 1.235x better than the TCQ body in output
#: space and 1.074x better than EXL3 K4, and on E2M1x2 below the cap (3.5
#: bpp) 1.34x better than the coset trellis
#: (``experiments/results/tessera_bitshift_tile.json``,
#: ``tessera_bitshift_tuple.json``).  **Not yet the default**, for two
#: reasons that are gates rather than doubts: the kernel lane has no window
#: GEMV (``pack_unit_for_kernel`` refuses the body), and the exact encoder is
#: O(2^window_bits) per position -- ~150 s per 2048x4096 tensor at L=14 in
#: the reference implementation, which is days per MoE model.  Both are on
#: the path; the default flips when a full expert layer has been timed on
#: a survivor-limited or Triton encoder and the kernel decodes the wire.
DEFAULT_BODY = BodyKind.TCQ
DEFAULT_WINDOW_BITS = 0
DEFAULT_WINDOW_SEED = 0
DEFAULT_WINDOW_SIGMA: "float | None" = None
DEFAULT_CHANNEL_SIGMA: "float | None" = None
#: What to do with an input Hessian when a caller supplies one.  These are
#: not encoder defaults -- ``encode_unit`` cannot invent a Hessian, so its own
#: ``ldl``/``refit_metric`` stay ``None`` and a weights-only encode is byte for
#: byte what it always was.  They are the ANSWER to "an exporter has H; now
#: what", and every one of them is measured
#: (``docs/measurements/tessera-ldlq-window-served-2026-09-02.md``): on
#: Qwen3-0.6B's FP8 wire at 4.07 bpp, served KL-vs-BF16 on vanilla vLLM goes
#: 0.1512 -> 0.1129 with LDLQ and -> 0.1046 with the full-Hessian row-scale
#: refit as well (top-1 agreement 78.1% -> 81.5%), at identical bytes; on six
#: GLM experts the out-space geomean is 0.932x at the same settings.
#: ``sigma`` and ``block`` are the pair the held-out weight-space sweep chose
#: over {0.3, 1, 3, 10} x {32, 128}; ``"hessian"`` is the exact quadratic, not
#: a diagonal power, because the row's true proxy loss has a closed form.
DEFAULT_LDLQ_SIGMA = 1.0
DEFAULT_LDLQ_BLOCK = 32

#: The config's spelling of each scale plane.  Defined here rather than beside
#: the other config helpers because the refit objective is keyed by it.
_PLANE_NAMES = {ScalePlaneKind.S6B: "s6b", ScalePlaneKind.LUT: "lut16",
                ScalePlaneKind.CHANNEL: "channel"}

#: The refit objective is per **scale plane**, and that is a measurement, not a
#: taste.  What decides it is not which objective is more faithful but which one
#: the plane's refit can *solve*:
#:
#: * ``channel`` -- one scalar per output row, rows independent under H, so the
#:   exact ``hessian`` quadratic has a closed-form minimiser and there is no
#:   table to fit.  It wins: 0.5982x vs h^1.0's 0.6376x out-space on the FP8
#:   wire, and served KL 0.1512 -> 0.1046
#:   (``tessera-ldlq-window-served-2026-09-02.md``).
#: * ``lut16`` -- one scale per sixteen input columns, all coupled by H, and
#:   sixteen shared table entries.  Under a **diagonal** metric the blocks
#:   decouple and both the coordinate step and the table fit are exact; under
#:   the full H neither is (Jacobi + one row step length, separable table fit),
#:   and the full-H arm converges to a worse point of *its own* quadratic --
#:   0.05541 vs h^1.0's 0.05393 on the fit rows, 0.05584 vs 0.05417 held out
#:   (``tessera-ldlq-lut-plane-served-2026-09-02.md``).
#: * ``s6b`` -- grouped ``(base, refine)`` words with no metric-aware refit at
#:   all, so LDLQ runs alone.  This one is a statement about the code, not a
#:   measurement.
#:
#: A plain string still means "this objective on every plane".  ``sigma`` and
#: ``block`` are the pair the held-out weight-space sweep chose over
#: {0.3, 1, 3, 10} x {32, 128}.
DEFAULT_REFIT_OBJECTIVE = MappingProxyType({
    "channel": "hessian",
    "lut16": "h^1.0",
    "s6b": "plain",
})


#: The three provenance fields that name *which* Hessian shaped the bytes.
#: ``capture_h_full.py`` writes all three: the sha of the calibration text,
#: the number of fit tokens, and the sha of the token ids themselves.  Two
#: captures that agree on all three are the same capture; two that differ on
#: any of them are different Hessians, and parts built against them are two
#: artifacts rather than one.  Required at construction rather than read with
#: ``.get`` -- an identity of ``None`` compares equal to another ``None``,
#: which is exactly how a merge guard goes vacuous.
HESSIAN_IDENTITY = ("text_sha256", "fit_tokens", "fit_ids_sha256")


def _check_refit_objective(obj: str) -> None:
    """One refit objective spelling, checked where every path can reach it."""
    if not isinstance(obj, str):
        raise GrammarError(f"a refit objective is a string, got {obj!r}")
    if obj not in ("hessian", "plain") and not obj.startswith("h^"):
        raise GrammarError(
            f"unknown refit objective {obj!r}: 'hessian' (the exact quadratic), "
            "'plain' (unweighted) or 'h^ALPHA' (a diagonal power)"
        )
    if obj.startswith("h^"):
        try:
            float(obj.removeprefix("h^"))
        except ValueError:
            raise GrammarError(f"{obj!r} is not a diagonal power h^ALPHA") from None


@dataclass(frozen=True)
class ActivationSource:
    """The input Hessians an export encodes against, and what to do with them.

    This is the one place the answer to "an exporter has H; now what" lives.
    ``export_checkpoint``, ``export_checkpoint_streaming`` and the experiment
    driver all take this object and call ``for_unit``, so the settings that
    shaped a checkpoint cannot drift between the library and a script -- which
    is the failure this class exists to prevent, the library path having had
    no Hessian at all while the script had a measured recipe.

    ``hessians`` maps a **unit** name to its ``[in_features, in_features]``
    second-moment matrix.  A unit name is the tensor name with one trailing
    ``.weight`` removed -- ``model.layers.0.mlp.up_proj`` -- which is what a
    forward hook on the module naturally produces (``capture_h_full.py``).
    There is deliberately no fallback lookup: a dict keyed the other way would
    partially match and encode part of the model weights-only, and an encode
    that silently falls back to RTN raises nothing and looks fine
    (``render-activations-keyed-by-qname``).  A missing key is refused.

    ``provenance`` is the capture's own record, carried into the exported
    config so an auditor can see which H shaped the bytes.  It must carry
    ``HESSIAN_IDENTITY``; anything else it holds (model, seqlen, source split)
    rides along unread.

    The defaults are the measured recipe -- see ``DEFAULT_LDLQ_SIGMA`` and
    ``DEFAULT_REFIT_OBJECTIVE``.  An objective is ``"hessian"`` (the exact
    quadratic), ``"plain"`` (no metric, the weights-only refit) or
    ``"h^ALPHA"`` for a diagonal power; ``refit_objective`` is either one of
    those, meaning every plane, or a ``{plane: objective}`` map, because the
    two planes that have a metric-aware refit were measured to want different
    ones.
    """

    hessians: "Mapping[str, torch.Tensor]"
    provenance: "dict"
    ldlq_sigma: "float | None" = DEFAULT_LDLQ_SIGMA
    ldlq_block: int = DEFAULT_LDLQ_BLOCK
    refit_objective: str = DEFAULT_REFIT_OBJECTIVE
    refit_reach_floor: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, dict):
            raise GrammarError(
                "an ActivationSource needs the capture's provenance dict: without "
                "it the exported config cannot say which Hessian shaped the bytes"
            )
        absent = [f for f in HESSIAN_IDENTITY if self.provenance.get(f) is None]
        if absent:
            raise GrammarError(
                f"the Hessian provenance has no {absent} -- these name the capture, "
                "and a merge guard that compares them against another part's "
                "missing values passes vacuously. Re-capture with capture_h_full.py"
            )
        if self.ldlq_sigma is not None and not self.ldlq_sigma > 0:
            raise GrammarError(
                f"the LDLQ regulariser must be positive (or None to turn LDLQ off), "
                f"got {self.ldlq_sigma}"
            )
        if self.ldlq_block < 1:
            raise GrammarError(f"the LDLQ block must be at least one column, got "
                               f"{self.ldlq_block}")
        obj = self.refit_objective
        if isinstance(obj, str):
            _check_refit_objective(obj)
        else:
            try:
                items = dict(obj)
            except TypeError:
                raise GrammarError(
                    f"refit_objective must be an objective or a plane -> objective "
                    f"mapping, got {obj!r}") from None
            if not items:
                raise GrammarError(
                    "an empty refit-objective map names no plane, so every export "
                    "through it would refuse; pass an objective string instead")
            unknown = sorted(set(items) - set(_PLANE_NAMES.values()))
            if unknown:
                raise GrammarError(
                    f"{unknown} are not scale planes: the map is keyed by the "
                    f"config's own plane spelling, one of "
                    f"{sorted(_PLANE_NAMES.values())}")
            for value in items.values():
                _check_refit_objective(value)
            object.__setattr__(self, "refit_objective", MappingProxyType(items))

    @staticmethod
    def unit_name(tensor_name: str) -> str:
        """The Hessian key for a tensor name: one trailing ``.weight`` removed."""
        return tensor_name.removesuffix(".weight")

    def objective_for(self, scale_plane: "ScalePlaneKind | None") -> str:
        """The refit objective this unit's **scale plane** was measured with.

        A plain ``refit_objective`` string applies everywhere.  A map is keyed
        by the plane, because what the refit can solve exactly is a property of
        the plane (see ``DEFAULT_REFIT_OBJECTIVE``) -- and then a caller that
        does not say which plane it is encoding on cannot be served a default,
        because the two measured answers disagree.
        """
        obj = self.refit_objective
        if isinstance(obj, str):
            return obj
        if scale_plane is None:
            raise GrammarError(
                "this ActivationSource carries a per-plane refit objective, so it "
                "needs the scale plane the unit is actually encoded on. Pass the "
                "plane from the SAME resolved recipe the encode uses "
                "(`resolve(q256).scale_plane`), never a re-derived one: an "
                "override the encode honoured and this call did not would price a "
                "different artifact than it ships"
            )
        key = _PLANE_NAMES[ScalePlaneKind(scale_plane)]
        if key not in obj:
            raise GrammarError(
                f"no refit objective for the {key!r} scale plane in {dict(obj)!r}. "
                f"Falling back to another plane's value would apply a measurement "
                f"made somewhere else, so the export refuses; name the plane, or "
                f"pass a single objective string to mean all of them"
            )
        return obj[key]

    def for_unit(self, tensor_name: str, in_features: int,
                 device: "str | torch.device" = "cpu",
                 scale_plane: "ScalePlaneKind | None" = None) -> dict:
        """The ``encode_linear`` keyword arguments this unit's Hessian implies.

        Refuses a missing key and a shape mismatch.  Both would otherwise be
        silent: a wrong key encodes the unit weights-only, and a Hessian of the
        wrong width would either broadcast or fail somewhere far from here.

        ``scale_plane`` is the plane the unit is encoded on, and it selects the
        refit objective when this source carries the per-plane map.
        """
        key = self.unit_name(tensor_name)
        if key not in self.hessians:
            raise GrammarError(
                f"no Hessian for {key!r}: the capture's keys must be the encoder's "
                f"unit names (tensor name minus one '.weight'). Encoding it "
                f"weights-only would raise nothing and quietly price a different "
                f"artifact, so the export refuses instead"
            )
        H = self.hessians[key]
        H = H.to(device=device, dtype=torch.float32)
        if H.ndim != 2 or H.shape[0] != H.shape[1] or H.shape[0] != in_features:
            raise GrammarError(
                f"{key}: H is {tuple(H.shape)} for {in_features} input features"
            )
        kwargs: dict = {"refit_reach_floor": bool(self.refit_reach_floor)}
        if self.ldlq_sigma is not None:
            from .compensate import block_ldl, regularize_hessian

            kwargs["ldl"] = block_ldl(
                regularize_hessian(H, sigma_reg=float(self.ldlq_sigma)),
                self.ldlq_block,
            )
            kwargs["ldl_block"] = self.ldlq_block
        objective = self.objective_for(scale_plane)
        if objective == "hessian":
            kwargs["refit_metric"] = H
        elif objective != "plain":
            alpha = float(objective.removeprefix("h^"))
            h = H.diagonal()
            kwargs["refit_metric"] = (h / h.mean()).pow(alpha)
        return kwargs

    @classmethod
    def from_capture(cls, path, **settings) -> "ActivationSource":
        """Load a ``capture_h_full.py`` payload and wrap it at ``settings``.

        Every driver that offers a ``--hessian`` flag comes through here, so
        the one thing they cannot disagree about is what a capture file means:
        ``H`` keyed by unit name, the capture's own provenance carried forward
        with the file's path stamped on it (a reader needs to know *which*
        file, not only which text), and the measured recipe filling in every
        setting the caller left out.  ``ldlq_sigma`` below zero is spelled by
        the CLIs as "LDLQ off" and lands here as ``None``.
        """
        import torch as _torch

        payload = _torch.load(str(path), map_location="cpu", weights_only=False)
        if "H" not in payload:
            raise GrammarError(
                f"{path} carries no 'H': a capture payload is "
                "{'H': {unit: [cols, cols]}, 'provenance': {...}}")
        sigma = settings.pop("ldlq_sigma", DEFAULT_LDLQ_SIGMA)
        if sigma is not None and float(sigma) < 0:
            sigma = None
        return cls(
            hessians=payload["H"],
            provenance=dict(payload.get("provenance") or {}, path=str(path)),
            ldlq_sigma=sigma, **settings,
        )

    def config_block(self) -> dict:
        """The ``activation_aware`` block the exported config records.

        Every field here is compared by the merge guard, and the whole block
        is what tells a reader that these bytes are **not** reproducible from
        the weights alone.
        """
        return {
            "ldlq_sigma": self.ldlq_sigma,
            "ldlq_block": self.ldlq_block,
            "refit_objective": (self.refit_objective
                                if isinstance(self.refit_objective, str)
                                else dict(self.refit_objective)),
            "refit_reach_floor": bool(self.refit_reach_floor),
            "hessian": dict(self.provenance),
            "note": "encoder-side only: the wire, the decoder and the lane are "
                    "unchanged, but this encode is not reproducible from the "
                    "weights alone -- it needs this Hessian.",
        }


_BODY_NAMES = {BodyKind.TCQ: "tcq", BodyKind.WINDOW: "window"}
#: The config's spelling of a projected field whose value varies with the
#: rung: the truth is then in ``wire.recipes``, and a reader that does not
#: know the table must not guess.
PER_RUNG = "per-rung"


@dataclass(frozen=True)
class WireRecipe:
    """The wire-level choices the exporter makes for a unit: body and plane.

    Tessera's grammar has five axes -- grid (tile x tuple), body (the
    shaping machine), rate, scale plane, and the route the decoded tile
    executes on -- and the first three and the plane are one grammar for
    the 4-bit and 8-bit tiles alike.  What differs per grid is which point of
    it the exporter writes, and that is a *recipe*: a function of the grid
    and the rung, spelled once here, read by the exporter, by PrismaQuant's
    render leg and accountant, and by the calculator, so that no consumer
    carries its own copy of a default that can drift.

    ``window_sigma`` / ``channel_sigma`` are the modelled source spreads in
    grid units (``None``: the amax-bounded source for a block plane, and
    ``scale_channel.default_channel_sigma`` for a CHANNEL plane).
    """

    body: BodyKind
    span: int
    scale_plane: ScalePlaneKind
    window_bits: int = 0
    window_seed: int = 0
    window_sigma: "float | None" = None
    channel_sigma: "float | None" = None

    def __post_init__(self) -> None:
        body = BodyKind(self.body)
        if body is BodyKind.WINDOW and self.window_bits < 1:
            raise GrammarError("a window recipe names its window_bits")
        if body is BodyKind.WINDOW and self.span != 1:
            raise GrammarError("a window recipe is span 1")
        if body is BodyKind.TCQ and self.window_bits:
            raise GrammarError("a TCQ recipe has no window_bits")

    def to_config(self) -> dict:
        """The recipe as the config spells it (one entry of ``wire.recipes``)."""
        return {
            "body": _BODY_NAMES[BodyKind(self.body)],
            "span": int(self.span),
            "plane": _PLANE_NAMES[ScalePlaneKind(self.scale_plane)],
            "window_bits": int(self.window_bits),
            "seed": int(self.window_seed),
            "sigma": None if self.window_sigma is None else float(self.window_sigma),
            "channel_sigma": (None if self.channel_sigma is None
                              else float(self.channel_sigma)),
        }

    @classmethod
    def from_config(cls, spec: dict) -> "WireRecipe":
        bodies = {name: kind for kind, name in _BODY_NAMES.items()}
        planes = {name: kind for kind, name in _PLANE_NAMES.items()}
        if spec.get("body") not in bodies:
            raise GrammarError(f"unknown body kind {spec.get('body')!r} in the recipe table")
        if spec.get("plane") not in planes:
            raise GrammarError(f"unknown scale plane {spec.get('plane')!r} in the recipe table")
        sigma, channel_sigma = spec.get("sigma"), spec.get("channel_sigma")
        return cls(
            body=bodies[spec["body"]], span=int(spec["span"]),
            scale_plane=planes[spec["plane"]],
            window_bits=int(spec.get("window_bits", 0)),
            window_seed=int(spec.get("seed", 0)),
            window_sigma=None if sigma is None else float(sigma),
            channel_sigma=None if channel_sigma is None else float(channel_sigma),
        )


#: The coset-trellis recipe (schema minor 1): the span-2 trellis over the
#: LUT scale plane.  The wire for E2M1 and for E2M1x2 at its cap.
TCQ_RECIPE = WireRecipe(
    body=DEFAULT_BODY, span=DEFAULT_SPAN, scale_plane=DEFAULT_SCALE_PLANE,
    window_bits=DEFAULT_WINDOW_BITS, window_seed=DEFAULT_WINDOW_SEED,
    window_sigma=DEFAULT_WINDOW_SIGMA, channel_sigma=DEFAULT_CHANNEL_SIGMA,
)

#: The window body's table width per grid.  L=14 on E4M3: the kernel lane
#: decodes the per-unit 2^L table at the span-2 kernel's cost through L=14
#: and 1.85x at L=16; the fused Viterbi encodes it at 1.1 s per 2048x4096
#: pass; on the wire it is 0.93x EXL3 K4 at 4.0 bpp against 0.985x at L=12.
#: L=12 below the E2M1x2 cap: the width the sub-cap wire arms were measured
#: at (1.06-1.10x EXL3 at 2.5-3.5 bpp); L=14 there is a measurement to run,
#: not a default to assume.
E4M3_WINDOW_BITS = 14
E2M1X2_SUBCAP_WINDOW_BITS = 12

#: E4M3 (schema minor 3 plane, minor 2 body): the bitshift-window trellis
#: over the CHANNEL plane -- one fp16 per output row on DIAG_SV times a
#: global, decoded to a stock per-channel FP8 tensor.
E4M3_RECIPE = WireRecipe(
    body=BodyKind.WINDOW, span=1, scale_plane=ScalePlaneKind.CHANNEL,
    window_bits=E4M3_WINDOW_BITS, window_seed=DEFAULT_WINDOW_SEED,
    window_sigma=DEFAULT_WINDOW_SIGMA, channel_sigma=DEFAULT_CHANNEL_SIGMA,
)

#: BF16 (the 16-bit route): the same window body over the same CHANNEL
#: plane the E4M3 route ships, at the same L.  Only the alphabet the table
#: snaps to changes, which is the whole point -- the E4M3 *alphabet* floors
#: the body at ~0.022 out-space from R=6 upward on a GLM expert while the
#: identical trellis over bf16 keeps halving (~1.93x per bit through R=7),
#: so the ceiling above ~6 bpp was the grid's and not the trellis's
#: (``docs/measurements/tessera16-alphabet-floor-2026-09-02.md``).  Its
#: decoded tile is a plain BF16 tensor, so the route is the BF16 GEMM
#: (W16A16) and there is no weight-side hardware format to satisfy.
BF16_WINDOW_BITS = E4M3_WINDOW_BITS
#: The modelled source spread in grid units.  It was **stated rather than
#: searched** -- ``scale_channel.default_channel_sigma`` walks a dyadic ladder
#: for the spread with the smallest nearest-value error, and on a grid with
#: eight exponent bits that error is scale-free over ~30 binades, so the
#: ladder is choosing between equals and would build a 4096 x 65536 float64
#: distance matrix to do it.  1.0 puts the L=14 table's quantiles
#: [7.6e-5, 4.05] deep inside the format with ~120 binades of margin at each
#: end and gives the body reach 4.00 sigma, next to E4M3's 4.08.
#:
#: **Searched since, and it is a gauge, not a knob** (issue #18,
#: ``experiments/bf16_l_sigma_sweep.py``).  With ``window_sigma`` left at
#: ``None`` the table is built at ``sigma = channel_sigma``, so both ends of
#: the scale move together; bf16 is closed under x2 and nearest-value snapping
#: commutes with x2.  So a **dyadic** shift is written down rather than felt:
#: the ALPHABET plane holds the doubled table and the fp32 global halves, the
#: file hash moves and *the decoded tensor does not*.  Measured on four dense
#: Qwen Linears, x0.25 / x0.5 / x1 / x2 / x4 decode **bit-identically** and
#: x0.75 / x1.5 / x3 decode bit-identically to each *other* -- two orbits, one
#: per odd part of the multiplier -- differing in ``wt`` by 0.007-0.02%.  The
#: invariant is the tensor, not the file, and there is no dyadic value of this
#: constant left to find.  The contrast is the proof that this is the grid's
#: property and not the mechanism's: over those same four units the identical
#: shift on E4M3 costs +5 to +19% ``wt`` at x2 and +70 to +98% at x4, because
#: that table runs past 448.
#:
#: What remains searchable is ``L`` and the ``window_sigma``/``channel_sigma``
#: *ratio* -- the table's spread against the row's, which is what sets reach.
BF16_CHANNEL_SIGMA = 1.0
BF16_RECIPE = WireRecipe(
    body=BodyKind.WINDOW, span=1, scale_plane=ScalePlaneKind.CHANNEL,
    window_bits=BF16_WINDOW_BITS, window_seed=DEFAULT_WINDOW_SEED,
    window_sigma=DEFAULT_WINDOW_SIGMA, channel_sigma=BF16_CHANNEL_SIGMA,
)

#: E2M1x2 below the coset trellis's cap: the window body over the same LUT
#: plane the cap recipe uses.
E2M1X2_SUBCAP_RECIPE = WireRecipe(
    body=BodyKind.WINDOW, span=1, scale_plane=ScalePlaneKind.LUT,
    window_bits=E2M1X2_SUBCAP_WINDOW_BITS, window_seed=DEFAULT_WINDOW_SEED,
    window_sigma=DEFAULT_WINDOW_SIGMA, channel_sigma=DEFAULT_CHANNEL_SIGMA,
)


def tcq_cap_q256(grid: PayloadGrid) -> int:
    """The coset trellis's highest rung on ``grid`` in q256 per position:
    ``payload_bits - 1`` per tuple, the bit the trellis spends on its code."""
    return grid.rate_cap * 256 // grid.arity


def _window_bits_for(default: int, grid: PayloadGrid, q256: "int | None") -> int:
    """The table width a window recipe needs at rung ``q256``: ``L >= R``.

    A window position's ``R`` new bits are the low ``R`` bits of the state,
    so a table narrower than the rate cannot hold them and ``encode_unit``
    refuses -- correctly, and after the recipe has already claimed a width it
    cannot honour.  Deciding it here instead means the recipe table states a
    width that works at every rung it covers, which is what a checkpoint's
    own ``wire.recipes`` is for.

    A no-op on every grid that ships today: E4M3 and E2M1x2 cap at 8 bits per
    code against tables of 14 and 12.  It bites only on BF16, whose 16
    payload bits run past the 14-bit table the route is measured at, and
    there it widens the table rather than refusing the rung.
    """
    if q256 is None:
        return default
    rate = min(grid.payload_bits, -(-int(q256) * grid.arity // 256))
    return max(default, rate)


def wire_recipe(grid: PayloadGrid, q256: "int | None" = None) -> WireRecipe:
    """The wire the exporter writes for a unit on ``grid`` at rung ``q256``.

    Measured on the six GLM-5.3-Flash routed experts against EXL3 on the
    same held-out rows (``docs/tessera-one-format.md`` §4,
    ``docs/measurements/tessera-window-body-2026-09-02.md``), and flipped
    2026-09-02 once its two mechanical gates closed -- the kernel lane
    decodes the window body bit-exactly over every plane, and the fused
    Viterbi encodes it 15-26x faster than the reference, bit-exact:

    * **BF16**, every rung: ``BF16_RECIPE`` -- the identical window body
      over the identical CHANNEL plane at the identical L, with the table
      snapped to bf16 instead of E4M3 and the source spread stated at 1.0.
      Its decoded tile is a plain BF16 tensor (W16A16).  The TCQ body is
      not reachable on this grid at all: a 65536-anchor forest is what the
      encoder already refuses, and the window body never scores it.
    * **E4M3**, every rung: ``E4M3_RECIPE`` -- the window body over the
      CHANNEL plane, L=14.  0.93x EXL3 K4 at 4.0 bpp and 0.92-0.95x at 5.0
      on the wire (0.985x / 1.016x at L=12), before LDLQ; the coset trellis
      over LUT16 it replaces was 1.20x / 1.23x.  Its decoded tile is a stock
      per-channel FP8 tensor, so the route is the FP8 MMA (W8A8).
    * **E2M1x2 below the cap** (``q256 < tcq_cap_q256(grid)``, 3.5 body bits
      per weight): ``E2M1X2_SUBCAP_RECIPE`` -- the window body over LUT16,
      L=12.  1.06-1.10x EXL3 at 2.5-3.5 bpp where the coset trellis is
      1.36-1.43x.
    * **E2M1x2 at the cap** and **E2M1**: ``TCQ_RECIPE``.  At the cap the
      structured coset table beats the window on the wire at L=12 (1.170x
      against 1.244x) and at L=14 (1.21x): the window pays the table's
      entropy and the code bit is cheaper.  E2M1 (arity 1) is unmeasured
      under the window body and keeps the recipe it was measured with.

    The table this defines is what the config records per rung
    (``recipe_table``), so a checkpoint carries its own meaning even if
    these lines move again.
    """
    if grid.arity == 1 and grid.name == "E4M3":
        return E4M3_RECIPE
    if grid.arity == 1 and grid.name == "BF16":
        bits = _window_bits_for(BF16_WINDOW_BITS, grid, q256)
        if bits == BF16_RECIPE.window_bits:
            return BF16_RECIPE
        return WireRecipe(
            body=BF16_RECIPE.body, span=1, scale_plane=BF16_RECIPE.scale_plane,
            window_bits=bits, window_seed=BF16_RECIPE.window_seed,
            window_sigma=BF16_RECIPE.window_sigma,
            channel_sigma=BF16_RECIPE.channel_sigma,
        )
    if grid.arity == 2 and grid.name.startswith("E2M1") and q256 is not None \
            and q256 < tcq_cap_q256(grid):
        return E2M1X2_SUBCAP_RECIPE
    return TCQ_RECIPE


@dataclass(frozen=True)
class RecipeRange:
    """One row of the recipe table: the recipe every rung in [lo, hi] gets."""

    q256_lo: int
    q256_hi: int
    recipe: WireRecipe

    def to_config(self) -> dict:
        return {"q256_lo": int(self.q256_lo), "q256_hi": int(self.q256_hi),
                **self.recipe.to_config()}

    @classmethod
    def from_config(cls, spec: dict) -> "RecipeRange":
        return cls(int(spec["q256_lo"]), int(spec["q256_hi"]), WireRecipe.from_config(spec))


def rung_ceiling(grid: PayloadGrid) -> int:
    """The highest rung any body admits on ``grid``, in q256 per position.

    The window body may spend the grid's whole payload width at a position
    (``_plan_for``), so the ceiling is ``payload_bits`` per tuple; the TCQ
    body's cap is one bit lower and is refused by the schedule, not here.
    """
    return grid.payload_bits * 256 // grid.arity


def recipe_table(grid: PayloadGrid, resolve=None) -> "tuple[RecipeRange, ...]":
    """The recipe at every rung of ``grid``, as contiguous ranges.

    ``resolve(grid, q256)`` defaults to ``wire_recipe``; an exporter passes
    its own resolver, which applies the caller's explicit overrides on top
    of the recipe, so the table it writes describes the checkpoint it built
    rather than the module's default.  The table is a function of the
    resolver alone, never of the plan: two parts of one checkpoint built by
    the same code carry identical tables whatever rungs each used, which is
    what lets the merge guard compare them.
    """
    resolve = wire_recipe if resolve is None else resolve
    ranges: "list[RecipeRange]" = []
    for q in range(1, rung_ceiling(grid) + 1):
        recipe = resolve(grid, q)
        if ranges and ranges[-1].recipe == recipe:
            ranges[-1] = RecipeRange(ranges[-1].q256_lo, q, recipe)
        else:
            ranges.append(RecipeRange(q, q, recipe))
    return tuple(ranges)


def recipe_at(table: "tuple[RecipeRange, ...]", q256: int) -> WireRecipe:
    """The table's recipe for rung ``q256``; a rung outside every range is refused."""
    for entry in table:
        if entry.q256_lo <= q256 <= entry.q256_hi:
            return entry.recipe
    raise GrammarError(f"rung {q256} is outside the recipe table's ranges")


@dataclass(frozen=True)
class ExportedUnit:
    """One serialised Linear, with the bytes it actually cost."""

    name: str
    blob: bytes
    rows: int
    columns: int
    q256: int
    exact_bytes: int

    @property
    def params(self) -> int:
        return self.rows * self.columns

    @property
    def bpp(self) -> Fraction:
        """Bits per quantizable parameter, exact -- counted, never estimated."""
        return Fraction(self.exact_bytes * 8, self.params)


@dataclass(frozen=True)
class ExportReport:
    """What was written, and what it weighs."""

    units: "tuple[ExportedUnit, ...]"
    passthrough_bytes: int
    quantized_bytes: int
    quantized_params: int
    grid_digest: str

    @property
    def body_bpp(self) -> Fraction:
        """bpp over *quantizable* parameters only (principle 12).

        Passthrough tensors -- embeddings, norms, anything the plan left alone
        -- are excluded from the denominator and reported separately, because
        a bpp that silently averages them in is not comparable to anyone's.
        """
        if not self.quantized_params:
            return Fraction(0)
        return Fraction(self.quantized_bytes * 8, self.quantized_params)

    @property
    def total_bytes(self) -> int:
        return self.passthrough_bytes + self.quantized_bytes


@lru_cache(maxsize=256)
def _plan_for(
    grid: PayloadGrid, q256: int, columns: int, body: BodyKind = BodyKind.TCQ,
    source_sigma: "float | None" = None,
):
    """Rate schedule and forests for one (grid, rung, width).

    Cached because the forests are an exhaustive per-rate optimisation and are
    identical for every Linear of the same width at the same rung -- on a
    288-expert MoE layer that is hundreds of units sharing one plan, and
    rebuilding it per tensor is the export's largest avoidable cost.  A
    window body has no forests: the second element is then the grid itself,
    which is what every decode-side entry point accepts in their place.

    The rate ceiling depends on the body.  The TCQ trellis spends one bit of
    the payload on its code, so its cap is ``payload_bits - 1``; the window
    body's shaping is the ``L - R`` bits of shared history, not a code bit,
    so a position may spend the grid's whole width -- ``R = payload_bits``
    is an ordinary rung there (E2M1x2 at 4.0 body bits per weight).

    ``source_sigma`` is the spread, in grid units, of the Gaussian a TCQ
    forest is optimised against -- the CHANNEL plane's rows are scaled to it
    -- and ``None`` is the amax-bounded source a block plane delivers.
    """
    root = Fraction(q256 * grid.arity, 256)
    body = BodyKind(body)
    cap = grid.payload_bits if body is BodyKind.WINDOW else grid.rate_cap
    rates = bresenham_rate_schedule(root, columns, cap=cap)
    if body is BodyKind.WINDOW:
        return rates, grid
    samples = None if source_sigma is None else GAUSSIAN_SOURCE(1 << 14, float(source_sigma))
    forests = {
        rate: build_forest(rate, samples=samples, grid=grid) for rate in sorted(set(rates))
    }
    return rates, forests


def encode_linear_planes(
    weight: torch.Tensor,
    *,
    grid: PayloadGrid,
    q256: int,
    name: str = "unit",
    code: ConvCode = DEFAULT_CODE,
    group: int = DEFAULT_GROUP,
    half: int = DEFAULT_HALF,
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    completion: "int | None" = 0,
    verify: bool = True,
    scale_refit: int = DEFAULT_SCALE_REFIT,
    span: "int | None" = None,
    scale_plane: "ScalePlaneKind | None" = None,
    trellis_weighting: str = DEFAULT_TRELLIS_WEIGHTING,
    body: "BodyKind | None" = None,
    window_bits: "int | None" = None,
    window_seed: "int | None" = None,
    window_sigma: "float | None" = DEFAULT_WINDOW_SIGMA,
    channel_sigma: "float | None" = DEFAULT_CHANNEL_SIGMA,
    ldl: "torch.Tensor | None" = None,
    ldl_block: int = DEFAULT_LDLQ_BLOCK,
    refit_metric: "torch.Tensor | None" = None,
    refit_reach_floor: bool = False,
) -> "tuple[ExportedUnit, EncodedUnit, object]":
    """Encode one ``[out_features, in_features]`` weight to artifact bytes.

    Returns ``(exported, unit, forests)``: the bytes with their exact cost,
    and the planes and forests they were built from, for a caller that
    materialises the same unit a second way (``stock.materialize_stock``)
    and needs it to be *this* encoding and not a re-run of it.  ``encode_linear``
    is the same call returning the bytes alone.

    ``span``, ``scale_plane``, ``body``, ``window_bits`` and ``window_seed``
    default to ``None``, which means *the recipe*: ``wire_recipe(grid,
    q256)`` resolves each one the caller left unset, so a caller that names
    none of them gets the shipping wire for its grid, and one that names
    some overrides only those.  ``window_sigma``/``channel_sigma`` are the
    modelled source spreads (``encode_unit``).

    ``completion`` is the second rate axis and it was previously nailed shut at
    zero here.  A column at body rate ``R`` may spend up to ``cap - R`` further
    bits selecting among the descendants its trellis subset reaches; ``None``
    spends every one of them, an integer spends at most that many, and ``0``
    spends none.  It is a real rate: the artifact pays for what it spends, so
    ``(q256, completion)`` is a two-dimensional rate grid, not a rung and a
    switch.  The default stays ``0`` so the exporter's rung names keep meaning
    the rate they have always meant -- ``q256`` alone -- and a caller that wants
    the other axis asks for it.

    ``ldl``/``ldl_block``/``refit_metric``/``refit_reach_floor`` are the
    activation-aware encoder settings (``encode_unit``): the input Hessian's
    LDL factor for this unit's columns, and the error the row-scale refit
    minimises.  They change no byte of the wire's grammar -- an artifact built
    with them is read by the same decoder -- but an encode that uses them is
    not reproducible from the weights alone, so an exporter that sets them
    records where its Hessian came from.

    ``verify`` reads the bytes back and compares to the encoder's own
    reconstruction.  It is on by default and costs one decode: the guarantee
    that the shipped bytes mean what the surrogate priced is worth more than
    the milliseconds, and an exporter that only *believes* it round-trips is
    how a rendering confound gets into an artifact.
    """
    if weight.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D weight, got {tuple(weight.shape)}")
    rows, columns = weight.shape
    if rows % grid.arity:
        raise GrammarError(
            f"{name}: {rows} rows is not divisible by the grid arity {grid.arity}"
        )
    recipe = _resolve_recipe(
        grid, span, scale_plane, body, window_bits, window_seed, window_sigma,
        channel_sigma,
    )(q256)
    span, scale_plane, body = recipe.span, recipe.scale_plane, recipe.body
    window_bits, window_seed = recipe.window_bits, recipe.window_seed
    window_sigma, channel_sigma = recipe.window_sigma, recipe.channel_sigma
    source_sigma = channel_sigma if scale_plane is ScalePlaneKind.CHANNEL else None
    rates, forests = _plan_for(grid, q256, columns, body, source_sigma)
    if body is BodyKind.WINDOW and completion not in (None, 0):
        raise GrammarError(f"{name}: a window body has no completion axis")
    unit = encode_unit(
        weight, forests, rates, code,
        rotation=rotation, with_diagonals=with_diagonals,
        completion=0 if body is BodyKind.WINDOW else completion, group=group, half=half,
        scale_refit=scale_refit, span=span, scale_plane=scale_plane,
        trellis_weighting=trellis_weighting,
        body=body, window_bits=window_bits, window_seed=window_seed,
        window_sigma=window_sigma, channel_sigma=channel_sigma,
        ldl=ldl, ldl_block=ldl_block, refit_metric=refit_metric,
        refit_reach_floor=refit_reach_floor,
    )
    # ``q256`` here is the rung's PER-POSITION rate (the R-number in a rung
    # name, and what ``artifact_bpp`` prices).  ``build_unit_artifact`` declares
    # the per-CODE rate, and a code spans ``arity`` positions.  Passing the
    # per-position number straight through produces a legal artifact whose
    # manifest states half the rate it carries -- silent, and exactly the
    # confusion ``build_unit_artifact``'s own comment flags.
    _, region, blob = build_unit_artifact(
        unit, name, forests, q256 * grid.arity, code
    )
    if verify:
        recovered = read_unit_artifact(blob, device=weight.device)
        reference = reconstruct_unit(unit, forests, code)
        if not torch.equal(recovered, reference):
            raise GrammarError(
                f"{name}: the bytes do not decode to the encoder's own "
                "reconstruction -- refusing to write a unit whose surrogate "
                "and payload disagree"
            )
    exported = ExportedUnit(
        name=name, blob=blob, rows=rows, columns=columns,
        q256=q256, exact_bytes=len(region),
    )
    return exported, unit, forests


def encode_linear(weight: torch.Tensor, **kwargs) -> ExportedUnit:
    """``encode_linear_planes`` returning the serialised unit alone.

    The keyword surface is ``encode_linear_planes``'s, unchanged: ``grid``
    and ``q256`` are required, the wire fields default to the recipe.
    """
    return encode_linear_planes(weight, **kwargs)[0]


def export_checkpoint(
    tensors: "dict[str, torch.Tensor]",
    plan: "dict[str, int]",
    out_dir: "str | Path",
    *,
    grid: PayloadGrid,
    code: ConvCode = DEFAULT_CODE,
    group: int = DEFAULT_GROUP,
    half: int = DEFAULT_HALF,
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    extra_config: "dict | None" = None,
    verify: bool = True,
    scale_refit: int = DEFAULT_SCALE_REFIT,
    span: "int | None" = None,
    scale_plane: "ScalePlaneKind | None" = None,
    trellis_weighting: str = DEFAULT_TRELLIS_WEIGHTING,
    body: "BodyKind | None" = None,
    window_bits: "int | None" = None,
    window_seed: "int | None" = None,
    window_sigma: "float | None" = DEFAULT_WINDOW_SIGMA,
    channel_sigma: "float | None" = DEFAULT_CHANNEL_SIGMA,
    activation: "ActivationSource | None" = None,
) -> ExportReport:
    """Write ``tensors`` to ``out_dir``, encoding every name ``plan`` rates.

    The wire fields left ``None`` resolve to ``wire_recipe(grid, ...)``
    exactly as in ``encode_linear``; the config records the resolved recipe.

    ``plan`` maps tensor name -> per-position body rate in q256 units.  A name
    in ``plan`` that is absent from ``tensors`` is an error rather than a
    no-op: a plan that silently fails to apply is how an artifact ends up
    heavier than the allocation that justified it.

    ``activation`` supplies per-unit input Hessians and turns on the measured
    activation-aware recipe (LDLQ plus the full-Hessian row-scale refit).  It
    is refused, not ignored, when a planned unit has no Hessian: an encode
    that quietly falls back to weights-only raises nothing and ships a
    different artifact from the one that was priced.  The bytes stay on the
    same wire -- both levers are encoder-side -- and the config records which
    capture shaped them.
    """
    from safetensors.torch import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    missing = sorted(set(plan) - set(tensors))
    if missing:
        raise KeyError(
            f"plan names {len(missing)} tensor(s) not present: {missing[:5]}"
        )
    resolve = _resolve_recipe(
        grid, span, scale_plane, body, window_bits, window_seed, window_sigma,
        channel_sigma,
    )
    table = recipe_table(grid, lambda _grid, q: resolve(q))

    payload: "dict[str, torch.Tensor]" = {}
    units: "list[ExportedUnit]" = []
    passthrough_bytes = 0
    for name, tensor in tensors.items():
        if name in plan:
            unit = encode_linear(
                tensor, grid=grid, q256=plan[name], name=name, code=code,
                group=group, half=half, rotation=rotation,
                with_diagonals=with_diagonals, verify=verify,
                scale_refit=scale_refit, trellis_weighting=trellis_weighting,
                **_recipe_kwargs(resolve(plan[name])),
                **(activation.for_unit(
                    name, tensor.shape[1], tensor.device,
                    scale_plane=resolve(plan[name]).scale_plane)
                   if activation is not None else {}),
            )
            units.append(unit)
            payload[name + BLOB_SUFFIX] = torch.frombuffer(
                bytearray(unit.blob), dtype=torch.uint8
            )
        else:
            payload[name] = tensor.contiguous().cpu()
            passthrough_bytes += tensor.numel() * tensor.element_size()

    save_file(payload, str(out / "model.safetensors"))

    quantized_bytes = sum(u.exact_bytes for u in units)
    quantized_params = sum(u.params for u in units)
    report = ExportReport(
        units=tuple(units),
        passthrough_bytes=passthrough_bytes,
        quantized_bytes=quantized_bytes,
        quantized_params=quantized_params,
        grid_digest=grid_digest(grid),
    )

    _write_config(out, grid, code, group, half, rotation, with_diagonals,
                  report, plan, extra_config, scale_refit, trellis_weighting, table,
                  activation)
    return report


def _resolve_recipe(grid, span, scale_plane, body, window_bits, window_seed,
                    window_sigma, channel_sigma):
    """The per-rung resolver an exporter and ``encode_linear`` share.

    Returns ``resolve(q256) -> WireRecipe``: the fields the caller left
    ``None`` come from ``wire_recipe(grid, q256)``, the ones it named override
    them for every rung alike.  A window body has no super-symbols, so its
    span is resolved to 1 rather than making every window caller spell
    ``span=1`` to escape a TCQ default that does not apply to it; a CHANNEL
    plane's source spread resolves to ``default_channel_sigma`` so the TCQ
    forest and the encoder's rows model the same Gaussian.  Rungs that
    resolve to different recipes are not refused: the config records the
    whole table (``recipe_table``) and replays each unit at its own meaning.
    """
    from .scale_channel import default_channel_sigma

    def resolve(q256: "int | None") -> WireRecipe:
        recipe = wire_recipe(grid, q256)
        r_body = BodyKind(recipe.body if body is None else body)
        if r_body is BodyKind.TCQ and recipe.body is not BodyKind.TCQ:
            # A caller that names the coset trellis over a window recipe
            # gets the trellis's own wire for the body's fields -- its span
            # from TCQ_RECIPE, no table -- and the grid recipe's plane.
            recipe = WireRecipe(
                body=BodyKind.TCQ, span=TCQ_RECIPE.span, scale_plane=recipe.scale_plane,
                window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
            )
        r_span = recipe.span if span is None else int(span)
        if r_body is BodyKind.WINDOW:
            r_span = 1
        r_plane = ScalePlaneKind(recipe.scale_plane if scale_plane is None else scale_plane)
        # A caller that names the TCQ body over a window recipe gets the
        # trellis, not the recipe's table width: the width belongs to the
        # body, and a TCQ recipe has none.
        r_bits = recipe.window_bits if window_bits is None else int(window_bits)
        r_seed = recipe.window_seed if window_seed is None else int(window_seed)
        r_sigma = recipe.window_sigma if window_sigma is None else float(window_sigma)
        r_csigma = recipe.channel_sigma if channel_sigma is None else float(channel_sigma)
        if r_plane is ScalePlaneKind.CHANNEL and r_csigma is None:
            r_csigma = default_channel_sigma(grid)
        return WireRecipe(
            body=r_body, span=r_span, scale_plane=r_plane, window_bits=r_bits,
            window_seed=r_seed, window_sigma=r_sigma, channel_sigma=r_csigma,
        )

    return resolve


def _recipe_kwargs(recipe: WireRecipe) -> dict:
    """``encode_linear``'s wire keywords for one resolved recipe."""
    return dict(
        span=recipe.span, scale_plane=recipe.scale_plane, body=recipe.body,
        window_bits=recipe.window_bits, window_seed=recipe.window_seed,
        window_sigma=recipe.window_sigma, channel_sigma=recipe.channel_sigma,
    )


def _projected(recipes: "tuple[WireRecipe, ...]", field, mixed=PER_RUNG):
    """A recipe field for the config's flat keys: its value when every recipe
    the checkpoint used agrees, else ``mixed`` -- the flat keys are a
    projection of ``wire.recipes`` over the plan's rungs, kept for readers
    of the earlier configs, and a projection that averaged two bodies into
    one would be a lie."""
    values = {field(recipe) for recipe in recipes}
    return next(iter(values)) if len(values) == 1 else mixed


def _used_recipes(table: "tuple[RecipeRange, ...]", rungs) -> "tuple[WireRecipe, ...]":
    """The distinct recipes the rungs in ``rungs`` resolve to on ``table``;
    every recipe of the table when no rung is known."""
    rungs = sorted(set(rungs))
    if not rungs:
        return tuple(dict.fromkeys(entry.recipe for entry in table))
    return tuple(dict.fromkeys(recipe_at(table, q) for q in rungs))


def _write_config(out: Path, grid, code, group, half, rotation, with_diagonals,
                  report: "ExportReport", plan: "dict[str, int]",
                  extra_config: "dict | None", scale_refit: int = 0,
                  trellis_weighting: str = "none",
                  table: "tuple[RecipeRange, ...] | None" = None,
                  activation: "ActivationSource | None" = None) -> None:
    if table is None:
        table = recipe_table(grid)
    used = _used_recipes(table, plan.values())
    span = _projected(used, lambda r: int(r.span), mixed=None)
    plane = _projected(used, lambda r: _PLANE_NAMES[ScalePlaneKind(r.scale_plane)])
    body = _projected(used, lambda r: _BODY_NAMES[BodyKind(r.body)])
    window_bits = _projected(used, lambda r: int(r.window_bits), mixed=None)
    window_seed = _projected(used, lambda r: int(r.window_seed), mixed=None)
    window_sigma = _projected(
        used, lambda r: None if r.window_sigma is None else float(r.window_sigma), mixed=None)
    channel_sigma = _projected(
        used, lambda r: None if r.channel_sigma is None else float(r.channel_sigma),
        mixed=None)
    config = {
        "quant_method": "tessera",
        "container_version": CONTAINER_VERSION,
        "blob_suffix": BLOB_SUFFIX,
        "grid": {
            # The digest is the wire identity and the only field a reader may
            # trust.  The name and base are recorded so the config can be
            # *audited* -- a config that cannot say which grid it used forces
            # every reader to reverse a hash to answer "what format is this?".
            "digest": grid_digest(grid),
            "name": grid.name,
            "base": grid.name.split("x")[0],
            "partition": grid.partition,
            "arity": grid.arity,
            "size": grid.size,
            "rate_cap": grid.rate_cap,
        },
        "conv_memory": code.memory,
        # The generators are wire too (bound into the encoder profile id): a
        # code of the same memory order with different taps emits streams
        # that decode to other weights.  Octal strings, the notation the
        # code's own table uses; a config without this key means the table's
        # default for its memory order, which is what every earlier artifact
        # was written with.
        "conv_generators": [oct(g) for g in code.generators],
        # The trellis span is wire (manifest field, profile-id tag).  Recorded
        # here as well so a merge can refuse parts built at different spans
        # without opening a blob.
        # ``weighting`` is the Viterbi's branch-metric weight (an encoder
        # setting: ``none`` = per-half normalised error, ``scale`` = true
        # squared error); the merge guard compares it like ``scale.refit``.
        "trellis": {"span": span, "weighting": str(trellis_weighting)},
        # The BODY kind (schema minor 2).  ``window_bits`` is wire (manifest
        # field, profile-id tag); ``seed`` and ``sigma`` are the table's
        # construction parameters -- the table itself is on the plane, so a
        # reader never needs them, but a replay does and a merge must compare
        # them: two halves over different tables are two artifacts.  A config
        # without this key means the TCQ body, which is every artifact before
        # the field existed.  These flat keys, ``trellis.span`` and
        # ``scale.plane`` are projections of ``wire.recipes`` over the rungs
        # this checkpoint used: when those rungs' recipes differ they read
        # ``per-rung`` (``null`` for the numbers) and the table is the truth.
        "body": {"kind": body, "window_bits": window_bits, "seed": window_seed,
                 "sigma": window_sigma},
        # ``refit`` counts trellis passes (= refits); ``schedule`` says how they
        # interleave, because the same count meant a different encoder before
        # 61df165 (k refits BETWEEN k+1 passes) -- the merge guard compares both.
        # ``plane`` is the segment-2b kind: ``s6b`` (E8M0 base + nibble),
        # ``lut16`` (nibble into a per-unit sixteen-entry E4M3 table) or
        # ``channel`` (schema minor 3: one fp16 word per output row on
        # DIAG_SV times a global; ``sigma`` is the source spread, in grid
        # units, the rows were scaled to and the table/forest modelled).
        "scale": {"group": group, "half": half, "refit": scale_refit,
                  "schedule": "amax" if scale_refit == 0 else "trailing-refit",
                  "plane": plane, "sigma": channel_sigma},
        # The recipe at every rung of the grid, as contiguous q256 ranges:
        # body, span, plane, window table parameters and modelled spreads.  A
        # function of the exporter's resolver alone, never of the plan, so
        # every part of one checkpoint carries the same table (the merge
        # guard compares it) and a replay of any rung finds its meaning here
        # (``encode_settings_from_config``).
        "wire": {"recipes": [entry.to_config() for entry in table]},
        "rotation": rotation.name,
        "with_diagonals": bool(with_diagonals),
        "route_status": "unbacked",
        "requires_serve_flags": [],
        # A unit is one trellis blob, not a sliceable tensor: the path runs down
        # rows within a column, so a row-parallel split cuts the trellis along
        # its own state. EXL3 narrows tensor dims and is TP-agnostic; Tessera
        # must be *re-encoded* per rank, which makes an artifact TP-specific.
        # Declared so a loader cannot quietly use it at the wrong degree.
        "tp_size": 1,
        "accounting": {
            "quantized_params": report.quantized_params,
            "quantized_bytes": report.quantized_bytes,
            "passthrough_bytes": report.passthrough_bytes,
            "body_bpp": float(report.body_bpp),
            "body_bpp_exact": [report.body_bpp.numerator,
                               report.body_bpp.denominator],
        },
        # Which Hessian shaped these bytes, and what was done with it.
        # ``null`` means a weights-only encode -- reproducible from the source
        # weights and this config alone.  A dict means it is NOT: the bytes
        # depend on the named capture, so the merge guard compares every field
        # here and refuses parts built against different Hessians, and
        # ``encode_settings_from_config`` refuses to hand back encode arguments
        # that would silently replay the unit weights-only.
        "activation_aware": None if activation is None else activation.config_block(),
        "plan": dict(plan),
        "rungs_q256": sorted({u.q256 for u in report.units}),
    }
    if extra_config:
        config.update(extra_config)
    (out / "tessera_config.json").write_text(json.dumps(config, indent=2))


def export_checkpoint_streaming(
    source_dir: "str | Path",
    out_dir: "str | Path",
    plan: "dict[str, int]",
    *,
    grid: PayloadGrid,
    code: ConvCode = DEFAULT_CODE,
    group: int = DEFAULT_GROUP,
    half: int = DEFAULT_HALF,
    rotation: RotationState = RotationState.NONE,
    with_diagonals: bool = False,
    device: "str | torch.device" = "cuda",
    extra_config: "dict | None" = None,
    verify: bool = True,
    scale_refit: int = DEFAULT_SCALE_REFIT,
    copy_aux: bool = True,
    progress=None,
    shard_filter: "set[str] | None" = None,
    span: "int | None" = None,
    scale_plane: "ScalePlaneKind | None" = None,
    trellis_weighting: str = DEFAULT_TRELLIS_WEIGHTING,
    body: "BodyKind | None" = None,
    window_bits: "int | None" = None,
    window_seed: "int | None" = None,
    window_sigma: "float | None" = DEFAULT_WINDOW_SIGMA,
    channel_sigma: "float | None" = DEFAULT_CHANNEL_SIGMA,
    activation: "ActivationSource | None" = None,
) -> ExportReport:
    """Export shard-by-shard, holding one shard in memory at a time.

    The in-memory ``export_checkpoint`` cannot touch the models this format
    exists for: the target is a 100B-plus checkpoint whose weights never fit
    beside their own encoding.  One output shard is written per input shard, so
    the mapping stays 1:1 and a partial run is inspectable rather than opaque.

    Encoding runs on ``device``; the trellis is the whole cost of an export and
    it is a GPU job (principle 7).

    ``shard_filter`` restricts the run to a subset of input shards.  The 1:1
    shard mapping is what makes this safe: shards share no state -- the plan is
    per-tensor, the forests are rebuilt per (grid, rung, width) and cached, and
    nothing accumulates across shards except the report -- so N boxes each
    taking a disjoint subset produce exactly the files one box would have
    written, and the run becomes embarrassingly parallel across a fleet.  The
    index and config a filtered run writes cover **only its own shards**; the
    caller merges them.  This exists because a 320B-parameter export is nine
    hours on one GB10 and the second one was idle at 4 W.

    ``activation`` supplies per-unit input Hessians and turns on the measured
    activation-aware recipe (LDLQ plus the full-Hessian row-scale refit).  It
    is refused, not ignored, when a planned unit has no Hessian: an encode
    that quietly falls back to weights-only raises nothing and ships a
    different artifact from the one that was priced.  The bytes stay on the
    same wire -- both levers are encoder-side -- and the config records which
    capture shaped them.
    """
    import shutil

    from safetensors import safe_open
    from safetensors.torch import save_file

    src = Path(source_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    index_path = src / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: "dict[str, list[str]]" = {}
        for tensor_name, shard in weight_map.items():
            shards.setdefault(shard, []).append(tensor_name)
    else:
        shards = {}
        for shard_path in sorted(src.glob("*.safetensors")):
            with safe_open(str(shard_path), framework="pt") as handle:
                shards[shard_path.name] = list(handle.keys())

    if shard_filter is not None:
        unknown = sorted(set(shard_filter) - set(shards))
        if unknown:
            raise KeyError(f"shard_filter names absent shards: {unknown[:5]}")
        shards = {k: v for k, v in shards.items() if k in shard_filter}
        if not shards:
            raise ValueError("shard_filter selected no shards")

    known = {name for names in shards.values() for name in names}
    missing = sorted(set(plan) - known)
    if missing and shard_filter is None:
        raise KeyError(
            f"plan names {len(missing)} tensor(s) not present: {missing[:5]}"
        )
    resolve = _resolve_recipe(
        grid, span, scale_plane, body, window_bits, window_seed, window_sigma,
        channel_sigma,
    )
    table = recipe_table(grid, lambda _grid, q: resolve(q))

    units: "list[ExportedUnit]" = []
    passthrough_bytes = 0
    new_weight_map: "dict[str, str]" = {}

    for position, (shard, names) in enumerate(sorted(shards.items()), start=1):
        payload: "dict[str, torch.Tensor]" = {}
        with safe_open(str(src / shard), framework="pt") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if name in plan:
                    unit = encode_linear(
                        tensor.to(device), grid=grid, q256=plan[name], name=name,
                        code=code, group=group, half=half, rotation=rotation,
                        with_diagonals=with_diagonals, verify=verify,
                        scale_refit=scale_refit,
                        trellis_weighting=trellis_weighting,
                        **_recipe_kwargs(resolve(plan[name])),
                        **(activation.for_unit(
                            name, tensor.shape[1], device,
                            scale_plane=resolve(plan[name]).scale_plane)
                           if activation is not None else {}),
                    )
                    units.append(unit)
                    key = name + BLOB_SUFFIX
                    payload[key] = torch.frombuffer(
                        bytearray(unit.blob), dtype=torch.uint8
                    )
                else:
                    payload[name] = tensor.contiguous()
                    passthrough_bytes += tensor.numel() * tensor.element_size()
        for key in payload:
            new_weight_map[key] = shard
        save_file(payload, str(out / shard), metadata={"format": "pt"})
        del payload
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if progress is not None:
            progress(position, len(shards), shard, len(units))

    report = ExportReport(
        units=tuple(units),
        passthrough_bytes=passthrough_bytes,
        quantized_bytes=sum(u.exact_bytes for u in units),
        quantized_params=sum(u.params for u in units),
        grid_digest=grid_digest(grid),
    )
    (out / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": report.total_bytes},
         "weight_map": new_weight_map}, indent=2))
    _write_config(out, grid, code, group, half, rotation, with_diagonals,
                  report, plan, extra_config, scale_refit, trellis_weighting, table,
                  activation)
    if copy_aux:
        for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
            for aux in src.glob(pattern):
                if aux.name in {"model.safetensors.index.json"}:
                    continue
                shutil.copy2(aux, out / aux.name)
    return report


def read_checkpoint_config(out_dir: "str | Path") -> dict:
    return json.loads((Path(out_dir) / "tessera_config.json").read_text())


def encode_settings_from_config(config: dict, q256: "int | None" = None) -> dict:
    """The ``encode_linear`` keyword arguments that reproduce a written checkpoint.

    A config with a ``wire.recipes`` table resolves the body, span, plane and
    window parameters for rung ``q256`` from the table; when the rungs the
    checkpoint used (``rungs_q256``) share one recipe the rung may be
    omitted, and when they do not a caller that names no rung is refused
    rather than handed the wrong body.

    A config written before a setting existed means the value the exporter
    had *then*, not the value it defaults to now -- the exporter's defaults
    moved on 2026-09-01 (span 2, LUT plane, refit 4, scale-weighted trellis)
    while the 151 GiB GLM export on disk was built at span 1, S6b, refit 0,
    unweighted, under a config that names none of those fields.  Any replay
    that rebuilds encode arguments from a config goes through here, so a
    missing key resolves to its legacy meaning rather than to today's default.
    ``conv_generators`` is read when present and otherwise resolved from the
    memory order's table, which is what every earlier artifact used.

    A config whose ``activation_aware`` block is not ``null`` is **refused**.
    Those bytes were shaped by a named Hessian, and no dict of encode keywords
    can carry one: handing the caller the weights-only settings would replay
    the unit as a different artifact and raise nothing -- the same silent-RTN
    failure the export path refuses on a missing key.  A caller holding the
    capture builds an ``ActivationSource`` and passes it to the exporter; this
    function is not where a Hessian arrives.
    """
    activation_aware = config.get("activation_aware")
    if activation_aware is not None:
        identity = activation_aware.get("hessian", {})
        raise GrammarError(
            "this checkpoint was encoded activation-aware and cannot be replayed "
            "from its config alone: the bytes depend on the Hessian "
            f"{ {f: identity.get(f) for f in HESSIAN_IDENTITY} }. Rebuild that "
            "capture, wrap it in an ActivationSource and pass it to the exporter."
        )
    memory = int(config.get("conv_memory", DEFAULT_CODE.memory))
    generators = config.get("conv_generators")
    code = (ConvCode(memory=memory, generators=tuple(int(g, 8) for g in generators))
            if generators else ConvCode(memory=memory))
    scale = config.get("scale", {})
    trellis = config.get("trellis", {})
    plane = scale.get("plane", "s6b")
    planes = {name: kind for kind, name in _PLANE_NAMES.items()}
    if plane not in planes and plane != PER_RUNG:
        raise GrammarError(f"unknown scale plane {plane!r} in config")
    body = config.get("body", {"kind": "tcq"})
    kind = body.get("kind", "tcq")
    if kind not in ("tcq", "window", PER_RUNG):
        raise GrammarError(f"unknown body kind {kind!r} in config")
    wire = config.get("wire")
    if wire is not None:
        table = tuple(RecipeRange.from_config(e) for e in wire.get("recipes", ()))
        if not table:
            raise GrammarError("the config's wire.recipes table is empty")
        if q256 is None:
            # Without a rung, the checkpoint's own rungs decide: one recipe
            # across them replays without naming one; several refuse.
            rungs = config.get("rungs_q256") or list(config.get("plan", {}).values())
            recipes = _used_recipes(table, rungs)
            if len(recipes) != 1:
                raise GrammarError(
                    "this checkpoint's recipe varies with the rung "
                    f"({len(recipes)} recipes over its rungs); name the unit's q256 "
                    "to replay it"
                )
            recipe = recipes[0]
        else:
            recipe = recipe_at(table, int(q256))
    else:
        if plane == PER_RUNG:
            raise GrammarError("config says per-rung scale planes but carries no wire.recipes")
        if kind == PER_RUNG:
            raise GrammarError("config says per-rung bodies but carries no wire.recipes")
        channel_sigma = scale.get("sigma")
        sigma = body.get("sigma")
        recipe = WireRecipe(
            body=BodyKind.WINDOW if kind == "window" else BodyKind.TCQ,
            span=int(trellis.get("span", 1)) if kind != "window" else 1,
            scale_plane=planes[plane],
            window_bits=int(body.get("window_bits", 0)),
            window_seed=int(body.get("seed", 0)),
            window_sigma=None if sigma is None else float(sigma),
            channel_sigma=None if channel_sigma is None else float(channel_sigma),
        )
    return dict(
        code=code,
        group=int(scale.get("group", DEFAULT_GROUP)),
        half=int(scale.get("half", DEFAULT_HALF)),
        rotation=RotationState[config.get("rotation", "NONE")],
        with_diagonals=bool(config.get("with_diagonals", False)),
        scale_refit=int(scale.get("refit", 0)),
        trellis_weighting=str(trellis.get("weighting", "none")),
        **_recipe_kwargs(recipe),
    )


def grid_from_config(config: dict) -> PayloadGrid:
    """Resolve the payload grid a config names, and check it against the digest."""
    from .alphabet import BF16_GRID, E2M1_GRID, E4M3_GRID, grid_digest, tuple_grid

    spec = config["grid"]
    base = {"E2M1": E2M1_GRID, "E4M3": E4M3_GRID, "BF16": BF16_GRID}.get(spec.get("base"))
    if base is None:
        raise GrammarError(f"unknown grid base {spec.get('base')!r} in config")
    arity = int(spec.get("arity", 1))
    grid = base if arity == 1 else tuple_grid(base, arity, spec.get("partition", "coset"))
    if grid_digest(grid) != spec["digest"]:
        raise GrammarError(
            f"config names grid {spec.get('name')!r} but its digest does not match "
            "the grid this exporter builds from that name; refusing to replay"
        )
    return grid


def _shard_holding(out: Path, key: str) -> Path:
    """Locate the shard holding ``key``, honouring a written index.

    ``export_checkpoint_streaming`` writes one shard per input shard plus an
    index; only the in-memory ``export_checkpoint`` writes a lone
    ``model.safetensors``.  A reader that assumes the single-file layout can
    read back nothing this format actually exports at scale, so the index is
    consulted first and the single file is the fallback, not the rule.
    """
    index_path = out / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        if key not in weight_map:
            raise KeyError(f"{key!r} is not in this checkpoint's index")
        return out / weight_map[key]
    return out / "model.safetensors"


def load_tessera_weight(
    out_dir: "str | Path", name: str, device: "str | torch.device" = "cpu"
) -> torch.Tensor:
    """Decode one Linear back out of a written checkpoint."""
    from safetensors import safe_open

    out = Path(out_dir)
    config = read_checkpoint_config(out)
    key = name + config.get("blob_suffix", BLOB_SUFFIX)
    with safe_open(str(_shard_holding(out, key)), framework="pt") as handle:
        if key not in handle.keys():
            raise KeyError(f"{name!r} is not a quantized unit in this checkpoint")
        blob = handle.get_tensor(key)
    return read_unit_artifact(bytes(blob.numpy().tobytes()), device=device)


#: The rate schedule and forests ``encode_linear`` hands the encoder for a
#: grid at a rung, under a public name: a render leg that must stay
#: bit-identical to the exporter builds its plan here and nowhere else.
plan_for = _plan_for
