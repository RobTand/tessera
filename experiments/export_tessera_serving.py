#!/usr/bin/env python
"""Export a dense checkpoint as Tessera wires for Tessera's own vLLM plugin.

THREE FAMILIES, ONE PLUGIN.  ``tessera.serving.scheme`` names a family by the
stock tile a module decodes to: ``TESSERA_NVFP4`` (an E2M1-based grid over the
LUT plane, span-2 TCQ body -> the NVFP4 tile, W4A4), ``TESSERA_FP8`` (the
E4M3 grid over the CHANNEL plane, window body -> the per-channel FP8 pair,
W8A8), or ``TESSERA_BF16`` (the BF16 grid over the CHANNEL plane, window body
-> a plain bfloat16 tile, W16A16).  ``--grid``/``--q256`` set the default per
Linear and ``--plan-json`` overrides it per tensor, so one checkpoint may
carry all three and a single serve executes each on its own path: that is the
product an allocator targets.  The 16-bit route exists because the E4M3
*alphabet* -- not the trellis -- floors the window body's error from R=6
upward, so above ~6 bpp an 8-bit tile has nothing left to buy
(``docs/measurements/tessera-bf16-route-2026-09-02.md``).

ALL THREE HAVE A PLUGIN ROUTE.  ``TESSERA_BF16`` was the exception until #9:
``serving/bf16_route.py`` decodes its window body to a plain bfloat16 tile in
both residency modes, ``runtime_contract.json`` v5 attests q256 1792 on
sm_121, and ``--grid BF16`` needs no override.  The ``--stock-twin`` of a BF16
arm is still worth writing -- it is the control the route is measured against,
and the served pair is in the receipt above -- but it is no longer the only
thing that arm can serve.

WHAT THIS SCRIPT MAY WRITE IS WHAT THE PLUGIN PUBLISHES A DECODE FOR.
``check_recipe`` gates the default ``(grid, q256)`` and every ``--plan-json``
override against the packaged ``runtime_contract.json`` before the first
encode, so a wire the pinned runtime cannot read is refused at export rather
than at load (#41).  The encoder is untouched: ``wire_recipe`` still writes
the sub-cap window body and every research encode of it still runs.  The
override is ``--allow-unserveable``, and it is stamped into the manifest.

The checkpoint declares ``quantization_config.quant_method: "tessera"``, which
is what selects the plugin: there is no serve flag to enable it, only
``TESSERA_SERVE_MODE`` to declare the residency.

ONE BLOB PER vLLM MODULE.  Every role a vLLM fusion stacks (q/k/v, gate/up)
is encoded as its own Tessera unit and framed into a ``tessera.fused``
container in stacking order; unfused Linears are a one-member container.  A
fused module's roles must share one family (vLLM builds one method per
module); an NVFP4 module's roles are checked at export for the exact binade
shift the lane applies at load (``shared_lut_global``), so an unserveable
group is refused here, not there.

THE STOCK TWIN.  ``--stock-twin DIR`` also writes the materialisation of the
SAME wires (``tessera.stock.materialize_stock``; NVFP4 groups moved onto one
shared global) that vanilla vLLM serves with no plugin, so a served
comparison is between one encode and its two servings rather than between two
encodes.  A BF16 module's twin tensor is a plain ``<module>.weight`` bfloat16
tile and rides no config group; a twin all of whose modules are BF16 carries
**no** ``quantization_config`` at all, because it is then an ordinary BF16
checkpoint and declaring one would only invite a runtime to look for
tensors that are not there.

The manifest states what is on disk and what the plugin holds resident per
mode.  ``lm_head`` and the embeddings stay BF16 (PrismaQuant's body-only
convention); a Linear whose rows are not a whole number of tuples, or a
fused module not all of whose roles are quantizable, is passed through as
BF16 and named in ``ignore``.  Naming it there is not bookkeeping: the plugin
REFUSES a Linear that is neither declared nor ignored, so one mistyped target
is a refusal rather than a silently BF16 artifact.

THE ARTIFACT IS TENSOR-PARALLELISM-AGNOSTIC, and this exporter never encodes
per rank.  One whole unit per role is written once; a serve with tp_size > 1
loads the whole unit on every rank and cuts it at load
(``tessera.serving.sharding``), so the same checkpoint serves any TP degree and
re-sharding is a serve flag rather than a re-export.  Encoding per rank would
make the bytes a function of the machine they were built for, and a unit cut
for 4 ranks could not be re-cut for 8.

ROUTED-MoE EXPERTS ARE NOT EXPORTED YET, and are refused by name rather than
skipped.  A transformers-5 checkpoint packs a layer's experts into 3-D tensors
(``mlp.experts.gate_up_proj`` ``[E, K, 2N]``, ``...down_proj`` ``[E, N, K]``);
``quantizable`` separates them from the 2-D body and ``main`` either passes
them through as BF16 (named in ``ignore``, so the plugin serves the MoE layer
unquantized instead of refusing it) or, if a plan names one, refuses with the
tensor and its shape.  When the expert route lands, the work is: encode one
unit per expert per role, frame them into one container per vLLM MoE module
(the fused rule below already groups by module name and is rank-agnostic),
declare ``structure: "routed_moe"`` on the scheme, and decode to the stock
PACKED expert layouts vLLM's fused-MoE kernels read.  The grouping rule that a
fused module's roles must share ONE family holds for experts exactly as it does
for q/k/v: vLLM builds one method per module.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_stock_compressed import (  # noqa: E402
    FP8_INPUTS, FP8_WEIGHTS, NVFP4_INPUTS, NVFP4_WEIGHTS, regex_target)
from tessera.alphabet import (  # noqa: E402
    BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid)
from tessera.bf16_route import BF16_FAMILY  # noqa: E402
from tessera.export import (  # noqa: E402
    DEFAULT_CODE, DEFAULT_LDLQ_BLOCK, DEFAULT_LDLQ_SIGMA,
    ActivationSource, encode_linear_planes, wire_recipe)
from tessera.fused import pack_fused, shared_lut_global  # noqa: E402
from tessera.serving.contract import load_serving_contract  # noqa: E402
from tessera.serving.scheme import refuse_unserveable_wire  # noqa: E402
from tessera.stock import materialize_stock, share_global, stock_bytes  # noqa: E402
from tessera.unit_artifact import parse_unit_artifact  # noqa: E402

FUSED = (
    (re.compile(r"^(.*\.self_attn\.)(q_proj|k_proj|v_proj)\.weight$"), "qkv_proj", ("q_proj", "k_proj", "v_proj")),
    # NOT scoped to ``.mlp.``: a shared expert is its own MLP module
    # (``...mlp.shared_experts.{gate,up,down}_proj`` in the checkpoint) and vLLM
    # merges ITS gate/up too -- ``Glm5NextMLP`` takes ``prefix=f"{prefix}.gate_up_proj"``
    # for whatever prefix it is built at (glm5next/nvidia/model.py:124, :216).
    # Naming the unmerged leaves there declares two modules vLLM never builds
    # and leaves the one it does build undeclared, which the plugin refuses at
    # load.  The lookahead keeps ROUTED experts out: their gate/up merge into
    # ``w13`` inside the FusedMoE, which is a different mechanism entirely.
    (re.compile(r"^(?!.*\.experts\.\d+\.)(.*\.)(gate_proj|up_proj)\.weight$"), "gate_up_proj", ("gate_proj", "up_proj")),
)
#: A body Linear, and WHICH decoder layer it belongs to.  Not
#: ``startswith("model.layers.")``: a multimodal checkpoint roots its decoder
#: under a sub-model (GLM-5.3-Flash is ``model.language_model.layers.N.``), and
#: the prefix test silently found NOTHING there -- an "export" that quantized
#: zero Linears and reported success.  The vision tower is ``model.visual.
#: blocks.N.``, which this does not match, so it stays BF16 by the same rule
#: rather than by a second exclusion list.
BODY_LAYER = re.compile(r"^model\.(?:[^.]+\.)*layers\.(\d+)\.")

#: A ROUTED expert leaf in the unpacked (per-expert 2-D) source layout.  The
#: ``\d+`` segment is load-bearing: it is what distinguishes a routed expert
#: from ``mlp.shared_experts.gate_proj``, which is an ordinary dense Linear and
#: is quantized as one.
ROUTED_EXPERT_2D = re.compile(
    r"^(?P<moe>.*\.mlp)\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.weight$")

#: The MoE ROUTER.  ``mlp.gate`` is not a projection (a dense MLP has
#: ``gate_proj``, never ``gate``); it is the little Linear that decides which
#: experts a token visits, and the runtime gives it no quantized route at all.
#:
#: ``GateLinear.__init__`` takes no ``quant_config`` and passes none to
#: ``ReplicatedLinear`` (``layers/fused_moe/router/gate_linear.py:50-58``), so
#: it always gets ``UnquantizedLinearMethod`` and this plugin is NEVER asked
#: about the router.  Encoding it therefore does not produce a slow route or a
#: wrong route: it deletes ``mlp.gate.weight`` from the checkpoint and puts wire
#: tensors in its place that no loader has a mapping for, and the router either
#: fails as an unexpected key or is left with the weight it was born with.
#: Belt and braces, ``GateLinear.forward`` reads ``self.weight`` directly at all
#: six of its dispatch tiers and never calls ``quant_method.apply`` (:179-228),
#: choosing the tier from ``self.weight.dtype`` (:84,:101,:128,:165,:174) -- so
#: even a method that WAS installed would be dead code.
#:
#: That is a route status, not a taste: on the pinned build the runtime has no
#: path that executes these bytes for this module, which is the one carve-out
#: principle 9 allows.  Attested against
#: ``prismaquant/glm53-mia-sm121:487ecf187``.
MOE_ROUTER = re.compile(r"^(?P<moe>.*\.mlp)\.(?:gate|router)\.weight$")

#: A PACKED expert stack: one rank-3 tensor holding every expert of a layer.
#: Rank alone does NOT identify one -- GLM-5.3-Flash's attention carries
#: ``k_conv1d.weight [8192, 1, 4]``, and treating that as an expert stack put
#: ``...self_attn`` (the whole attention block, every Linear in it) into the
#: checkpoint's ``ignore`` list.  A tensor is a packed expert stack because of
#: where it sits, not because it has three axes.
PACKED_EXPERT_ND = re.compile(
    r"^(?P<moe>.*\.mlp)\.experts\.(?P<proj>gate_up_proj|down_proj|gate_proj|up_proj)\.weight$")

NVFP4 = "TESSERA_NVFP4"
FP8 = "TESSERA_FP8"
BF16 = BF16_FAMILY


def grid_for(name: str):
    if name == "E4M3":
        return E4M3_GRID
    if name == "BF16":
        return BF16_GRID
    match = re.fullmatch(r"(E2M1)(?:x(\d+))?", name)
    if not match:
        raise SystemExit(
            f"unknown grid {name!r}; one of E2M1, E2M1x2 (NVFP4 route), E4M3 "
            "(FP8 route) or BF16 (the 16-bit route)")
    return tuple_grid(E2M1_GRID, int(match.group(2) or 1))


def family_for(grid) -> str:
    if grid.name == "BF16":
        return BF16
    return FP8 if grid.name == "E4M3" else NVFP4


def check_recipe(grid, q256: int, where: "str | None" = None, *,
                 allow_unserveable: bool = False, overrides: "list | None" = None):
    """The recipe this plugin build publishes a decode for, or a refusal.

    THE SEAM IS HERE, and it is the export-time half of the load-time gate.
    ``refuse_unserveable_wire`` lives in ``tessera.serving.scheme`` -- beside
    ``validate_tessera_scheme``, off the same ``ROUTES`` table and the same
    packaged ``runtime_contract.json``, importable without torch -- so the
    producer and the consumer read one authority rather than two opinions.
    What lives HERE is only the call, and it is placed before the first encode:
    the plan's default grid and every ``--plan-json`` override are checked at
    argument time, which is every path into the encode loop.  A refusal that
    arrives after the encode is not a refusal; it is a bill.

    It refuses at the SERVING boundary and nowhere else.  ``wire_recipe`` and
    ``encode_linear`` keep their full range: sub-cap ``E2M1x2`` is the rate
    frontier's own body of work and every research encode of it still runs.
    This script is the one that writes ``quant_method: "tessera"``, so this is
    where the pinned runtime's reach becomes a constraint.

    IT FAILS CLOSED WITH AN EXPLICIT, STAMPED OVERRIDE, which is the shape
    principle 9 gives this carve-out: refuse unless the run carries an explicit
    per-run override, and stamp it on the artifact.  ``--allow-unserveable``
    is that override, and it exists because real workflows need it: ``--grid
    E2M1``, whose route holds the grid while the contract publishes no measured
    range for it, and sub-cap ``E2M1x2``, which the rate frontier encodes
    constantly.  The reason to export either is the ``--stock-twin``, which
    vanilla vLLM serves with no plugin at all.  (``--grid BF16`` was the
    original example and is no longer one: the 16-bit route (#9) decodes it,
    so it passes the gate rather than needing the override.)  Overridden refusals land verbatim in the manifest's
    ``serving_gate`` block, so an artifact that cannot be served says so in its
    own bytes rather than in a shell history.
    """
    recipe = wire_recipe(grid, q256)
    target = where or f"--grid {grid.name} --q256 {q256}"
    try:
        refuse_unserveable_wire(grid.name, q256, recipe.body.name, recipe.scale_plane.name,
                                span=recipe.span, target=target)
    except ValueError as exc:
        if not allow_unserveable:
            raise SystemExit(
                f"{exc}\n\nThis script writes a checkpoint that declares quant_method "
                "\"tessera\", so a wire the plugin cannot decode is refused HERE rather than "
                "hours later at load. Pass --allow-unserveable to write it anyway as a research "
                "artifact: the refusal is then stamped verbatim into the manifest's "
                "serving_gate block, and the checkpoint still will not load under this "
                "plugin build.") from exc
        print(f"  --allow-unserveable: writing a wire this plugin build cannot decode. {exc}",
              flush=True)
        if overrides is not None:
            overrides.append({"target": target, "grid": grid.name, "q256": int(q256),
                              "body": recipe.body.name, "plane": recipe.scale_plane.name,
                              "refusal": str(exc)})
    return recipe


def module_of(tensor_name: str) -> str:
    return tensor_name[: -len(".weight")]


def fused_module(tensor_name: str):
    """``(fused module name, ordered member tensor names)`` or ``None``."""
    for pattern, fused, members in FUSED:
        match = pattern.match(tensor_name)
        if match:
            return match.group(1) + fused, tuple(match.group(1) + m + ".weight" for m in members)
    return None


def git_hash() -> str:
    """The commit this build came from -- from git, or from the environment.

    A build that runs on a synced copy of the tree has no ``.git`` and used to
    stamp ``unknown``, which is a provenance hole in an artifact whose whole
    claim is that the surrogate, the KL and the bytes are one rendering.
    ``TESSERA_GIT`` is how the caller supplies it when git cannot.
    """
    import os

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        return os.environ.get("TESSERA_GIT", "unknown")


def quantizable(src: Path):
    """The body's ``.weight`` tensors, split by WHAT KIND OF LAYER OWNS THEM.

    Returns ``(shards, shapes, expert_shapes, routed_shapes)``:

    * ``shapes`` -- every 2-D dense-Linear weight, one blob per vLLM module.
      Shared experts live here: ``mlp.shared_experts.gate_proj`` is a plain
      ``Glm5NextMLP`` Linear, not a routed unit.
    * ``expert_shapes`` -- every ``.weight`` of rank 3 or more, a
      transformers-5 PACKED expert stack.
    * ``routed_shapes`` -- 2-D routed-expert leaves, the UNPACKED per-expert
      layout (``...mlp.experts.7.gate_proj.weight``).

    The third split is the one that has to exist.  These are 2-D, so without
    it they land in ``shapes`` and are planned as ordinary dense Linears --
    864 of them per projection on GLM-5.3-Flash -- and the export succeeds,
    hours later, into a checkpoint whose ``config_groups`` name modules that
    vLLM does not build, which the plugin then refuses at LOAD.  A refusal
    that arrives after the encode is not a refusal; it is a bill.
    """
    index = src / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for tensor, shard in weight_map.items():
            shards.setdefault(shard, []).append(tensor)
    else:
        shards = {}
        for path in sorted(src.glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                shards[path.name] = list(handle.keys())
    shapes, expert_shapes, routed_shapes = {}, {}, {}
    for shard, names in shards.items():
        with safe_open(str(src / shard), framework="pt") as handle:
            for name in names:
                if not (name.endswith(".weight") and BODY_LAYER.match(name)):
                    continue
                shape = tuple(handle.get_slice(name).get_shape())
                if len(shape) >= 3:
                    # Only an expert stack by NAME; anything else of rank 3 is
                    # not a Linear at all (a conv1d), so it is not a Linear the
                    # plugin must be told about and needs no ignore entry.
                    if PACKED_EXPERT_ND.match(name):
                        expert_shapes[name] = shape
                elif len(shape) == 2:
                    (routed_shapes if ROUTED_EXPERT_2D.match(name) else shapes)[name] = shape
    return shards, shapes, expert_shapes, routed_shapes


def body_layer(name: str) -> int:
    """The decoder-layer index owning this tensor."""
    match = BODY_LAYER.match(name)
    if match is None:
        raise SystemExit(f"{name} is not a body tensor; BODY_LAYER should have filtered it out")
    return int(match.group(1))


def packed_expert_orientation(name: str, shape, config: dict):
    """Which axis of a PACKED expert stack is the output, from the config.

    ``[E, A, B]`` is ambiguous on its face and the conventions genuinely
    differ across transformers-5 architectures (``gate_up_proj`` appears both
    as ``[E, hidden, 2*inter]`` and as ``[E, 2*inter, hidden]``).  So the
    answer is read off ``hidden_size``/``moe_intermediate_size`` rather than
    assumed -- and REFUSED when the dims cannot decide it.

    That refusal is not hypothetical.  On GLM-5.3-Flash
    ``hidden_size == 4096`` and ``moe_intermediate_size == 2048``, so a packed
    ``gate_up_proj`` would be ``[E, 4096, 4096]`` and NO dim comparison can
    orient it.  A guess there transposes every expert in silence.
    """
    text = config.get("text_config", config)
    hidden = text.get("hidden_size")
    inter = text.get("moe_intermediate_size", text.get("intermediate_size"))
    if hidden is None or inter is None:
        raise SystemExit(f"cannot orient the packed expert stack {name} {list(shape)}: "
                         "config.json declares no hidden_size/moe_intermediate_size")
    _experts, a, b = shape
    gate_up = name.endswith("gate_up_proj.weight") or name.endswith("gate_up_proj")
    out_dim = 2 * inter if gate_up else hidden
    in_dim = hidden if gate_up else inter
    a_is_out, b_is_out = (a == out_dim and b == in_dim), (b == out_dim and a == in_dim)
    if a_is_out and b_is_out:
        raise SystemExit(
            f"cannot orient the packed expert stack {name} {list(shape)}: both axis orders fit "
            f"(out={out_dim}, in={in_dim}). Re-export this checkpoint unpacked, or teach the "
            "exporter this architecture's convention explicitly -- guessing transposes every "
            "expert silently.")
    if a_is_out:
        return "out_first"
    if b_is_out:
        return "in_first"
    raise SystemExit(
        f"cannot orient the packed expert stack {name} {list(shape)}: neither axis order fits "
        f"hidden_size={hidden} moe_intermediate_size={inter} (expected out={out_dim}, in={in_dim})")


def stock_targets(modules):
    """compressed-tensors targets for member modules plus their fused names (the stock exporter's rule)."""
    found = sorted(set(modules))
    fused_names = set()
    by_prefix: dict[str, set[str]] = {}
    for m in found:
        for pattern, fused, member_names in FUSED:
            match = pattern.match(m + ".weight")
            if match:
                by_prefix.setdefault(match.group(1) + fused, set()).add(match.group(2))
    for fused_name, present in by_prefix.items():
        for pattern, fused, member_names in FUSED:
            if fused_name.endswith(fused) and present == set(member_names):
                fused_names.add(fused_name)
    return [regex_target(m) for m in sorted(set(found) | fused_names)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--grid", default="E2M1x2",
                    help="default grid per Linear: E2M1, E2M1x2 (NVFP4), E4M3 (FP8) or BF16 (W16A16)")
    ap.add_argument("--q256", type=int, default=896, help="default body bits per 256 weights")
    ap.add_argument("--plan-json", type=Path, default=None,
                    help='{"tensor.weight": {"grid": "E4M3"|"BF16", "q256": 1024} | "PASSTHROUGH", ...} per-tensor overrides')
    ap.add_argument("--input-scales", type=Path, default=None,
                    help="safetensors carrying <module>.input_global_scale per NVFP4 Linear (a stock NVFP4 "
                         "export); required when any module takes the NVFP4 route")
    ap.add_argument("--stock-twin", type=Path, default=None,
                    help="also write the compressed-tensors materialisation of the same wires here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="encode only the first N layers (smoke)")
    ap.add_argument("--hessian", type=Path, default=None,
                    help="capture_h_full.py payload: full input Hessians keyed by the tensor's module "
                         "name.  Enables the activation-aware encoder settings below; an encode that "
                         "uses them is not reproducible from the weights alone, so the file's own "
                         "provenance is copied into the manifest.")
    ap.add_argument("--ldlq-sigma", type=float, default=DEFAULT_LDLQ_SIGMA,
                    help="Hessian regulariser for LDLQ cross-column feedback; a negative value turns LDLQ off")
    ap.add_argument("--ldlq-block", type=int, default=DEFAULT_LDLQ_BLOCK, help="LDLQ input-feature block")
    ap.add_argument("--refit-metric", default=None,
                    help="error the scale refit minimises: plain | hessian | h^ALPHA. "
                         "Default: the measured objective for each unit's own scale "
                         "plane (export.DEFAULT_REFIT_OBJECTIVE), which is not one "
                         "value -- the exact quadratic on the CHANNEL plane, the "
                         "diagonal h^1.0 on the LUT plane")
    ap.add_argument("--refit-reach-floor", action="store_true",
                    help="hold every refit row scale high enough that the pass's target stays inside the body's reach")
    ap.add_argument("--allow-unserveable", action="store_true",
                    help="write wires this plugin build publishes no decode for (see check_recipe). "
                         "The checkpoint is then a RESEARCH artifact that will not load under this "
                         "plugin; every refusal is stamped verbatim into the manifest's "
                         "serving_gate block. Needed today by --grid BF16, whose wire has no "
                         "plugin route and whose --stock-twin is what gets served.")
    args = ap.parse_args()

    # The activation-aware settings fire when, and only when, a Hessian is
    # here: the encoder cannot invent one, and a weights-only export must stay
    # the byte-for-byte artifact it was.  Given one, the defaults are the
    # measured recipe (export.DEFAULT_LDLQ_*), overridable per run.  The recipe
    # itself lives in ``ActivationSource``, not here: this script and the
    # library exporters must not carry two copies of it, which is the drift
    # that let the library path encode weights-only while the script did not.
    activation = None
    if args.hessian:
        settings = {"ldlq_sigma": args.ldlq_sigma, "ldlq_block": args.ldlq_block,
                    "refit_reach_floor": args.refit_reach_floor}
        if args.refit_metric is not None:      # else: the measured per-plane map
            settings["refit_objective"] = args.refit_metric
        activation = ActivationSource.from_capture(args.hessian, **settings)

    default_grid = grid_for(args.grid)
    # Every path into the encode loop passes through here: a tensor takes the
    # default (grid, q256) or a --plan-json override, and both are gated before
    # a single unit is encoded.
    gate_overrides: list = []
    check_recipe(default_grid, args.q256,
                 allow_unserveable=args.allow_unserveable, overrides=gate_overrides)
    overrides = {}
    if args.plan_json:
        for name, spec in json.loads(args.plan_json.read_text()).items():
            if spec in ("BF16", "PASSTHROUGH"):
                # The bare string is PASSTHROUGH -- copy the source tensor,
                # quantise nothing.  ``{"grid": "BF16", "q256": N}`` is the
                # opposite: the 16-bit trellis route at rate N, whose tile is
                # bf16 but whose wire is 4-8 bpp.  ``"BF16"`` is kept as a
                # spelling of passthrough because plans were written with it;
                # ``"PASSTHROUGH"`` is the one that cannot be misread.
                overrides[name] = None
            else:
                g = grid_for(spec["grid"])
                check_recipe(g, int(spec["q256"]), where=name,
                             allow_unserveable=args.allow_unserveable, overrides=gate_overrides)
                overrides[name] = (g, int(spec["q256"]))
    src_config = json.loads((args.src / "config.json").read_text())
    shards, shapes, expert_shapes, routed_shapes = quantizable(args.src)
    if not shapes and not expert_shapes and not routed_shapes:
        raise SystemExit(
            f"no body weight tensors found under {args.src}. BODY_LAYER matches "
            f"``model.<...>.layers.<N>.``; this checkpoint's names do not, so there is nothing "
            "to export rather than nothing to do.")

    # A routed expert must be refused HERE -- before a single unit is encoded.
    # These leaves are 2-D, so the only thing standing between them and being
    # planned as dense Linears is this check.
    planned_routed = sorted(set(overrides) & set(routed_shapes))
    if planned_routed:
        first = planned_routed[0]
        raise SystemExit(
            f"the plan names {len(planned_routed)} ROUTED expert tensor(s), e.g. {first} "
            f"{list(routed_shapes[first])}. A routed expert is not a dense Linear: vLLM builds "
            "one FusedMoE module per layer, not one Linear per expert, so a checkpoint declaring "
            f"{module_of(first)} in config_groups names a module vLLM never creates and the "
            "plugin refuses it at load. Routed-MoE export is declared in its own `moe` block. "
            "Remove them from the plan to pass them through as BF16.")
    planned_routers = sorted(n for n in overrides if MOE_ROUTER.match(n))
    if planned_routers:
        raise SystemExit(
            f"the plan names {len(planned_routers)} MoE ROUTER tensor(s), e.g. "
            f"{planned_routers[0]}. vLLM builds the router as GateLinear, which takes no "
            "quant_config at all, so it always gets UnquantizedLinearMethod and this plugin is "
            "never asked about it: the wire would replace mlp.gate.weight with tensors no loader "
            "maps, and the routing weight would be missing rather than quantized. (Its forward "
            "also reads self.weight directly at every dispatch tier and never calls "
            "quant_method.apply, so even an installed method would be dead code.) The pinned "
            "runtime has no route for these bytes here. Remove them from the plan to pass the "
            "router through as BF16.")
    planned_experts = sorted(set(overrides) & set(expert_shapes))
    if planned_experts:
        first = planned_experts[0]
        raise SystemExit(
            f"the plan names {len(planned_experts)} packed expert tensor(s), e.g. {first} "
            f"{list(expert_shapes[first])} (orientation "
            f"{packed_expert_orientation(first, expert_shapes[first], src_config)}): routed-MoE "
            "expert export is a follow-up. The wires, the container framing and the scheme's "
            "structure field are all in place; what is missing is the per-expert encode and the "
            "decode to vLLM's packed expert layouts. Remove them from the plan to pass them "
            "through as BF16.")
    unknown = sorted(set(overrides) - set(shapes))
    if unknown:
        raise SystemExit(f"plan names tensors that are not 2-D body weights here: {unknown[:5]}")
    if expert_shapes:
        print(f"  {len(expert_shapes)} packed expert tensors stay BF16 and are named in ignore "
              f"(routed-MoE export is a follow-up); e.g. {sorted(expert_shapes)[0]}", flush=True)
    if routed_shapes:
        layers = sorted({body_layer(n) for n in routed_shapes})
        print(f"  {len(routed_shapes)} routed expert tensors across layers {layers} stay BF16 and "
              f"are named in ignore (routed-MoE export is a follow-up); "
              f"e.g. {sorted(routed_shapes)[0]}", flush=True)
    plan: dict[str, tuple] = {}          # tensor -> (grid, q256, rows, cols)
    passthrough: list[str] = []
    for name, (rows, cols) in shapes.items():
        layer = body_layer(name)
        if args.layers is not None and layer >= args.layers:
            passthrough.append(name); continue
        if name in overrides and overrides[name] is None:
            passthrough.append(name); continue
        if MOE_ROUTER.match(name):
            passthrough.append(name); continue
        grid, q256 = overrides.get(name, (default_grid, args.q256))
        if rows % (grid.arity * 32) or cols % 16:
            passthrough.append(name); continue
        plan[name] = (grid, q256, rows, cols)
    # group by vLLM module: a fused module needs every role quantizable on ONE recipe
    modules: dict[str, list[str]] = {}
    for name in list(plan):
        fused = fused_module(name)
        if fused is None:
            modules[module_of(name)] = [name]
            continue
        key, members = fused
        if key in modules:
            continue
        recipes = {(plan[m][0].name, plan[m][1]) for m in members if m in plan}
        if all(m in plan for m in members) and len(recipes) == 1:
            modules[key] = list(members)
        else:
            why = "not every role is quantizable" if not all(m in plan for m in members) else f"roles disagree {sorted(recipes)}"
            print(f"  passthrough {key}: {why}; vLLM builds one method per fused module", flush=True)
            for m in members:
                if m in plan:
                    del plan[m]
                    passthrough.append(m)
    owned = {m for members in modules.values() for m in members}
    assert owned == set(plan)

    input_scales = {}
    if args.input_scales:
        with safe_open(str(args.input_scales), framework="pt") as handle:
            for key in handle.keys():
                if key.endswith(".input_global_scale"):
                    input_scales[key] = float(handle.get_tensor(key).float().reshape(-1)[0])
    needs_scales = [m for m, members in modules.items() if family_for(plan[members[0]][0]) == NVFP4]
    if needs_scales and not input_scales:
        raise SystemExit(f"{len(needs_scales)} modules take the NVFP4 route (W4A4 needs a static input scale) "
                         "but no --input-scales was given")

    args.out.mkdir(parents=True, exist_ok=True)
    twin = args.stock_twin
    if twin is not None:
        twin.mkdir(parents=True, exist_ok=True)
    started = time.time()
    new_weight_map: dict[str, str] = {}
    twin_weight_map: dict[str, str] = {}
    config_groups: dict[str, dict] = {}
    units: dict[str, dict] = {}
    module_records: dict[str, dict] = {}
    twin_modules: dict[str, list[str]] = {NVFP4: [], FP8: [], BF16: []}
    twin_records: dict[str, dict] = {}
    ignore = ["lm_head", "model.embed_tokens"]
    passthrough_bytes = 0
    weights_cache: dict[str, torch.Tensor] = {}
    done = 0
    total = len(plan)

    pending_modules = dict(modules)
    for shard, names in sorted(shards.items()):
        shard_payload: dict[str, torch.Tensor] = {}
        twin_payload: dict[str, torch.Tensor] = {}
        with safe_open(str(args.src / shard), framework="pt") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if name in plan:
                    weights_cache[name] = tensor
                else:
                    shard_payload[name] = tensor
                    twin_payload[name] = tensor
                    passthrough_bytes += tensor.numel() * tensor.element_size()
                    if name in passthrough:
                        # ``ignore`` is read against vLLM's OWN Linear names, and
                        # vLLM builds one Linear per FUSED module: ignoring
                        # q_proj/k_proj/v_proj leaves ``qkv_proj`` neither
                        # declared nor ignored, and the plugin refuses that
                        # checkpoint at load.  A passed-through role therefore
                        # ignores the fused module it belongs to.
                        fused_here = fused_module(name)
                        ignore.append(fused_here[0] if fused_here else module_of(name))
        for module, members in list(pending_modules.items()):
            if not all(m in weights_cache for m in members):
                continue
            grid, q256, _r, _c = plan[members[0]]
            family = family_for(grid)
            recipe = wire_recipe(grid, q256)
            roles = []
            role_records = []
            stock_tensors: dict[str, dict] = {}
            for member in members:
                weight = weights_cache.pop(member).to(args.device, torch.float32).contiguous()
                # A missing key renders RTN and raises nothing; ``for_unit``
                # refuses instead, and is the same call the library exporters make.
                extra = ({} if activation is None else
                         activation.for_unit(member, weight.shape[1], args.device,
                                             scale_plane=recipe.scale_plane))
                exported, unit, forests = encode_linear_planes(
                    weight, grid=grid, q256=q256, name=member, verify=not args.no_verify, **extra)
                extra.clear()
                parse_unit_artifact(exported.blob, device=args.device)      # the reader accepts what we wrote
                role = module_of(member).rsplit(".", 1)[-1]
                roles.append((role, exported.rows, exported.blob, unit, forests))
                stock_tensors[member] = materialize_stock(unit, forests, DEFAULT_CODE)
                role_records.append({
                    "tensor": member, "role": role, "rows": exported.rows, "cols": exported.columns,
                    "grid": grid.name, "q256": q256, "family": family,
                    "wire_bytes": exported.exact_bytes, "blob_bytes": len(exported.blob),
                    "wire_bpp": float(exported.bpp), "own_global": float(unit.scale_global),
                    "resident_bytes_stock": stock_bytes(stock_tensors[member]),
                })
                done += 1
                if done % 20 == 0 or done == total:
                    print(f"  [{done}/{total}] {member}  {time.time() - started:.0f}s", flush=True)
            blob = pack_fused([(r, rows, b) for r, rows, b, _u, _f in roles])
            rows_total = sum(r[1] for r in roles)
            cols = role_records[0]["cols"]
            record = {
                "family": family, "grid": grid.name, "q256": q256, "roles": role_records,
                "container_bytes": len(blob), "rows": rows_total, "cols": cols,
                "wire_bytes": sum(r["wire_bytes"] for r in role_records),
            }
            shard_payload[f"{module}.wire_bytes"] = torch.frombuffer(bytearray(blob), dtype=torch.uint8).clone()
            if family == NVFP4:
                shared, _moved = shared_lut_global(
                    [u.scale_lut for _, _, _, u, _ in roles], [float(u.scale_global) for _, _, _, u, _ in roles],
                    [r for r, *_ in roles])
                # the A-side static scale: vLLM's NVFP4 scheme takes the MAX over fused shards
                scale_keys = [f"{module_of(m)}.input_global_scale" for m in members]
                missing = [k for k in scale_keys if k not in input_scales]
                if missing:
                    raise SystemExit(f"no {missing} in {args.input_scales}; W4A4 cannot serve {module}")
                a_scale = max(input_scales[k] for k in scale_keys)
                shard_payload[f"{module}.trellis_input_global_scale"] = torch.tensor([a_scale], dtype=torch.float32)
                record.update({"shared_global": shared, "input_global_scale": a_scale,
                               "resident_bytes_resident_mode": rows_total * cols // 2 + rows_total * cols // 16})
                if twin is not None:
                    moved, divisor = share_global({module_of(m): stock_tensors[m] for m in members})
                    for m in members:
                        for key, value in moved[module_of(m)].items():
                            twin_payload[f"{module_of(m)}.{key}"] = value.cpu()
                        twin_payload[f"{module_of(m)}.input_global_scale"] = torch.tensor(
                            [input_scales[f"{module_of(m)}.input_global_scale"]], dtype=torch.float32)
                    record["twin_shared_divisor"] = divisor
            elif family == BF16:
                # Resident here is the DECODED tile -- 16 bits a weight, the
                # source precision.  It is the correctness path and not a size
                # claim; the product mode is streamed, and the wire it streams
                # is ``wire_bytes`` above.
                record["resident_bytes_resident_mode"] = rows_total * cols * 2
                if twin is not None:
                    for m in members:
                        # One tensor, under the ORIGINAL name: the twin is an
                        # ordinary BF16 checkpoint, not a compressed one.
                        twin_payload[m] = stock_tensors[m]["weight"].cpu()
            else:
                record["resident_bytes_resident_mode"] = rows_total * cols + rows_total * 4
                if twin is not None:
                    for m in members:
                        for key, value in stock_tensors[m].items():
                            twin_payload[f"{module_of(m)}.{key}"] = value.cpu()
            if twin is not None:
                twin_modules[family].extend(module_of(m) for m in members)
                twin_records[module] = {"family": family, "members": [module_of(m) for m in members],
                                        "resident_bytes": sum(stock_bytes(stock_tensors[m]) for m in members)}
            scheme = {
                # ``structure`` names what kind of vLLM layer this is: ``dense``
                # (a LinearBase, one blob per module) today.  The expert route
                # will declare ``routed_moe`` here rather than change the schema.
                "family": family, "structure": "dense",
                "grid": grid.name, "body": recipe.body.name, "plane": recipe.scale_plane.name,
                "q256": q256, "rows": rows_total, "columns": cols, "wire_bytes": len(blob),
                "roles": [[r, rows] for r, rows, *_ in roles],
            }
            config_groups[f"tessera_{module.replace('.', '_')}"] = {"format": "TESSERA", "targets": [module], "scheme": scheme}
            module_records[module] = record
            for r in role_records:
                units[r["tensor"]] = r
            del pending_modules[module]
        save_file({k: v.contiguous() for k, v in shard_payload.items()}, str(args.out / shard), metadata={"format": "pt"})
        for key in shard_payload:
            new_weight_map[key] = shard
        print(f"wrote {shard}: {len(shard_payload)} tensors", flush=True)
        if twin is not None:
            save_file({k: v.contiguous() for k, v in twin_payload.items()}, str(twin / shard), metadata={"format": "pt"})
            for key in twin_payload:
                twin_weight_map[key] = shard
            print(f"wrote twin {shard}: {len(twin_payload)} tensors", flush=True)
    if pending_modules:
        raise SystemExit(f"modules never completed: {sorted(pending_modules)}")

    # A packed expert stack stays BF16; naming its MODULE in ``ignore`` is what
    # lets the plugin serve that MoE layer unquantized instead of refusing it.
    for name in expert_shapes:
        ignore.append(module_of(name).rsplit(".", 1)[0])
    # An UNPACKED routed expert is ignored at the FusedMoE's OWN prefix, not at
    # its 2592 checkpoint leaves.  Attested against the pinned build
    # ``prismaquant/glm53-mia-sm121:487ecf187``, three hops:
    #   models/glm5next/nvidia/model.py:239  FusedMoEFactory(prefix=f"{prefix}.experts")
    #   layers/fused_moe/layer.py:221        layer_name = prefix
    #   layers/fused_moe/routed_experts.py:122,:201
    #                                        quant_config.get_quant_method(self, self.layer_name)
    # So the string the plugin tests is ``<layer>.mlp.experts`` and no leaf name
    # is ever offered to it.  Naming the parent cannot reach the shared experts
    # beside it: ``shared_experts`` is a SIBLING of ``experts``, and both the
    # plugin's test and compressed-tensors' are exact/fnmatch, not prefix
    # subsumption.
    for name in routed_shapes:
        ignore.append(ROUTED_EXPERT_2D.match(name).group("moe") + ".experts")
    ignore = sorted(set(ignore))
    config = src_config
    config["quantization_config"] = {
        # The field that selects Tessera's own vLLM plugin (entry point
        # ``tessera = tessera.serving:register``).  No serve flag enables it.
        "quant_method": "tessera", "format": "mixed-precision",
        "config_groups": config_groups, "ignore": ignore,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    if len(shards) > 1:
        size = sum((args.out / s).stat().st_size for s in shards)
        (args.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": size}, "weight_map": new_weight_map}, indent=2))
    aux_patterns = ("*.json", "*.txt", "*.jinja", "*.model")
    for pattern in aux_patterns:
        for aux in args.src.glob(pattern):
            if aux.name in ("config.json", "model.safetensors.index.json"):
                continue
            shutil.copy2(aux, args.out / aux.name)
            if twin is not None:
                shutil.copy2(aux, twin / aux.name)

    by_family = {}
    for fam in (NVFP4, FP8, BF16):
        recs = [m for m in module_records.values() if m["family"] == fam]
        if not recs:
            continue
        params = sum(r["rows"] * r["cols"] for m in recs for r in m["roles"])
        wire = sum(m["wire_bytes"] for m in recs)
        resident = sum(m["resident_bytes_resident_mode"] for m in recs)
        by_family[fam] = {
            "modules": len(recs), "units": sum(len(m["roles"]) for m in recs), "quantized_params": params,
            "wire_bytes": wire, "wire_bpp": float(Fraction(wire * 8, params)),
            "resident_mode_bytes": resident, "resident_mode_bpp": float(Fraction(resident * 8, params)),
        }
    params = sum(r["rows"] * r["cols"] for r in units.values())
    wire = sum(r["wire_bytes"] for r in units.values())
    on_disk = sum(m["container_bytes"] for m in module_records.values())
    resident = sum(m["resident_bytes_resident_mode"] for m in module_records.values())
    totals = {
        "quantized_params": params, "modules": len(module_records), "units": len(units),
        "wire_bytes": wire, "wire_bpp": float(Fraction(wire * 8, params)) if params else None,
        "on_disk_bytes": on_disk, "on_disk_bpp": float(Fraction(on_disk * 8, params)) if params else None,
        "resident_mode_bytes": resident, "resident_mode_bpp": float(Fraction(resident * 8, params)) if params else None,
        "streamed_mode_note": "the prepared planes (~wire bytes + per-unit tables) plus one transient decoded tile per forward",
        "by_family": by_family,
        "passthrough_bytes": passthrough_bytes,
        "checkpoint_bytes": sum((args.out / s).stat().st_size for s in shards),
    }
    families = sorted({m["family"] for m in module_records.values()})
    manifest = {
        "source": str(args.src), "git": git_hash(), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arm": f"tessera {default_grid.name} q256={args.q256}" + (f" + plan {args.plan_json}" if args.plan_json else "")
               + f" -> tessera.serving {'+'.join(families)}",
        "default": {"grid": default_grid.name, "q256": args.q256}, "plan_json": str(args.plan_json) if args.plan_json else None,
        "input_scales_from": str(args.input_scales) if args.input_scales else None,
        "activation_aware": None if activation is None else activation.config_block(),
        # What the SERVING gate decided, in the artifact rather than in a shell
        # history: which contract version bounded it, whether the run carried
        # the override, and the verbatim refusal for every wire written anyway.
        # An empty list is the shippable state; a non-empty one says this
        # checkpoint will not load under the plugin build named here.
        "serving_gate": {
            "contract": "tessera/serving/runtime_contract.json",
            "contract_version": load_serving_contract()["contract_version"],
            "allow_unserveable": bool(args.allow_unserveable),
            "unserveable_overrides": gate_overrides,
        },
        "stock_twin": str(twin) if twin is not None else None,
        "totals": totals, "modules": module_records,
    }
    (args.out / "tessera_serving_manifest.json").write_text(json.dumps(manifest, indent=2))

    if twin is not None:
        twin_groups = {}
        if twin_modules[NVFP4]:
            twin_groups[f"group_{len(twin_groups)}"] = {
                "format": "nvfp4-pack-quantized", "weights": dict(NVFP4_WEIGHTS),
                "input_activations": dict(NVFP4_INPUTS), "targets": stock_targets(twin_modules[NVFP4])}
        if twin_modules[FP8]:
            twin_groups[f"group_{len(twin_groups)}"] = {
                "format": "float-quantized", "weights": dict(FP8_WEIGHTS),
                "input_activations": dict(FP8_INPUTS), "targets": stock_targets(twin_modules[FP8])}
        twin_config = json.loads((args.src / "config.json").read_text())
        if twin_groups:
            # A BF16 module is in no group: its twin tensor is the ordinary
            # ``<module>.weight``, so it is ignored by the quantization config
            # exactly as ``lm_head`` is.
            twin_config["quantization_config"] = {
                "quant_method": "compressed-tensors", "format": "mixed-precision",
                "config_groups": twin_groups,
                "ignore": sorted(set(ignore) | set(twin_modules[BF16])),
                "quantization_status": "compressed",
            }
        else:
            # Every module decoded to a plain bf16 tile: this IS a BF16
            # checkpoint.  Declaring an empty config would tell a runtime to
            # look for compressed tensors that do not exist.
            twin_config.pop("quantization_config", None)
        (twin / "config.json").write_text(json.dumps(twin_config, indent=2))
        if len(shards) > 1:
            size = sum((twin / s).stat().st_size for s in shards)
            (twin / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": size}, "weight_map": twin_weight_map}, indent=2))
        twin_resident = sum(r["resident_bytes"] for r in twin_records.values())
        (twin / "tessera_stock_twin_manifest.json").write_text(json.dumps({
            "source": str(args.src), "git": git_hash(), "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "wire_checkpoint": str(args.out), "arm": manifest["arm"] + " (stock twin of the same wires)",
            "totals": {"quantized_params": params, "modules": len(twin_records),
                       "resident_bytes": twin_resident,
                       "resident_bpp": float(Fraction(twin_resident * 8, params)) if params else None,
                       "checkpoint_bytes": sum((twin / s).stat().st_size for s in shards)},
            "modules": twin_records,
        }, indent=2))
    print(json.dumps(totals, indent=2))
    print(f"elapsed {time.time() - started:.0f}s -> {args.out}" + (f" (twin -> {twin})" if twin is not None else ""))


if __name__ == "__main__":
    main()
