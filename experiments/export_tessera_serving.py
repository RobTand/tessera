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
fused module's roles must share one family -- vLLM builds one method per
module -- and with it the grid, body and scale plane that method's route
decodes (``module_scheme_key``).  They need NOT share a RATE: every decoder in
``tessera.serving`` reads a role from that role's own manifest, so a fused
group whose members took different rungs is written as one container with a
per-role ``q256`` list in its scheme, and the runtime publishes that rule as a
value a producer's group allocator reads (``runtime_contract.json``'s
``fused_module``, contract v6, #37).  An NVFP4 module's roles are additionally
checked at export for the exact binade shift the lane applies at load
(``shared_lut_global``), so an unserveable group is refused here, not there.

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
is a refusal rather than a silently BF16 artifact.  The naming follows the
tensors WRITTEN, not the tensors the body pattern matched
(``ignored_modules``), so a Linear outside the decoder body -- a vision tower,
an MTP sidecar -- is named too.

THE ARTIFACT IS TENSOR-PARALLELISM-AGNOSTIC, and this exporter never encodes
per rank.  One whole unit per role is written once; a serve with tp_size > 1
loads the whole unit on every rank and cuts it at load
(``tessera.serving.sharding``), so the same checkpoint serves any TP degree and
re-sharding is a serve flag rather than a re-export.  Encoding per rank would
make the bytes a function of the machine they were built for, and a unit cut
for 4 ranks could not be re-cut for 8.

ROUTED-MoE EXPERTS ARE EXPORTED FROM AN EXPLICIT SOURCE LAYOUT, and the
plannable unit is the STACK.  A ``--plan-json`` entry keyed ``<moe>.experts``
(not one of its 864 leaves) gives every expert of that stack one rung; the
exporter writes ONE ``tessera.fused`` container per expert per PROJECTION under
``<moe>.experts.{e}.{proj}.wire`` and declares one ``config_groups`` entry with
``structure: "routed_moe"``, the two groups ``w13`` (gate then up, the row order
``RoutedExperts._load_w13`` narrows to) and ``w2``, and a per-group
``wire_stride``.  The stride is the MAXIMUM over the group's blobs, derived here
rather than passed in, because the blob length follows the data (the manifest's
exact-ratio ``global_scale`` rides as a varint whose width follows its value)
and ``moe_layout.unpack_moe_wires`` refuses a stride that is not what the
loaded lengths imply.  The consumer is ``tessera.serving.moe_route``, which
decodes those containers into vLLM's stock per-channel FP8 expert parameters
and runs the runtime's own fused-MoE kernel.

Everything about a stack is refused BEFORE the first encode
(``plan_expert_stack``): a family with no expert route
(``scheme.MOE_BUILDERS``), an expert index set that is not ``0..E-1``, a
missing projection, geometry that differs across experts, and rows or columns
the route's mainloop cannot take.  A dense Linear that fails the last of those
is passed through -- safe, because it is one module; a routed stack cannot be
half passed through, because vLLM builds ONE method for the whole stack.  The
construction gate runs on the stack name too: the census receipt's
``offered_non_linear`` row is what says the runtime asks this plugin about
``...mlp.experts`` at all.

Not planning a stack leaves it where it was -- source precision, named in
``ignore`` at the ``FusedMoE`` prefix -- so every checkpoint written before
this is byte-identical under the same command line.

THE PACKED 3-D SOURCE LAYOUT (``mlp.experts.gate_up_proj``,
``...down_proj``) is accepted only when the stack's plan states one of two
closed conventions: ``out_first_chunked`` means ``[E, 2N, K]`` with gate then
up and ``[E, K, N]`` down; ``in_first_interleaved`` means ``[E, K, 2N]`` with
gate/up alternating and ``[E, N, K]`` down.  The exporter checks those exact
shapes against config.json, slices them into canonical per-expert matrices,
and stamps the source convention on the scheme and every manifest role.  It
never infers a convention from dimensions: ``hidden_size == 2 *
moe_intermediate_size`` makes a gate/up source square, while dimensions never
state chunked versus interleaved.  Missing or invented conventions therefore
refuse before encoding.  ``quantizable`` separates both layouts from the 2-D
body, and an unplanned stack of either kind passes through at source precision
named in ``ignore``.  The grouping rule that a fused module's roles must share
one family holds for experts exactly as it does for q/k/v: vLLM builds one
method per module.
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
    FP8_INPUTS, FP8_WEIGHTS, NVFP4_INPUTS, NVFP4_WEIGHTS, regex_target,
    stock_quantization_config)
from tessera.alphabet import (  # noqa: E402
    BF16_GRID, E2M1_GRID, E4M3_GRID, tuple_grid)
from tessera.bf16_route import BF16_FAMILY  # noqa: E402
from tessera.export import (  # noqa: E402
    DEFAULT_CODE, DEFAULT_LDLQ_BLOCK, DEFAULT_LDLQ_SIGMA,
    ActivationSource, encode_linear_planes, wire_recipe)
from tessera.fused import pack_fused, shared_lut_global  # noqa: E402
from tessera.serving.contract import (  # noqa: E402
    classify_construction, construction_entry, load_serving_contract)
from tessera.serving.scheme import (  # noqa: E402
    MOE_BUILDERS, MOE_GROUP_PROJECTIONS, MOE_GROUPS,
    MOE_SHARD_PROJECTIONS as SHARD_PROJECTION, STRUCTURE_DENSE,
    STRUCTURE_ROUTED_MOE, MOE_SOURCE_IN_FIRST_INTERLEAVED,
    MOE_SOURCE_OUT_FIRST_CHUNKED, MOE_SOURCE_UNPACKED,
    refuse_unreachable_lane, refuse_unserveable_wire, validate_tessera_moe_scheme)
from tessera.stock import (  # noqa: E402
    FLOAT_QUANTIZED, MIXED_PRECISION, NVFP4_PACK_QUANTIZED, materialize_stock,
    share_global, stock_bytes, vllm_fp4_predicate)
from tessera.unit_artifact import parse_unit_artifact  # noqa: E402
from tessera.serving_parts import (  # noqa: E402
    BODY_LAYER, SCHEMA as PART_SCHEMA, export_identity, parse_partition,
    partition_owner, sha256_file)

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
#: blocks.N.``, which this does not match, so it stays BF16 -- and is named in
#: ``ignore`` by ``ignored_modules``, which runs over the tensors WRITTEN rather
#: than over the ones this pattern matched.  Staying BF16 and being named are
#: different facts and the plugin reads the second one (#86).
# BODY_LAYER is shared with the whole-layer partition ownership rule.

#: GLM writes descriptive names; LFM writes shard ids. The shared scheme
#: table normalises both, while the emitted wire retains the source spelling
#: so the model's own FusedMoE mapping can hand it the shard id.
EXPERT_SOURCE_PROJECTIONS = tuple(
    dict.fromkeys((*SHARD_PROJECTION.values(), *SHARD_PROJECTION.keys())))

#: A ROUTED expert leaf in the unpacked (per-expert 2-D) source layout.  The
#: ``\d+`` segment is load-bearing: it is what distinguishes a routed expert
#: from ``mlp.shared_experts.gate_proj``, which is an ordinary dense Linear and
#: is quantized as one.  The owner is named ``mlp`` by GLM and ``feed_forward``
#: by LFM; its source projections are respectively gate/up/down and w1/w3/w2.
ROUTED_EXPERT_2D = re.compile(
    r"^(?P<moe>.*\.(?:mlp|feed_forward))\.experts\.(?P<expert>\d+)\."
    rf"(?P<proj>{'|'.join(map(re.escape, EXPERT_SOURCE_PROJECTIONS))})\.weight$")

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
#: principle 9 allows.  Attested against the GLM pin
#: ``prismaquant/glm53-mia-sm121:487ecf187`` and LFM's exact derived image
#: ``sha256:337dae6b...``: ``Lfm2MoeSparseMoeBlock`` constructs the same
#: ``ReplicatedLinear`` at ``<feed_forward>.gate`` and the expert factory at
#: its sibling ``.experts`` (PrismaBuild action ``e3f322afb4ab...``).
MOE_ROUTER = re.compile(
    r"^(?P<moe>.*\.(?:mlp|feed_forward))\.(?:gate|router)\.weight$")

#: A PACKED expert stack: one rank-3 tensor holding every expert of a layer.
#: Rank alone does NOT identify one -- GLM-5.3-Flash's attention carries
#: ``k_conv1d.weight [8192, 1, 4]``, and treating that as an expert stack put
#: ``...self_attn`` (the whole attention block, every Linear in it) into the
#: checkpoint's ``ignore`` list.  A tensor is a packed expert stack because of
#: where it sits, not because it has three axes.
#:
#: THE SUFFIX IS NOT WHAT MAKES A STACK, so callers match this pattern against
#: the name in its ``.weight`` spelling rather than against the name as it sits
#: on disk.  transformers-5 stores a packed stack as an ``nn.Parameter`` on the
#: experts module rather than as a child Linear's ``weight``, so the tensor on
#: disk is ``...mlp.experts.gate_up_proj`` with no suffix at all -- 98 of them
#: on ``/mnt/shared/models/Qwen3.8-Flash-Next``, the one packed-source
#: checkpoint on this box (96 decoder-body stacks plus two MTP-sidecar stacks
#: outside ``BODY_LAYER``).  ``quantizable`` used to require the suffix before
#: it looked at anything, so those tensors were classified as NOTHING and no
#: plan-time refusal could name them; ``ignored_modules`` reads them through
#: the same probe (#86).  ``packed_expert_orientation`` already read both
#: spellings, which is how far the inconsistency reached before it was found.
PACKED_EXPERT_ND = re.compile(
    r"^(?P<moe>.*\.mlp)\.experts\.(?P<proj>gate_up_proj|down_proj|gate_proj|up_proj)\.weight$")

#: A module vLLM builds under a SECOND name, beside the checkpoint's spelling.
#: Not a taste and not a guess: CONSTRUCTED on the pinned build
#: ``prismaquant/glm53-mia-sm121:487ecf187`` with a recording quant config, the
#: GLM vision tower offers ``visual.blocks.N.attn.qkv_proj`` -- because
#: ``Glm5NextVisionAttention`` builds its projection at
#: ``f"{prefix}.qkv_proj" if quant_config else f"{prefix}.qkv"``
#: (``models/glm5next/nvidia/multimodal.py:167``) -- while the tensor on disk is
#: ``...attn.qkv.weight`` and the module ATTRIBUTE path is ``...attn.qkv``.  The
#: run, its script and its verbatim output are
#: ``docs/measurements/glm53-vision-tower-prefixes-2026-09-03.md``.
#: Which name exists depends on whether that runtime
#: passes a quant config, which the producer cannot know; an ignore entry is
#: only ever LOOKED UP or prefix-mapped -- never silently dropped, since
#: ``TesseraConfig.apply_vllm_mapper`` refuses a name the mapper maps away --
#: so carrying both spellings costs nothing and carrying
#: one is a load-time refusal in whichever world the runtime turns out to be
#: in.  Body attention never reaches here: its q/k/v arrive unmerged and the
#: FUSED table already names ``qkv_proj``.
MERGED_ALIASES = ((re.compile(r"^(.*\.)qkv\.weight$"), "qkv_proj"),)

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


def module_scheme_key(grid, q256: int) -> tuple:
    """The facts every role of one vLLM-fused module must agree on.

    NOT ``(grid, q256)`` (#37).  vLLM builds one quant method per module, so
    what has to be shared is what that method is built from -- the family, and
    with it the grid, body and scale plane its route decodes to one tile.  The
    RATE is not one of them: every decoder in ``tessera.serving`` reads a role
    from that role's OWN manifest, and the runtime says so in a value
    (``runtime_contract.json``'s ``fused_module.fields``, checked against
    ``scheme.FUSED_MODULE_FIELDS``).  The old key compared the rate too, so a
    group whose members took different rungs -- which is exactly what
    PrismaQuant's group knapsack allocates, folded ON by default -- was passed
    through at source precision by this script, and the allocator's chosen point
    had no Tessera export at all.

    The body and the plane are DERIVED from the rung here rather than assumed
    constant, because ``wire_recipe`` picks them per rung: ``E2M1x2`` writes the
    window body below the coset cap and the TCQ body at it.  Two such members
    would decode on two different decoders, and this key separates them, so the
    relaxation cannot let a mixed-body group through the back door.  (Every
    sub-cap ``E2M1x2`` rung is refused by ``check_recipe`` before this anyway;
    the key does not rely on that.)
    """
    recipe = wire_recipe(grid, q256)
    return (family_for(grid), grid.name, recipe.body.name, recipe.scale_plane.name)


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
        # The family this checkpoint will declare for the wire, so the gate
        # reads THAT route's published range rather than resolving the route
        # from the grid alone -- ambiguous the moment two routes hold one
        # grid, with the winner decided by dict order (#51).
        refuse_unserveable_wire(grid.name, q256, recipe.body.name, recipe.scale_plane.name,
                                family=family_for(grid), span=recipe.span, target=target)
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


def unrouted_modules(src_config, modules):
    """Which of these vLLM modules the pinned runtime will NOT route to a plugin.

    THE PRODUCER MAY NOT KEEP THIS ROSTER.  ``LinearBase.__init__`` takes
    ``UnquantizedLinearMethod()`` in the ``quant_config is None`` branch
    *without calling* ``get_quant_method`` (vLLM 0.28,
    ``model_executor/layers/linear.py:258``), so a projection a model builds
    with ``quant_config=None`` is invisible to every quantization plugin --
    ours cannot refuse it, warn about it, or even see the prefix.  A wire
    written there deletes the ``<module>.weight`` the runtime wants and puts
    bytes in its place that no loader maps: not a slow route, no route, and no
    refusal either.

    Principle 14 says a claim about what a runtime DOES is derived from a
    machine-readable table that runtime publishes.  The table is
    ``runtime_contract.json``'s ``construction`` block (contract v11), whose
    rows are generated from the census receipts under
    ``docs/measurements/construction/`` -- and the census itself
    (``tools/tessera_construction_census.py``) OBSERVES the answer by building
    the model the way the loader does with a probe quant config that records
    every prefix it is offered.  Nothing here reads source, and nothing here
    keeps a list.

    Returns ``{checkpoint module: (verdict, vllm module pattern)}`` for every
    module that is not ``offered``:

    * ``never_offered`` -- the runtime builds this module with
      ``quant_config=None``.  On GLM-5.3-Flash that is every attention
      projection, the whole KDA layer, the indexer and the vision tower.
    * ``absent`` -- the runtime builds no module of this name at all.  It is
      what a fused role named at the wrong seam looks like: the checkpoint has
      ``self_attn.{q,k,v}_proj`` on a KDA layer and vLLM builds ONE
      ``self_attn.in_proj_qkvbfg_a``, so the ``qkv_proj`` this script would
      declare corresponds to nothing.
    * ``uncensused`` -- no census covers this architecture.  That is an honest
      gap, not a clearance: the answer is unknown, so the wire is unpriced.
    """
    architectures = list(src_config.get("architectures") or ())
    entry = construction_entry(architectures)
    if entry is None:
        return {module: ("uncensused", "-") for module in modules}, None
    verdicts = {}
    for module in modules:
        verdict, pattern = classify_construction(entry, module)
        if verdict != "offered":
            verdicts[module] = (verdict, pattern)
    return verdicts, entry


def _unrouted_refusal(verdicts, architectures) -> str:
    kinds = {}
    for module, (verdict, pattern) in sorted(verdicts.items()):
        kinds.setdefault(verdict, []).append((module, pattern))
    lines = [f"{len(verdicts)} planned module(s) are not routed through this plugin by the "
             f"pinned runtime, so a wire written there is dead weight and the "
             f"<module>.weight it replaced is gone:"]
    reasons = {
        "never_offered": "built with quant_config=None -- vLLM's own BF16 method, and "
                         "get_quant_method is never called, so this plugin cannot even refuse it",
        "absent": "the runtime builds no module of this name (a fused role named at the wrong "
                  "seam, or a name from another architecture)",
        "uncensused": f"no construction census covers {architectures}; run "
                      "tools/tessera_construction_census.py inside the serving image and add "
                      "the receipt under docs/measurements/construction/",
    }
    for verdict, rows in sorted(kinds.items()):
        lines.append(f"  {verdict} ({len(rows)}): {reasons[verdict]}")
        for module, pattern in rows[:12]:
            lines.append(f"    {module}   -> {pattern}")
        if len(rows) > 12:
            lines.append(f"    ... and {len(rows) - 12} more")
    return "\n".join(lines)

def check_lanes(lanes, grid, q256: int, where: "str | None" = None) -> None:
    """Refuse the plan unless every requested LANE can read this rung's wire.

    The second half of the export-time seam, and the half #104 says was
    missing.  ``check_recipe`` asks whether the ROUTE publishes a decode for
    these bytes; this asks whether a named lane INSIDE that route can read
    them -- a different question, and one whose answer was previously
    discovered per module at LOAD, where the route catches it and serves the
    same bytes through its other path.  So a checkpoint built to exercise the
    window GEMV exercised nothing and said so nowhere.

    Placed at the same call sites as ``check_recipe`` -- the default rung and
    every ``--plan-json`` override, at argument time -- because a refusal
    after the encode is not a refusal; it is a bill.  There is no override:
    ``--require-lane`` is itself the opt-in, and a run that wants the rung
    without the lane simply does not pass the flag.
    """
    if not lanes:
        return
    recipe = wire_recipe(grid, q256)
    target = where or f"--grid {grid.name} --q256 {q256}"
    for lane in lanes:
        try:
            rates = refuse_unreachable_lane(
                lane, grid=grid.name, q256=int(q256), rate_cap=grid.rate_cap,
                body=recipe.body.name, plane=recipe.scale_plane.name,
                window_bits=int(recipe.window_bits), target=target)
        except ValueError as exc:
            raise SystemExit(
                f"{exc}\n\n--require-lane {lane} was passed, so this plan is refused HERE -- "
                "before a single unit is encoded. Drop the flag to write the rung anyway (it "
                "serves, through the route's other path), or re-plan onto a rung the lane "
                "reads.") from exc
        print(f"  --require-lane {lane}: {target} -> column rates {list(rates)}, readable",
              flush=True)


def module_of(tensor_name: str) -> str:
    return tensor_name[: -len(".weight")]


def fused_module(tensor_name: str):
    """``(fused module name, ordered member tensor names)`` or ``None``."""
    for pattern, fused, members in FUSED:
        match = pattern.match(tensor_name)
        if match:
            return match.group(1) + fused, tuple(match.group(1) + m + ".weight" for m in members)
    return None


def ignored_modules(tensor_name: str, shape) -> tuple[str, ...]:
    """The vLLM module names ``ignore`` must carry for a tensor written at source precision.

    Empty when the tensor is not a Linear weight the plugin can be asked
    about.  This is a RULE over the tensors the export actually writes, not a
    roster beside them: a roster is a second place to remember, and it goes
    stale in silence -- which is how the vision tower came to be passed
    through and never named.  The plugin refuses a ``LinearBase`` that is
    neither declared nor ignored, so the completeness has to follow the bytes.

    Three cases, and the two expert ones are why this is not simply
    ``module_of``:

    * a FUSED role names its fused parent, because vLLM builds one method per
      fused module.  Ignoring ``q_proj``/``k_proj``/``v_proj`` leaves
      ``qkv_proj`` neither declared nor ignored, which is the refusal again.
      The rule reaches outside the body too: ``Glm5NextVisionMLP`` builds one
      ``MergedColumnParallelLinear`` at ``{prefix}.gate_up_proj``
      (pinned build ``prismaquant/glm53-mia-sm121:487ecf187``,
      ``models/glm5next/nvidia/multimodal.py:102-107``), exactly as the body's
      MLP does.
    * a ROUTED expert leaf names the FusedMoE's OWN prefix, not one of its
      2592 checkpoint leaves.  Attested against the same build, three hops:
      ``models/glm5next/nvidia/model.py:239`` ``FusedMoEFactory(prefix=
      f"{prefix}.experts")``; ``layers/fused_moe/layer.py:221`` ``layer_name =
      prefix``; ``layers/fused_moe/routed_experts.py:122,:201``
      ``quant_config.get_quant_method(self, self.layer_name)``.  So the string
      the plugin tests is ``<layer>.mlp.experts`` and no leaf name is ever
      offered to it.  Naming the parent cannot reach the shared experts beside
      it: ``shared_experts`` is a SIBLING of ``experts``, and both the plugin's
      test and compressed-tensors' are exact/fnmatch, not prefix subsumption.
    * a PACKED expert stack (rank 3 or more) names the same FusedMoE prefix.
      Rank alone does not identify one -- GLM-5.3-Flash's attention carries
      ``k_conv1d.weight [8192, 1, 4]`` and a conv is not a Linear the plugin
      is ever asked about -- so the test is where the tensor sits.

    Over-naming is cheap here and under-naming is a load-time refusal: an
    ignore entry for a module the runtime never builds is never looked up,
    which is why a 2-D weight that is not a Linear at all (an embedding table)
    is named rather than second-guessed.
    """
    # A packed expert stack may carry NO ``.weight`` suffix -- transformers-5
    # stores it as a parameter on the experts module -- so the patterns are
    # matched against the name in its ``.weight`` spelling and the suffix is
    # not what decides whether a rank-3 tensor is a stack.
    probe = tensor_name if tensor_name.endswith(".weight") else tensor_name + ".weight"
    if len(shape) >= 3:
        packed = PACKED_EXPERT_ND.match(probe)
        return (packed.group("moe") + ".experts",) if packed else ()
    if not tensor_name.endswith(".weight") or len(shape) != 2:
        return ()
    routed = ROUTED_EXPERT_2D.match(probe)
    if routed:
        return (routed.group("moe") + ".experts",)
    fused = fused_module(probe)
    if fused:
        return (fused[0],)
    names = [module_of(probe)]
    names.extend(match.group(1) + alias for pattern, alias in MERGED_ALIASES
                 if (match := pattern.match(probe)))
    return tuple(names)


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
                if not BODY_LAYER.match(name):
                    continue
                # ``.weight`` is not what makes a tensor a body weight, and
                # gating on it here is what dropped a whole layout: a
                # transformers-5 packed expert stack is an ``nn.Parameter`` on
                # the experts module, so the tensor on disk is
                # ``...mlp.experts.gate_up_proj`` with no suffix at all.  It
                # landed in NO bucket -- not dense, not packed, not routed -- so
                # the export succeeded, encoded the dense body for hours, and
                # produced a checkpoint the plugin refuses at load because the
                # FusedMoE module reached neither ``config_groups`` nor the
                # plan-time refusal.  ``probe`` is the same idiom
                # ``ignored_modules`` uses (#86): match the patterns against the
                # name in its ``.weight`` spelling, so one convention decides
                # what a name means and the suffix decides nothing.
                probe = name if name.endswith(".weight") else name + ".weight"
                bare_packed = not name.endswith(".weight") and PACKED_EXPERT_ND.match(probe)
                if not name.endswith(".weight") and not bare_packed:
                    continue
                shape = tuple(handle.get_slice(name).get_shape())
                if bare_packed:
                    # RANK IS STILL ASKED, and a bare name that fails it is
                    # REFUSED rather than filed.  "Nothing else in a decoder
                    # layer is called ``experts.<projection>``" would be an
                    # assertion about every checkpoint yet to exist, and that
                    # is the same shape as the ``len(shape) >= 3`` rule §9.2
                    # already retired once.  A packed stack stacks experts, so
                    # it has an expert axis; a two-axis tensor under this name
                    # is something this exporter has not been shown, and the
                    # honest answer is to say so before the encode rather than
                    # to file it in whichever bucket happens to be nearest.
                    if len(shape) < 3:
                        raise SystemExit(
                            f"{name} {list(shape)} is named like a packed expert stack "
                            f"(``<moe>.experts.<projection>``) but has rank {len(shape)}, and a "
                            "packed stack carries an expert axis. Refusing rather than guessing: "
                            "filing it as an expert stack would leave the module BF16 and named "
                            "in ignore, and filing it as a dense Linear would declare a module "
                            "vLLM never builds -- and neither is a fact about this tensor. Teach "
                            "the exporter this architecture's layout explicitly.")
                    expert_shapes[name] = shape
                elif len(shape) >= 3:
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


#: The scheme owns canonical roles and their runtime row order, for both
#: the sidecar reader and this writer.
EXPERT_PROJECTIONS = tuple(p for g in MOE_GROUPS for p in MOE_GROUP_PROJECTIONS[g])
#: Which group a projection rides, inverted from the table above.
PROJECTION_GROUP = {p: g for g, ps in MOE_GROUP_PROJECTIONS.items() for p in ps}


def expert_stacks(routed_shapes):
    """``{<moe>.experts: {expert index: {projection: (tensor name, shape)}}}``.

    THE STACK IS THE PLANNABLE UNIT, not the leaf.  vLLM builds one method per
    ``RoutedExperts`` module and the sidecar declares one scheme for it
    (``scheme.validate_tessera_moe_scheme``), so family, grid, body and plane
    are stack facts and the rung is a stack-and-projection fact -- which is the
    per-layer expert uniformity this house's allocators already enforce.  A
    plan naming one expert's ``gate_proj`` is refused for the same reason a
    plan naming ``q_proj`` alone is: it describes half a module the runtime
    builds whole.
    """
    stacks: dict[str, dict[int, dict[str, tuple]]] = {}
    for name, shape in routed_shapes.items():
        match = ROUTED_EXPERT_2D.match(name)
        stack = match.group("moe") + ".experts"
        expert = stacks.setdefault(stack, {}).setdefault(int(match.group("expert")), {})
        source_projection = match.group("proj")
        projection = SHARD_PROJECTION.get(source_projection, source_projection)
        if projection in expert:
            other = expert[projection][0]
            raise SystemExit(
                f"the expert stack {stack} supplies canonical projection {projection!r} twice: "
                f"{other} and {name}. They are two source spellings for one runtime shard; "
                "refusing rather than letting checkpoint order choose which bytes are served.")
        expert[projection] = (name, tuple(shape))
    return stacks


def packed_expert_stacks(expert_shapes):
    """Group physical rank-3 expert tensors by the runtime stack they feed.

    A physical projection may be spelled with or without ``.weight``.  The
    suffix cannot decide identity, and two spellings of the same projection
    are refused instead of allowing safetensors order to choose the bytes.
    """
    stacks: dict[str, dict[str, tuple]] = {}
    for name, shape in expert_shapes.items():
        probe = name if name.endswith(".weight") else name + ".weight"
        match = PACKED_EXPERT_ND.match(probe)
        if match is None:  # ``quantizable`` owns this classification.
            raise SystemExit(
                f"{name} {list(shape)} was classified as a packed expert tensor but does "
                "not match PACKED_EXPERT_ND; refusing a source the planner cannot name")
        stack = match.group("moe") + ".experts"
        projection = match.group("proj")
        found = stacks.setdefault(stack, {})
        if projection in found:
            other = found[projection][0]
            raise SystemExit(
                f"the packed expert stack {stack} supplies {projection!r} twice: {other} "
                f"and {name}. They are two physical spellings for one source tensor; "
                "refusing rather than letting checkpoint order choose which bytes are served.")
        found[projection] = (name, tuple(shape))
    return stacks


def plan_expert_stack(stack: str, experts: dict, grid, q256: int, *,
                      source_layout: str = MOE_SOURCE_UNPACKED):
    """Everything about a planned expert stack that must be refused BEFORE the encode.

    A routed stack is 864 units on GLM-5.3-Flash and ~75 minutes of GPU per
    layer, so every disagreement between what the checkpoint holds and what the
    plugin's expert route reads is asked here rather than discovered at load.
    Returns the stack's plan record: the family, the two groups' geometry, and
    the per-unit work list in the row order the groups stack.

    The refusals, and why each is a refusal rather than a repair:

    * a family with no expert route -- ``scheme.MOE_BUILDERS`` is the one home
      for that rule, and its absences are measured (the NVFP4 oracle's
      ``swiglu_limit`` clamp, a 16-bit stack being the passthrough already);
    * an expert index set that is not ``0..E-1`` -- the parameter is
      ``[E, ...]`` and the loader indexes it by ``expert_id``, so a gap is a
      row of zeros served as an expert;
    * a missing projection -- ``w13`` is the gate/up PAIR, and a stack missing
      one has no second half for the tile;
    * geometry that differs across experts -- one stack is one tile, so one
      shape;
    * rows or columns the route's mainloop cannot take.  A dense Linear that
      fails this is passed through, which is safe because it is one module; a
      routed stack cannot be half passed through, because vLLM builds ONE
      method for it.
    """
    family = family_for(grid)
    if family not in MOE_BUILDERS:
        raise SystemExit(
            f"the plan gives the expert stack {stack} grid {grid.name} ({family}), which has no "
            f"expert route in this plugin build (scheme.MOE_BUILDERS names {sorted(MOE_BUILDERS)}). "
            "The absences are measured, not preferred: the fused-MoE oracle resolves an NVFP4 "
            "expert arm only under a swiglu_limit clamp that changes the arithmetic the experts "
            "execute (docs/measurements/nvfp4-moe-oracle-2026-09-02.md), and a 16-bit stack is "
            "the passthrough quantization_config.ignore already gives. Plan this stack on a "
            "family with a route, or leave it out to pass it through as BF16.")
    indices = sorted(experts)
    if indices != list(range(len(indices))):
        missing = sorted(set(range(max(indices) + 1)) - set(indices))
        raise SystemExit(
            f"the expert stack {stack} is missing expert(s) {missing[:8]} of "
            f"{max(indices) + 1}. The wire parameter is [E, ...] and the loader writes row "
            "expert_id, so a gap is not a smaller stack -- it is a row of zeros decoded as an "
            "expert. Refusing rather than renumbering.")
    geometry: dict[str, tuple] = {}
    for index in indices:
        found = experts[index]
        absent = [p for p in EXPERT_PROJECTIONS if p not in found]
        if absent:
            raise SystemExit(
                f"the expert stack {stack} expert {index} carries {sorted(found)} and is missing "
                f"{absent}. A group is exactly the shards the runtime loads into it "
                f"({dict(MOE_GROUP_PROJECTIONS)}, scheme.MOE_GROUP_SHARDS): w13 is the gate/up "
                "PAIR, so a stack missing one half has no second half for the tile.")
        for projection, (_name, shape) in found.items():
            if geometry.setdefault(projection, shape) != shape:
                raise SystemExit(
                    f"the expert stack {stack} holds {projection} at {list(geometry[projection])} "
                    f"and at {list(shape)} on expert {index}. One stack is one tile, so one "
                    "shape; a stack whose experts disagree cannot be [E, rows, cols].")
    hidden = geometry["gate_proj"][1]
    inter = geometry["gate_proj"][0]
    for projection, shape in geometry.items():
        want = (hidden, inter) if projection == "down_proj" else (inter, hidden)
        if shape != want:
            raise SystemExit(
                f"the expert stack {stack} holds {projection} at {list(shape)}; with gate_proj "
                f"[{inter}, {hidden}] the runtime's tile wants {list(want)}. w13 is [2N, K] and "
                "w2 is [K, N] over the same expert, so the two groups' geometry is one pair of "
                "numbers and this checkpoint's is not.")
    groups = {}
    for group in MOE_GROUPS:
        projections = MOE_GROUP_PROJECTIONS[group]
        rows_each = geometry[projections[0]][0]
        columns = geometry[projections[0]][1]
        if rows_each % (grid.arity * 32) or columns % 16:
            raise SystemExit(
                f"the expert stack {stack} group {group} is {rows_each}x{columns} per "
                f"projection, which the {grid.name} encoder cannot cut: rows must be a whole "
                f"number of tuples (grid.arity * 32 = {grid.arity * 32}) and columns a multiple "
                "of 16. A dense Linear that fails this is passed through; a routed stack cannot "
                "be half passed through, because vLLM builds ONE method per stack.")
        groups[group] = {"rows": rows_each * len(projections), "columns": columns,
                         "roles": [[p, rows_each] for p in projections],
                         "rows_each": rows_each}
    units = []
    for index in indices:
        for group in MOE_GROUPS:
            for projection in MOE_GROUP_PROJECTIONS[group]:
                name, shape = experts[index][projection]
                units.append({"tensor": name, "wire": name[: -len(".weight")] + ".wire",
                              "source_tensor": name, "source_layout": source_layout,
                              "source_slice": {"expert": index, "selector": "whole",
                                               "transpose": False},
                              "expert": index, "projection": projection, "group": group,
                              "rows": shape[0], "cols": shape[1]})
    return {"stack": stack, "family": family, "grid": grid, "q256": int(q256),
            "experts": len(indices), "hidden_size": hidden, "intermediate_size": inter,
            "source_layout": source_layout, "groups": groups, "units": units}


def _packed_config_geometry(config: dict, stack: str) -> tuple[int, int, int]:
    """The three dimensions a packed source must prove against config.json."""
    text = config.get("text_config", config)
    hidden = text.get("hidden_size")
    inter = text.get("moe_intermediate_size", text.get("intermediate_size"))
    experts = next((text.get(key) for key in
                    ("n_routed_experts", "num_experts", "num_local_experts")
                    if text.get(key) is not None), None)
    if not all(isinstance(value, int) and value > 0
               for value in (experts, hidden, inter)):
        raise SystemExit(
            f"cannot validate packed expert stack {stack}: config.json must declare positive "
            "integer expert count (n_routed_experts/num_experts/num_local_experts), "
            f"hidden_size, and moe_intermediate_size/intermediate_size; got experts={experts!r} "
            f"hidden_size={hidden!r} intermediate_size={inter!r}")
    return experts, hidden, inter


def plan_packed_expert_stack(stack: str, sources: dict, grid, q256: int, *,
                             source_layout: str, config: dict):
    """Normalise one explicitly-described packed source to canonical units.

    The convention is deliberately not inferred from shape.  Orientation and
    gate/up split are independent facts, and the square ``hidden == 2 * inter``
    case demonstrates that shape cannot establish even the first one.
    """
    packed_layouts = (MOE_SOURCE_OUT_FIRST_CHUNKED,
                      MOE_SOURCE_IN_FIRST_INTERLEAVED)
    if source_layout not in packed_layouts:
        raise SystemExit(
            f"the packed expert stack {stack} requires source_layout to be one of "
            f"{packed_layouts}, got {source_layout!r}; the tensor does not state its "
            "orientation or whether gate/up is chunked or interleaved")
    if set(sources) != {"gate_up_proj", "down_proj"}:
        raise SystemExit(
            f"the packed expert stack {stack} carries physical projections "
            f"{sorted(sources)}, expected exactly ['down_proj', 'gate_up_proj']. The "
            "supported conventions describe one fused gate/up tensor and one down tensor; "
            "a different roster needs its own explicit source layout.")

    experts, hidden, inter = _packed_config_geometry(config, stack)
    expected = {
        MOE_SOURCE_OUT_FIRST_CHUNKED: {
            "gate_up_proj": (experts, 2 * inter, hidden),
            "down_proj": (experts, hidden, inter),
        },
        MOE_SOURCE_IN_FIRST_INTERLEAVED: {
            "gate_up_proj": (experts, hidden, 2 * inter),
            "down_proj": (experts, inter, hidden),
        },
    }[source_layout]
    for projection, want in expected.items():
        name, shape = sources[projection]
        if tuple(shape) != want:
            raise SystemExit(
                f"the packed expert stack {stack} declares source_layout={source_layout!r}, "
                f"under which {projection} must be {list(want)} from config.json; "
                f"{name} is {list(shape)}. Refusing rather than transposing or slicing a "
                "source whose convention disagrees with its plan.")

    synthetic: dict[int, dict[str, tuple]] = {}
    for expert in range(experts):
        synthetic[expert] = {
            "gate_proj": (f"{stack}.{expert}.gate_proj.weight", (inter, hidden)),
            "up_proj": (f"{stack}.{expert}.up_proj.weight", (inter, hidden)),
            "down_proj": (f"{stack}.{expert}.down_proj.weight", (hidden, inter)),
        }
    record = plan_expert_stack(
        stack, synthetic, grid, q256, source_layout=source_layout)
    for unit in record["units"]:
        projection = unit["projection"]
        physical_projection = ("gate_up_proj" if projection in
                               ("gate_proj", "up_proj") else "down_proj")
        unit["source_tensor"] = sources[physical_projection][0]
        if source_layout == MOE_SOURCE_OUT_FIRST_CHUNKED:
            selector = {"gate_proj": "first_half", "up_proj": "second_half",
                        "down_proj": "whole"}[projection]
            transpose = False
        else:
            selector = {"gate_proj": "even", "up_proj": "odd",
                        "down_proj": "whole"}[projection]
            transpose = True
        unit["source_slice"] = {
            "expert": unit["expert"], "selector": selector,
            "transpose": transpose,
        }
    record["source_tensors"] = sorted(name for name, _shape in sources.values())
    return record


def packed_expert_weight(source: torch.Tensor, unit: dict) -> torch.Tensor:
    """Slice one canonical ``[rows, columns]`` matrix from a source tensor."""
    if unit.get("source_layout") == MOE_SOURCE_UNPACKED:
        weight = source
    else:
        spec = unit["source_slice"]
        weight = source[int(spec["expert"])]
        selector = spec["selector"]
        if selector == "first_half":
            weight = weight[:unit["rows"]]
        elif selector == "second_half":
            weight = weight[unit["rows"]:]
        elif selector == "even":
            weight = weight[:, 0::2]
        elif selector == "odd":
            weight = weight[:, 1::2]
        elif selector != "whole":
            raise SystemExit(
                f"{unit['tensor']}: unknown packed source selector {selector!r}")
        if spec["transpose"]:
            weight = weight.transpose(0, 1)
    want = (unit["rows"], unit["cols"])
    if tuple(weight.shape) != want:
        raise SystemExit(
            f"{unit['tensor']}: source slice produced {list(weight.shape)}, expected "
            f"canonical expert matrix {list(want)}")
    return weight.contiguous()


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
                    help='{"tensor.weight": {"grid": "E4M3"|"BF16", "q256": 1024} | '
                         '"PASSTHROUGH", "<moe>.experts": {"grid": "E4M3", '
                         '"q256": 1024, "source_layout": "out_first_chunked"|'
                         '"in_first_interleaved"}, ...}')
    ap.add_argument("--input-scales", type=Path, default=None,
                    help="safetensors carrying <module>.input_global_scale per NVFP4 Linear (a stock NVFP4 "
                         "export); required when any module takes the NVFP4 route")
    ap.add_argument("--stock-twin", type=Path, default=None,
                    help="also write the compressed-tensors materialisation of the same wires here")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--layers", type=int, default=None, help="encode only the first N layers (smoke)")
    ap.add_argument("--partition", type=parse_partition, metavar="INDEX/COUNT",
                    help="write only whole layers owned by layer %% COUNT == INDEX; "
                         "non-body tensors belong to index 0. Merge every part before serving.")
    ap.add_argument("--partition-runtime-image",
                    help="exact repository@sha256 image pinned by the part's dispatch command")
    ap.add_argument("--hessian", type=Path, default=None,
                    help="capture_h_full.py payload: full input Hessians keyed by the tensor's module "
                         "name.  Enables the activation-aware encoder settings below; an encode that "
                         "uses them is not reproducible from the weights alone, so the file's own "
                         "provenance is copied into the manifest.")
    ap.add_argument("--ldlq-sigma", type=float, default=DEFAULT_LDLQ_SIGMA,
                    help="Hessian regulariser for LDLQ cross-column feedback; a negative value turns LDLQ off")
    ap.add_argument("--ldlq-block", type=int, default=DEFAULT_LDLQ_BLOCK, help="LDLQ input-feature block")
    ap.add_argument("--ldlq-block-budget", type=float, default=None,
                    help="derive each unit's LDLQ block from its own Hessian instead "
                         "of stating one: the largest block whose predicted penalty "
                         "against full feedback (compensate.block_penalty) is within "
                         "this ratio, floored at 1 column. Mutually exclusive with "
                         "--ldlq-block. Prices the same axis the two measured "
                         "populations disagree about by a factor of 70 at b=32.")
    ap.add_argument("--refit-metric", default=None,
                    help="error the scale refit minimises: plain | hessian | h^ALPHA. "
                         "Default: the measured objective for each unit's own scale "
                         "plane (export.DEFAULT_REFIT_OBJECTIVE), which is not one "
                         "value -- the exact quadratic on the CHANNEL plane, the "
                         "diagonal h^1.0 on the LUT plane")
    ap.add_argument("--refit-metric-trailing", default=None,
                    help="error the LAST refit minimises, when it should differ from "
                         "--refit-metric: plain | hessian | h^ALPHA, at the SAME pass "
                         "count (tessera#75's fair pair). Unset is the uniform "
                         "schedule -- the encode that was already there, byte for "
                         "byte. Recorded in the checkpoint's activation_aware block "
                         "and compared by the merge guard (tessera#103), so a part "
                         "encoded under one trailing objective cannot merge with a "
                         "part encoded under another.")
    ap.add_argument("--refit-reach-floor", action="store_true",
                    help="hold every refit row scale high enough that the pass's target stays inside the body's reach")
    ap.add_argument("--passthrough-unrouted", action="store_true",
                    help="pass a Linear the pinned runtime will not route through this plugin "
                         "through at source precision instead of refusing the export. The SAFE "
                         "direction: it never writes a wire nothing executes. See "
                         "unrouted_modules.")
    ap.add_argument("--allow-unrouted", action="store_true",
                    help="write the wire anyway for a Linear the pinned runtime will not route "
                         "(or that no census covers). A RESEARCH escape: the module keeps "
                         "vLLM's BF16 method, the weight it wants is not in the checkpoint, and "
                         "the refusal is stamped verbatim into the manifest's serving_gate "
                         "block.")

    ap.add_argument("--require-lane", action="append", default=None, metavar="LANE",
                    help="refuse the PLAN unless every wire it writes can be read by LANE -- a "
                         "lane named by the extension module_name_prefix runtime_contract.json "
                         "publishes it under (tessera_window_gemv). A lane is one launch inside "
                         "a route, and its eligibility is a function of the RUNG: a rung is a "
                         "root rate and bresenham mixes the two rates bracketing it, so q256 "
                         "1006 (root 3.93) is columns at rate 3 and 4 and every unit of that "
                         "checkpoint refuses the window GEMV at load -- silently, module by "
                         "module, while the census meant to measure the lane records the "
                         "fallback (issue #104). Checked at ARGUMENT time, before one unit is "
                         "encoded, and stamped into the manifest as requires_lanes so the "
                         "census requires it off the bytes. Repeatable.")
    ap.add_argument("--allow-unserveable", action="store_true",
                    help="write wires this plugin build publishes no decode for (see check_recipe). "
                         "The checkpoint is then a RESEARCH artifact that will not load under this "
                         "plugin; every refusal is stamped verbatim into the manifest's "
                         "serving_gate block. Needed today by --grid BF16, whose wire has no "
                         "plugin route and whose --stock-twin is what gets served.")
    args = ap.parse_args()
    if args.partition:
        if args.stock_twin is not None:
            ap.error("--partition does not support --stock-twin; assemble the wire checkpoint first")
        if args.out.exists():
            ap.error(f"partition output already exists: {args.out}; use a fresh directory")
        if not args.partition_runtime_image:
            ap.error("--partition requires --partition-runtime-image")
    elif args.partition_runtime_image:
        ap.error("--partition-runtime-image requires --partition")

    # The activation-aware settings fire when, and only when, a Hessian is
    # here: the encoder cannot invent one, and a weights-only export must stay
    # the byte-for-byte artifact it was.  Given one, the defaults are the
    # measured recipe (export.DEFAULT_LDLQ_*), overridable per run.  The recipe
    # itself lives in ``ActivationSource``, not here: this script and the
    # library exporters must not carry two copies of it, which is the drift
    # that let the library path encode weights-only while the script did not.
    activation = None
    if args.hessian:
        block: "int | dict" = args.ldlq_block
        if args.ldlq_block_budget is not None:
            if "--ldlq-block" in sys.argv:
                ap.error("--ldlq-block and --ldlq-block-budget both state the "
                         "block: one names it, the other derives it. Pass one.")
            block = {"max_penalty": args.ldlq_block_budget}
        settings = {"ldlq_sigma": args.ldlq_sigma, "ldlq_block": block,
                    "refit_reach_floor": args.refit_reach_floor}
        if args.refit_metric is not None:      # else: the measured per-plane map
            settings["refit_objective"] = args.refit_metric
        if args.refit_metric_trailing is not None:   # else: the uniform schedule
            settings["refit_objective_trailing"] = args.refit_metric_trailing
        activation = ActivationSource.from_capture(args.hessian, **settings)

    default_grid = grid_for(args.grid)
    # Every path into the encode loop passes through here: a tensor takes the
    # default (grid, q256) or a --plan-json override, and both are gated before
    # a single unit is encoded.
    gate_overrides: list = []
    required_lanes = list(dict.fromkeys(args.require_lane or ()))
    check_recipe(default_grid, args.q256,
                 allow_unserveable=args.allow_unserveable, overrides=gate_overrides)
    check_lanes(required_lanes, default_grid, args.q256)
    overrides = {}
    source_layout_overrides: dict[str, str] = {}
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
                check_lanes(required_lanes, g, int(spec["q256"]), where=name)
                overrides[name] = (g, int(spec["q256"]))
                if "source_layout" in spec:
                    source_layout_overrides[name] = spec["source_layout"]
    src_config = json.loads((args.src / "config.json").read_text())
    shards, shapes, expert_shapes, routed_shapes = quantizable(args.src)
    if not shapes and not expert_shapes and not routed_shapes:
        raise SystemExit(
            f"no body weight tensors found under {args.src}. BODY_LAYER matches "
            f"``model.<...>.layers.<N>.``; this checkpoint's names do not, so there is nothing "
            "to export rather than nothing to do.")

    # THE PLANNABLE ROUTED-MoE UNIT IS THE STACK.  A plan entry naming
    # ``<moe>.experts`` is an expert-route plan for the whole stack; a plan
    # entry naming one of its 864 leaves is the mis-plan below.  The split
    # happens before the ``unknown`` check, which knows only about 2-D dense
    # body weights.
    stacks = expert_stacks(routed_shapes)
    packed_stacks = packed_expert_stacks(expert_shapes)
    overlap = sorted(set(stacks) & set(packed_stacks))
    if overlap:
        raise SystemExit(
            f"expert stack(s) {overlap} exist in both unpacked per-expert and packed 3-D "
            "source layouts. They are two sources for the same runtime tile; refusing rather "
            "than letting checkpoint or shard order choose which bytes are served.")
    all_stacks = {**stacks, **packed_stacks}
    stack_overrides = {name: overrides.pop(name)
                       for name in sorted(set(overrides) & set(all_stacks))}

    # A routed expert LEAF must be refused HERE -- before a single unit is
    # encoded.  These leaves are 2-D, so the only thing standing between them
    # and being planned as dense Linears is this check.
    planned_routed = sorted(set(overrides) & set(routed_shapes))
    if planned_routed:
        first = planned_routed[0]
        raise SystemExit(
            f"the plan names {len(planned_routed)} ROUTED expert tensor(s), e.g. {first} "
            f"{list(routed_shapes[first])}. A routed expert is not a dense Linear: vLLM builds "
            "one FusedMoE module per layer, not one Linear per expert, so a checkpoint declaring "
            f"{module_of(first)} in config_groups names a module vLLM never creates and the "
            "plugin refuses it at load. The expert route is not what is missing -- name the "
            f"STACK instead: a plan entry keyed {module_of(first).rsplit('.experts.', 1)[0]}"
            ".experts gives every expert of that stack one rung, which is what the runtime "
            "builds one method for and what the sidecar declares one scheme for. Or remove "
            "these entries to pass the stack through as BF16.")
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
            f"{list(expert_shapes[first])}. A packed expert tensor is not a plannable module: "
            "vLLM builds one method for the whole stack. Name <moe>.experts instead and state "
            f"source_layout as one of {(MOE_SOURCE_OUT_FIRST_CHUNKED, MOE_SOURCE_IN_FIRST_INTERLEAVED)}, "
            "or remove the physical entry to pass the stack through at source precision.")
    misplaced_layouts = sorted(set(source_layout_overrides) - set(stack_overrides))
    if misplaced_layouts:
        raise SystemExit(
            f"source_layout is a routed-MoE stack property, but the plan attaches it to "
            f"{misplaced_layouts[:5]}. Put it on <moe>.experts; a dense tensor or physical "
            "expert leaf has no packed-source convention to apply.")
    unknown = sorted(set(overrides) - set(shapes))
    if unknown:
        raise SystemExit(f"plan names tensors that are not 2-D body weights here: {unknown[:5]}; "
                         f"a routed-MoE stack is planned under its own name "
                         f"({sorted(stacks)[0] if stacks else '<moe>.experts'}), not per leaf")

    # THE EXPERT PLAN, gated before the first encode.  Every check the plugin's
    # expert route makes at load has a producer-side twin here, because a
    # refusal that arrives after 864 units is not a refusal; it is a bill.
    stack_plan: dict[str, dict] = {}
    for stack in sorted(stack_overrides):
        spec = stack_overrides[stack]
        if spec is None:                     # PASSTHROUGH, spelled deliberately
            continue
        grid, q256 = spec
        source_layout = source_layout_overrides.get(stack)
        if stack in packed_stacks:
            if source_layout not in (MOE_SOURCE_OUT_FIRST_CHUNKED,
                                     MOE_SOURCE_IN_FIRST_INTERLEAVED):
                raise SystemExit(
                    f"the packed expert stack {stack} requires source_layout to be one of "
                    f"{(MOE_SOURCE_OUT_FIRST_CHUNKED, MOE_SOURCE_IN_FIRST_INTERLEAVED)}, "
                    f"got {source_layout!r}; dimensions do not state orientation or the "
                    "gate/up split")
            source_name = next(iter(packed_stacks[stack].values()))[0]
        else:
            if source_layout not in (None, MOE_SOURCE_UNPACKED):
                raise SystemExit(
                    f"the unpacked expert stack {stack} has one 2-D tensor per expert and "
                    f"requires source_layout={MOE_SOURCE_UNPACKED!r} when the field is "
                    f"present, got {source_layout!r}")
            source_layout = MOE_SOURCE_UNPACKED
            source_name = next(iter(stacks[stack].values()))["gate_proj"][0]
        layer = body_layer(source_name)
        if args.layers is not None and layer >= args.layers:
            raise SystemExit(
                f"the plan gives the expert stack {stack} a rung, but --layers {args.layers} "
                f"stops before its layer {layer}. One of the two is wrong and guessing which "
                "would either skip a stack the plan asked for or encode past the smoke bound.")
        if stack in packed_stacks:
            record = plan_packed_expert_stack(
                stack, packed_stacks[stack], grid, q256,
                source_layout=source_layout, config=src_config)
        else:
            record = plan_expert_stack(
                stack, stacks[stack], grid, q256, source_layout=source_layout)
        stack_plan[stack] = record
        print(f"  routed_moe {stack}: {record['experts']} experts x "
              f"{len(EXPERT_PROJECTIONS)} projections at {grid.name} q256={q256} "
              f"({len(record['units'])} units, source_layout={source_layout})", flush=True)
    packed_plans = sorted(stack for stack in stack_plan
                          if stack_plan[stack]["source_layout"] != MOE_SOURCE_UNPACKED)
    if activation is not None and packed_plans:
        raise SystemExit(
            f"--hessian was given with packed expert stack(s) {packed_plans}. Activation "
            "captures are keyed by logical per-expert modules, while this checkpoint carries "
            "two physical stack tensors; their Hessian ownership and slicing have not been "
            "attested. Refusing rather than silently encoding those experts weights-only.")
    if stack_plan and args.stock_twin is not None:
        raise SystemExit(
            f"--stock-twin was given and {len(stack_plan)} routed-MoE stack(s) are planned. The "
            "twin is the materialised compressed-tensors comparator, and this exporter writes no "
            "per-channel FP8 expert twin: the stacks' source tensors are consumed by the encode, "
            "so the twin would silently be a checkpoint with no experts in it. Export the stacks "
            "without a twin, or leave them out of the plan.")

    packed_passthrough = {name: shape for name, shape in expert_shapes.items()
                          if next(m for m in ignored_modules(name, shape)) not in stack_plan}
    if packed_passthrough:
        print(f"  {len(packed_passthrough)} unplanned packed expert tensors stay at source "
              f"precision and are named in ignore; e.g. {sorted(packed_passthrough)[0]}",
              flush=True)
    routed_passthrough = {n: shape for n, shape in routed_shapes.items()
                          if ROUTED_EXPERT_2D.match(n).group("moe") + ".experts" not in stack_plan}
    if routed_passthrough:
        layers = sorted({body_layer(n) for n in routed_passthrough})
        print(f"  {len(routed_passthrough)} routed expert tensors across layers {layers} stay "
              f"BF16 and are named in ignore (no plan entry names their stack); "
              f"e.g. {sorted(routed_passthrough)[0]}", flush=True)
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
    # Group by vLLM module: every role quantizable, and every role on the same
    # ``module_scheme_key`` -- one family/grid/body/plane.  The RATE may differ
    # per role (#37); see ``module_scheme_key``.
    modules: dict[str, list[str]] = {}
    for name in list(plan):
        fused = fused_module(name)
        if fused is None:
            modules[module_of(name)] = [name]
            continue
        key, members = fused
        if key in modules:
            continue
        recipes = {module_scheme_key(plan[m][0], plan[m][1]) for m in members if m in plan}
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
    if not plan and not stack_plan and args.layers is None:
        # An export that quantized NOTHING used to report success and write a
        # checkpoint with an empty ``config_groups``, which the plugin refuses
        # at load ("a Tessera checkpoint declares its wires in
        # config_groups").  Same shape of fault as #86: silent here, expensive
        # there.  ``--layers 0`` is how a passthrough copy is asked for on
        # purpose, so it stays legal.
        raise SystemExit(
            f"nothing was planned: all {len(shapes)} dense body weights were passed through. A "
            "Linear is planned only when its rows are a whole number of tuples (grid.arity * 32) "
            "and its columns a multiple of 16; a plan naming PASSTHROUGH, or a menu whose grid "
            "no tensor here fits, leaves nothing to encode. The checkpoint would carry an empty "
            "config_groups and the plugin refuses that at load. Pass --layers 0 to write a "
            "passthrough copy deliberately.")

    # THE CONSTRUCTION GATE, and it is placed here for the same reason
    # ``check_recipe`` is placed at argument time: before the first encode.  A
    # refusal that arrives after the encode is not a refusal; it is a bill.
    # It runs on the MODULE names, not the tensor names, because the module is
    # what the runtime builds and what ``config_groups`` will declare.
    unrouted, census = unrouted_modules(src_config, list(modules) + list(stack_plan))
    unrouted_records = [
        {"module": module, "vllm_module_pattern": pattern, "verdict": verdict}
        for module, (verdict, pattern) in sorted(unrouted.items())]
    if unrouted:
        message = _unrouted_refusal(unrouted, list(src_config.get("architectures") or ()))
        if args.passthrough_unrouted:
            print(f"  --passthrough-unrouted: {message}", flush=True)
            for module in unrouted:
                if module in stack_plan:
                    # A stack is one module with no members in ``plan``: drop
                    # the whole stack and its experts pass through as BF16,
                    # named in ignore by the same rule as any other.
                    del stack_plan[module]
                    continue
                for member in modules.pop(module):
                    del plan[member]
                    passthrough.append(member)
            print(f"  {len(unrouted)} module(s) pass through at source precision; "
                  f"{len(modules)} module(s) remain in the plan", flush=True)
            if not plan and not stack_plan and args.layers is None:
                # The same refusal as above, re-asked, because the check above
                # ran BEFORE this loop and this loop is the other way ``plan``
                # empties.  Without it, --passthrough-unrouted on a model whose
                # every module is unrouted writes the empty ``config_groups``
                # the earlier guard exists to prevent -- and it is the likelier
                # path of the two, since a wholly unrouted architecture is
                # exactly when someone reaches for the flag.
                raise SystemExit(
                    f"nothing is left to encode: --passthrough-unrouted passed "
                    f"through all {len(unrouted)} unrouted module(s) and no "
                    "module remains in the plan. The checkpoint would carry an "
                    "empty config_groups and the plugin refuses that at load. "
                    "This architecture has no module the pinned runtime routes "
                    "to a Tessera method: census it first, or pass --layers 0 "
                    "to write a passthrough copy deliberately.")
        elif args.allow_unrouted:
            print(f"  --allow-unrouted: {message}", flush=True)
        else:
            raise SystemExit(
                f"{message}\n\n"
                "This is a fact about the pinned runtime, read from "
                "runtime_contract.json's construction block (contract v"
                f"{load_serving_contract()['contract_version']}), which is generated from the "
                "census receipts under docs/measurements/construction/. Pass "
                "--passthrough-unrouted to export these at source precision (the safe fix), or "
                "--allow-unrouted to write the dead wire anyway as a research artifact -- the "
                "refusal is then stamped into the manifest's serving_gate block.")
    owned = {m for members in modules.values() for m in members}
    assert owned == set(plan)

    partition_record = None
    if args.partition:
        index, count = args.partition
        options = {key: value for key, value in vars(args).items()
                   if key not in {"src", "out", "partition", "partition_runtime_image",
                                  "device", "stock_twin", "plan_json", "hessian", "input_scales"}}
        options["plan"] = json.loads(args.plan_json.read_text()) if args.plan_json else None
        for key in ("hessian", "input_scales"):
            path = getattr(args, key)
            options[key + "_sha256"] = sha256_file(path) if path else None
        identity = export_identity(args.src, options, args.partition_runtime_image,
                                   Path(__file__).resolve().parents[1])
        from tessera.encoder_identity import encoder_fixture_id
        identity["encoder_fixture_id"] = encoder_fixture_id().hex()
        owns = lambda name: partition_owner(name, count) == index
        selected = sorted(name for names in shards.values() for name in names if owns(name))
        if not selected:
            raise SystemExit(f"partition {index}/{count} owns no source tensors")
        partition_record = {"schema": PART_SCHEMA, "index": index, "count": count,
                            "identity": identity, "source_tensors": selected}
        shards = {shard: [name for name in names if owns(name)] for shard, names in shards.items()}
        shards = {shard: names for shard, names in shards.items() if names}
        # Planning and the construction gate above see the same complete plan
        # on every worker. Only execution is partitioned, at whole-layer ownership.
        plan = {name: value for name, value in plan.items() if owns(name)}
        modules = {name: members for name, members in modules.items() if owns(name)}
        stack_plan = {name: value for name, value in stack_plan.items() if owns(name)}
        shapes = {name: shape for name, shape in shapes.items() if owns(name)}
        expert_shapes = {name: shape for name, shape in expert_shapes.items() if owns(name)}
        routed_shapes = {name: shape for name, shape in routed_shapes.items() if owns(name)}
        passthrough = [name for name in passthrough if owns(name)]
        print(f"partition {index}/{count}: {len(selected)} source tensors, "
              f"{len(stack_plan)} routed stacks, {len(modules)} dense modules", flush=True)

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
    ignore = (["lm_head", "model.embed_tokens"]
              if args.partition is None or args.partition[0] == 0 else [])
    passthrough_bytes = 0
    weights_cache: dict[str, torch.Tensor] = {}
    done = 0
    total = len(plan)

    # THE EXPERT WORK LIST, keyed by the SOURCE tensor so the encode happens
    # where the tensor is read: one layer's experts span four shards on
    # GLM-5.3-Flash, and holding a stack's 864 source tensors until the last
    # one arrives is 13.8 GB of BF16 for no reason.  The wire tensor is written
    # into the same shard its source came from; the two GROUP strides are the
    # maxima over every blob and are therefore known only at the end, which is
    # exactly when ``config.json`` is written.
    expert_units: dict[str, list[dict]] = {}
    for stack, record in stack_plan.items():
        for unit in record["units"]:
            expert_units.setdefault(unit["source_tensor"], []).append(
                dict(unit, stack=stack))
    moe_records = {stack: {"family": record["family"], "grid": record["grid"].name,
                           "q256": record["q256"], "experts": record["experts"],
                           "source_layout": record["source_layout"],
                           "hidden_size": record["hidden_size"],
                           "intermediate_size": record["intermediate_size"],
                           "roles": [], "container_bytes": 0, "wire_bytes": 0,
                           "resident_bytes_resident_mode": 0,
                           "group_blob_bytes": {g: [] for g in MOE_GROUPS},
                           "rows": sum(g["rows"] for g in record["groups"].values()),
                           "cols": record["hidden_size"]}
                   for stack, record in stack_plan.items()}
    moe_total = sum(len(units) for units in expert_units.values())
    moe_done = 0

    pending_modules = dict(modules)
    for shard, names in sorted(shards.items()):
        shard_payload: dict[str, torch.Tensor] = {}
        twin_payload: dict[str, torch.Tensor] = {}
        with safe_open(str(args.src / shard), framework="pt") as handle:
            for name in names:
                tensor = handle.get_tensor(name)
                if name in plan:
                    weights_cache[name] = tensor
                elif name in expert_units:
                    for unit in expert_units[name]:
                        stack_spec = stack_plan[unit["stack"]]
                        unit_grid, unit_q256 = stack_spec["grid"], stack_spec["q256"]
                        unit_recipe = wire_recipe(unit_grid, unit_q256)
                        weight = packed_expert_weight(tensor, unit).to(
                            args.device, torch.float32).contiguous()
                        # Same call as the dense path: a missing Hessian key renders
                        # RTN and raises nothing, ``for_unit`` refuses instead.
                        extra = ({} if activation is None else
                                 activation.for_unit(
                                     unit["tensor"], weight.shape[1], args.device,
                                     scale_plane=unit_recipe.scale_plane))
                        exported, unit_artifact_, _forests = encode_linear_planes(
                            weight, grid=unit_grid, q256=unit_q256,
                            name=unit["tensor"], verify=not args.no_verify, **extra)
                        extra.clear()
                        parse_unit_artifact(exported.blob, device=args.device)
                        # ONE container per expert PROJECTION -- the granularity of
                        # the runtime's shard ids and of
                        # ``scheme.expert_role_declarations``.  A packed physical
                        # tensor may supply several such logical projections.
                        blob = pack_fused([
                            (unit["projection"], exported.rows, exported.blob)])
                        shard_payload[unit["wire"]] = torch.frombuffer(
                            bytearray(blob), dtype=torch.uint8).clone()
                        stack_record = moe_records[unit["stack"]]
                        stack_record["group_blob_bytes"][unit["group"]].append(len(blob))
                        stack_record["container_bytes"] += len(blob)
                        stack_record["wire_bytes"] += exported.exact_bytes
                        stack_record["resident_bytes_resident_mode"] += (
                            exported.rows * exported.columns + exported.rows * 4)
                        stack_record["roles"].append({
                            "tensor": unit["tensor"],
                            "source_tensor": unit["source_tensor"],
                            "source_layout": unit["source_layout"],
                            "source_slice": unit["source_slice"],
                            "role": unit["projection"], "expert": unit["expert"],
                            "group": unit["group"], "rows": exported.rows,
                            "cols": exported.columns, "grid": unit_grid.name,
                            "q256": unit_q256, "family": stack_spec["family"],
                            "wire_bytes": exported.exact_bytes,
                            "blob_bytes": len(blob), "wire_bpp": float(exported.bpp),
                            "own_global": float(unit_artifact_.scale_global)})
                        del weight
                        moe_done += 1
                        if moe_done % 50 == 0 or moe_done == moe_total:
                            print(f"  [moe {moe_done}/{moe_total}] {unit['tensor']}  "
                                  f"{time.time() - started:.0f}s", flush=True)
                else:
                    shard_payload[name] = tensor
                    twin_payload[name] = tensor
                    passthrough_bytes += tensor.numel() * tensor.element_size()
                    # EVERY tensor written at source precision is named here,
                    # body or not.  ``ignore`` used to be assembled from three
                    # BODY_LAYER-gated sources, so a Linear outside the decoder
                    # body -- a vision tower, an MTP sidecar -- was passed
                    # through and never named, and the plugin refuses exactly
                    # that (#86).  Deriving the name from the tensor just
                    # written is what keeps the two facts one fact.
                    ignore.extend(ignored_modules(name, tensor.shape))
        for module, members in list(pending_modules.items()):
            if not all(m in weights_cache for m in members):
                continue
            grid, _q0, _r, _c = plan[members[0]]
            family = family_for(grid)
            # The module-level facts, which the grouping key already proved every
            # member shares.  The RATE is read per member below.
            recipe = wire_recipe(grid, plan[members[0]][1])
            rungs = [int(plan[m][1]) for m in members]
            roles = []
            role_records = []
            stock_tensors: dict[str, dict] = {}
            for member in members:
                member_grid, q256, _mr, _mc = plan[member]
                member_recipe = wire_recipe(member_grid, q256)
                weight = weights_cache.pop(member).to(args.device, torch.float32).contiguous()
                # A missing key renders RTN and raises nothing; ``for_unit``
                # refuses instead, and is the same call the library exporters make.
                extra = ({} if activation is None else
                         activation.for_unit(member, weight.shape[1], args.device,
                                             scale_plane=member_recipe.scale_plane))
                exported, unit, forests = encode_linear_planes(
                    weight, grid=member_grid, q256=q256, name=member,
                    verify=not args.no_verify, **extra)
                extra.clear()
                parse_unit_artifact(exported.blob, device=args.device)      # the reader accepts what we wrote
                role = module_of(member).rsplit(".", 1)[-1]
                roles.append((role, exported.rows, exported.blob, unit, forests))
                stock_tensors[member] = materialize_stock(unit, forests, DEFAULT_CODE)
                role_records.append({
                    "tensor": member, "role": role, "rows": exported.rows, "cols": exported.columns,
                    "grid": member_grid.name, "q256": q256, "family": family,
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
            # The module's rung is a scalar when its roles agree and the list of
            # theirs when they do not, the same spelling the scheme uses (#37).
            module_q256 = rungs[0] if len(set(rungs)) == 1 else list(rungs)
            record = {
                "family": family, "grid": grid.name, "q256": module_q256, "roles": role_records,
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
                # ``structure`` names what kind of vLLM layer this is.  This
                # loop writes ``dense`` -- one blob per LinearBase.  The other
                # value the plugin serves, ``routed_moe``, is declared after
                # the shard loop, where the expert stacks' strides are known.
                "family": family, "structure": STRUCTURE_DENSE,
                "grid": grid.name, "body": recipe.body.name, "plane": recipe.scale_plane.name,
                # An int when every role took the same rung -- what every
                # checkpoint before #37 carries -- and one rung per role, in
                # ``roles`` order, when they did not.  ``validate_tessera_scheme``
                # reads both; a plugin older than contract v6 refuses the list.
                "q256": module_q256, "rows": rows_total, "columns": cols, "wire_bytes": len(blob),
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

    # THE EXPERT STACKS' SCHEMES, written once every blob's length is known.
    # ``wire_stride`` is the MAXIMUM over the group's blobs and is derived here
    # rather than declared: the blob length follows the data (the manifest's
    # exact-ratio ``global_scale`` rides as a varint whose width follows its
    # value), so there is no number a caller could pass that the bytes would
    # not contradict -- and ``moe_layout.unpack_moe_wires`` refuses a stride
    # that is not what the loaded lengths imply, which is the same check from
    # the other side.
    for stack, spec in stack_plan.items():
        stack_record = moe_records[stack]
        recipe = wire_recipe(spec["grid"], spec["q256"])
        groups = {}
        for group in MOE_GROUPS:
            lengths = stack_record["group_blob_bytes"][group]
            want = spec["experts"] * len(MOE_GROUP_PROJECTIONS[group])
            if len(lengths) != want:
                raise SystemExit(
                    f"{stack} group {group}: {len(lengths)} container(s) written for {want} "
                    "(experts x projections). A group whose rows are not all there would be "
                    "served with zero rows for the rest.")
            groups[group] = {
                "q256": spec["q256"], "rows": spec["groups"][group]["rows"],
                "columns": spec["groups"][group]["columns"],
                "roles": spec["groups"][group]["roles"], "wire_stride": max(lengths)}
        scheme = {
            "family": spec["family"], "structure": STRUCTURE_ROUTED_MOE,
            "source_layout": spec["source_layout"],
            "grid": spec["grid"].name, "body": recipe.body.name,
            "plane": recipe.scale_plane.name, "experts": spec["experts"], "groups": groups,
        }
        # The reader is the gate, so the writer is held to it here rather than
        # at load: this is the exact function ``TesseraConfig`` calls, on the
        # exact dict about to be written.
        validate_tessera_moe_scheme(scheme, stack)
        config_groups[f"tessera_{stack.replace('.', '_')}"] = {
            "format": "TESSERA", "targets": [stack], "scheme": scheme}
        stack_record["structure"] = STRUCTURE_ROUTED_MOE
        stack_record["wire_stride"] = {g: groups[g]["wire_stride"] for g in MOE_GROUPS}
        stack_record.pop("group_blob_bytes")
        stack_record["roles"].sort(key=lambda r: (r["expert"], r["group"], r["role"]))
        module_records[stack] = stack_record
        for role in stack_record["roles"]:
            units[role["tensor"]] = role

    # The expert stacks and the routed leaves were named by the same rule as
    # they were written (``ignored_modules``), which is the point: one mechanism
    # decides what is passed through and what is declared BF16.  What is worth
    # asserting is that the two agree -- a plan-time passthrough the write loop
    # somehow did not name would be a load-time refusal, so it is a refusal
    # here instead.
    unnamed = sorted(n for n in passthrough
                     if not set(ignored_modules(n, shapes[n])) <= set(ignore))
    if unnamed:
        raise SystemExit(
            f"{len(unnamed)} tensor(s) were planned as passthrough but never named in ignore, "
            f"e.g. {unnamed[:3]}. The plugin refuses a Linear it is neither told to decode nor "
            "told to leave alone, so this checkpoint would fail at load.")
    ignore = sorted(set(ignore))
    moe_passthrough_modules = {m for source in (expert_shapes, routed_shapes)
                               for name, shape in source.items()
                               for m in ignored_modules(name, shape)
                               if m not in stack_plan}
    config = src_config
    config["quantization_config"] = {
        # The field that selects Tessera's own vLLM plugin (entry point
        # ``tessera = tessera.serving:register``).  No serve flag enables it.
        # ``format`` is NOT derived here, unlike the stock twin below.  vLLM's
        # FP4-model predicate (``ModelConfig.is_nvfp4_quantized``) requires
        # ``quantization == "compressed-tensors"``, which this is not, and
        # ``tessera.serving.config`` reads ``config_groups`` and never this
        # field.  Naming a format here would change nothing and assert
        # something; the label stays generic and the record below says what the
        # predicate resolves to and why (#92).
        "quant_method": "tessera", "format": MIXED_PRECISION,
        "config_groups": config_groups, "ignore": ignore,
    }
    tessera_fp4_predicate = vllm_fp4_predicate("tessera", MIXED_PRECISION)
    config_name = "tessera_part_config.json" if args.partition else "config.json"
    (args.out / config_name).write_text(json.dumps(config, indent=2))
    if len(shards) > 1 or args.partition:
        size = sum((args.out / s).stat().st_size for s in shards)
        (args.out / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": size}, "weight_map": new_weight_map}, indent=2))
    aux_patterns = () if args.partition else ("*.json", "*.txt", "*.jinja", "*.model")
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
            # What the pinned runtime does with each planned module, from the
            # contract's construction block -- and which way this run resolved
            # a module it will not route.  ``unrouted`` non-empty together with
            # ``allow_unrouted`` true is an artifact whose declared wires the
            # runtime never executes.
            "construction_census": None if census is None else {
                "architecture": census["architecture"], "runtime": census["runtime"],
                "model": census["model"], "receipt": census.get("receipt")},
            "allow_unrouted": bool(args.allow_unrouted),
            "passthrough_unrouted": bool(args.passthrough_unrouted),
            "unrouted": unrouted_records,
        },
        # WHAT THE ROUTED-MoE LAYERS GOT, as a value rather than stdout.
        # Unplanned stacks stay at source precision; planned stacks carry wires.
        # ``modules`` is exactly the set of ``ignore``
        # entries this block accounts for -- read off ``ignored_modules``, the
        # same rule that put them there, so this cannot drift from the list it
        # claims to summarise (#86).  Both counts are zero on a dense model,
        # which is the same statement in the other direction.
        "routed_moe": {
            # ``disposition``, NOT ``structure``.  ``structure`` is the scheme's
            # gate-read field, whose only legal values are ``scheme.STRUCTURES``
            # and whose reserved name for this case is the block's own key,
            # ``routed_moe``.  A producer-written free string under that field
            # name is the confusion principle 14 exists to prevent, even though
            # nothing in ``tessera.serving`` reads this file.  It is now a
            # THREE-valued fact, because a stack can be encoded: ``quantized``
            # when every routed stack got a wire, ``passed_through_bf16`` when
            # none did, ``mixed`` when the plan named some of them.
            "disposition": ("quantized" if stack_plan and not moe_passthrough_modules
                            else "mixed" if stack_plan else "passed_through_bf16"),
            "reason": ("stacks named in the plan carry routed_moe wires; the rest stay at "
                       "source precision and are named in ignore"),
            "packed_source_tensors": len(expert_shapes),
            "unpacked_source_tensors": len(routed_shapes),
            "quantized_stacks": sorted(stack_plan),
            "quantized_source_tensors": len(expert_units),
            "quantized_logical_units": sum(
                len(record["units"]) for record in stack_plan.values()),
            # The ignore entries this block accounts for -- read off
            # ``ignored_modules``, the same rule that put them there, so it
            # cannot drift from the list it summarises (#86).  A stack that was
            # ENCODED is not here: it is in ``config_groups``, and the plugin
            # refuses a target that is both declared and ignored.
            "modules": sorted(moe_passthrough_modules),
        },
        # WHICH LANE THIS ARTIFACT WAS BUILT TO EXERCISE, in the bytes.  The
        # census reads it and REFUSES a phase where the lane took zero modules,
        # so "this checkpoint is for the window GEMV" is a claim the artifact
        # carries and a gate can check rather than a sentence in a handover
        # (issue #104).  Empty means no lane was requested, which is the honest
        # state of every checkpoint written before this flag existed.
        "requires_lanes": required_lanes,
        "stock_twin": str(twin) if twin is not None else None,
        "vllm_fp4_predicate": tessera_fp4_predicate,
        "totals": totals, "modules": module_records,
    }
    if partition_record is not None:
        partition_record["output_sha256"] = {
            shard: sha256_file(args.out / shard) for shard in sorted(shards)}
        manifest["export_partition"] = partition_record
    (args.out / "tessera_serving_manifest.json").write_text(json.dumps(manifest, indent=2))

    if twin is not None:
        twin_groups = {}
        if twin_modules[NVFP4]:
            twin_groups[f"group_{len(twin_groups)}"] = {
                "format": NVFP4_PACK_QUANTIZED, "weights": dict(NVFP4_WEIGHTS),
                "input_activations": dict(NVFP4_INPUTS), "targets": stock_targets(twin_modules[NVFP4])}
        if twin_modules[FP8]:
            twin_groups[f"group_{len(twin_groups)}"] = {
                "format": FLOAT_QUANTIZED, "weights": dict(FP8_WEIGHTS),
                "input_activations": dict(FP8_INPUTS), "targets": stock_targets(twin_modules[FP8])}
        twin_config = json.loads((args.src / "config.json").read_text())
        # A BF16 module is in no group: its twin tensor is the ordinary
        # ``<module>.weight``, so it is ignored by the quantization config
        # exactly as ``lm_head`` is.  And when every module decoded to a plain
        # bf16 tile there are no groups at all -- this IS a BF16 checkpoint, and
        # the helper returns nothing to declare rather than an empty config that
        # would tell a runtime to look for compressed tensors that do not exist.
        #
        # This is the comparator arm, so it is the artifact #92 was about: it
        # goes through the same derivation as ``export_stock_compressed``, and
        # the predicate it resolves to is recorded on the twin's own manifest.
        twin_quant_config, twin_fp4_predicate = stock_quantization_config(
            twin_groups, sorted(set(ignore) | set(twin_modules[BF16])))
        if twin_quant_config is not None:
            twin_config["quantization_config"] = twin_quant_config
        else:
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
            "vllm_fp4_predicate": twin_fp4_predicate,
            # THE TWIN IS THE ARTIFACT THAT GETS SERVED, so it carries what
            # shaped its bytes rather than a path to something that does.  The
            # wire manifest has these three; the twin used to have only a
            # ``wire_checkpoint`` string pointing at it, which is provenance
            # only for as long as that directory outlives this one on this box
            # -- and a served KL is read off the twin, quoted long afterwards,
            # and compared against another twin.  Two arms of an A/B whose one
            # difference is an ``activation_aware`` field could not be told
            # apart from the artifacts that produced the numbers: ``arm``
            # reads "tessera E2M1x2 q256=896 -> ..." for both (tessera#60).
            # Nothing here changes a byte of the checkpoint.
            "default": manifest["default"],
            "input_scales_from": manifest["input_scales_from"],
            "activation_aware": manifest["activation_aware"],
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
