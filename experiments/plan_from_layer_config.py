#!/usr/bin/env python
"""Turn a PrismaQuant ``layer_config.json`` into the Tessera exporter's ``--plan-json``.

PrismaQuant's allocator writes a per-Linear assignment keyed by *qname*
(``model.layers.0.mlp.gate_proj``) carrying ``tessera_format`` in the
``TESSERA_<BASE>_K<arity>_R<rung>`` spelling.  ``export_tessera_serving.py``
wants a per-*tensor* plan (``model.layers.0.mlp.gate_proj.weight``) whose value
is ``{"grid": ..., "q256": ...}`` or the string ``"BF16"``.  This script is the
only place that translation lives, and it does three jobs beyond renaming:

**It refuses what cannot be served in one checkpoint.**  A non-Tessera
*quantised* choice (NVFP4, FP8_DYNAMIC, ...) has no route in the Tessera plugin,
so a plan that mixed one in would either be dropped silently by the exporter's
``unknown`` check or exported as something the allocator did not choose.  Those
are counted and refused by name.  A BF16 choice is not a refusal -- it is a
plain BF16 module, which is exactly what the exporter's passthrough does.

**It checks the fused-group invariant before the exporter does.**  vLLM builds
ONE method per fused module, so ``q/k/v`` must agree on ``(grid, q256)`` and so
must ``gate/up``.  The exporter's answer to a disagreement is to pass the whole
group through as BF16 and print it; that is the right answer at export time and
the wrong thing to discover after a 20-minute encode, so a disagreement is
reported here, up front, with the members and their rungs.

That is the case a **per-member (mink) allocation** lands in: PrismaQuant's
group knapsack gives one family per fused group and a rate per member, and this
serving path cannot express it.  **Refusing is the default, and it is the
answer**, because no single rate for the group is derivable from the objective
-- the members' rates differ precisely because their sensitivities do, and min
/ bytes-weighted / max are all taste, not arithmetic; and any of them moves
both the bytes and the loss off the point the DP chose, so the allocation that
served would not be the allocation that was selected.  ``--allow-fused-
disagreement`` is for the operator who wants the plan anyway, and what it now
writes is the plan that will **serve**: every member of a disagreeing group is
named ``"BF16"``, dropped from the unit table and the charged-bits total, and
the demotion is recorded (``fused_disagreements[].planned_as``,
``totals.demoted_to_bf16_params``).  Before that, the plan named Tessera rungs
for members the exporter was about to pass through, and the sidecar priced them
as Tessera -- a three-to-four-fold under-report in the one currency the byte
budget is spent in, produced by the file whose stated job is that the bytes
served are the bytes priced.

A whole-GROUP option name (``TESSERA_E4M3_K1_G3``) is refused by name for the
same reason: it is not a rung, no rate stands for it, and PrismaQuant is meant
to have expanded it to its members before the assignment was written.

**It says whether the plan covers the model.**  A PrismaQuant campaign may price
a *subset* -- ``--layer-stride 28`` prices decoder layer 0 and nothing else --
and an allocation over 7 of 197 Linears exports to a checkpoint that is 96%
BF16.  Two modes, and neither is the default guess:

* ``--cover as-allocated`` plans exactly the units the allocation names, and
  names every other body Linear ``"BF16"`` **explicitly** -- silence is not
  BF16, it is the exporter's ``--grid``/``--q256`` default, which is a 4-bit
  rung.  This is the allocation, unextrapolated.
* ``--cover broadcast-by-role`` applies the allocation's per-ROLE assignment at
  every depth.  This is an EXTRAPOLATION and is stamped as one in the sidecar
  (``coverage.mode``, ``coverage.extrapolated: true``): the allocator priced one
  layer and nothing here says the same rate is right at depth 27.  It is
  refused unless the allocation is single-layer and every target layer's shape
  for that role matches the priced one, because a broadcast across differing
  shapes is a different rate (the CHANNEL plane amortises over rows).

The sidecar ``<out>.provenance.json`` carries the source path, the allocation's
own ``__prismaquant__`` block, the coverage decision, and a per-unit table with
each unit's shape, rung, wire bytes as PrismaQuant charged them
(``prismaquant.tessera_formats.artifact_bpp``, when ``--prismaquant`` points at
a tree that has it) and the totals.  That table is what an export is checked
against: the bytes served must be the bytes priced.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tessera.control import (  # noqa: E402
    control_block,
    uniform_control,
    units_from_plan,
)
from tessera.errors import TesseraError  # noqa: E402

#: ``TESSERA_<BASE>_K<arity>_R<rung>`` -- the allocator's format spelling.
FORMAT = re.compile(r"^TESSERA_(?P<base>[A-Z0-9]+)_K(?P<arity>\d+)_R(?P<rung>\d+)$")
FAMILY = re.compile(r"^TESSERA_(?P<base>[A-Z0-9]+)_K(?P<arity>\d+)$")
#: ``TESSERA_<BASE>_K<arity>_G<n>`` -- a whole fused GROUP at one family with a
#: rung per member, the option PrismaQuant's group knapsack builds.  It is not a
#: rung and there is no rate it could stand for, so it is named here in order to
#: be refused by name: ``expand_fused_sibling_assignment`` is supposed to have
#: replaced it with the members' own rungs before an assignment is written, and
#: one that reaches a plan means that expansion did not happen.
GROUP = re.compile(r"^TESSERA_(?P<base>[A-Z0-9]+)_K(?P<arity>\d+)_G(?P<option>\d+)$")

#: The exporter's fused groups, as ``export_tessera_serving.FUSED`` spells them.
FUSED = (("self_attn", "qkv_proj", ("q_proj", "k_proj", "v_proj")),
         ("mlp", "gate_up_proj", ("gate_proj", "up_proj")))

#: What the allocator may pick that is not a Tessera wire and is still fine.
BF16_CHOICES = {"BF16", "bfloat16", "bf16"}


class PlanError(SystemExit):
    pass


def grid_of(family: str) -> str:
    """``TESSERA_E4M3_K1 -> "E4M3"``, ``TESSERA_E2M1_K2 -> "E2M1x2"``.

    The exporter's ``--grid`` vocabulary spells arity as an ``xN`` suffix and
    arity 1 as a bare base, which is the same object under a different name.
    """
    match = FAMILY.match(family)
    if not match:
        raise PlanError(f"not a Tessera family name: {family!r}")
    arity = int(match.group("arity"))
    return match.group("base") + ("" if arity == 1 else f"x{arity}")


def parse_entry(qname: str, entry) -> tuple:
    """``(kind, payload)``: ``("tessera", (grid, q256))``, ``("bf16", None)`` or ``("other", label)``."""
    if isinstance(entry, str):
        return ("bf16", None) if entry in BF16_CHOICES else ("other", entry)
    if not isinstance(entry, dict):
        raise PlanError(f"{qname}: unreadable layer_config entry {entry!r}")
    fmt = entry.get("tessera_format")
    if fmt:
        if GROUP.match(fmt):
            raise PlanError(
                f"{qname}: {fmt!r} is a whole-GROUP option (one family, a rung per member), "
                "not a rung, and there is no single rate it could stand for -- the members "
                "have different shapes and different sensitivities, which is the entire "
                "reason the option exists.  PrismaQuant expands it to its members' own rungs "
                "in expand_fused_sibling_assignment before writing an assignment; a plan that "
                "still carries the group name means that expansion did not run.  Re-export "
                "the layer_config from an allocator that expands it.")
        match = FORMAT.match(fmt)
        if not match:
            raise PlanError(f"{qname}: {fmt!r} is not the TESSERA_<BASE>_K<arity>_R<rung> spelling")
        family = f"TESSERA_{match.group('base')}_K{match.group('arity')}"
        declared = entry.get("tessera_family")
        if declared and declared != family:
            raise PlanError(f"{qname}: tessera_family {declared!r} disagrees with tessera_format {fmt!r}")
        rung = int(match.group("rung"))
        body_q256 = entry.get("tessera_body_rate_q256")
        if body_q256 is not None and int(body_q256) != rung:
            raise PlanError(f"{qname}: tessera_body_rate_q256 {body_q256} disagrees with {fmt!r}")
        return ("tessera", (grid_of(family), rung, family))
    label = entry.get("data_type") or entry.get("format") or entry.get("bits")
    if str(label) in BF16_CHOICES:
        return ("bf16", None)
    return ("other", str(label))


def body_weights(model: Path) -> dict:
    """``{tensor name: (rows, cols)}`` for every 2-D ``model.layers.*.weight``."""
    index = model / "model.safetensors.index.json"
    if index.exists():
        shards = sorted({s for s in json.loads(index.read_text())["weight_map"].values()})
    else:
        shards = sorted(p.name for p in model.glob("*.safetensors"))
    shapes = {}
    for shard in shards:
        with safe_open(str(model / shard), framework="pt") as handle:
            for name in handle.keys():
                if name.startswith("model.layers.") and name.endswith(".weight"):
                    shape = tuple(handle.get_slice(name).get_shape())
                    if len(shape) == 2:
                        shapes[name] = shape
    if not shapes:
        raise PlanError(f"no 2-D model.layers.*.weight tensors in {model}")
    return shapes


def role_of(qname: str) -> str:
    return qname.rsplit(".", 1)[-1]


def layer_of(qname: str) -> int:
    parts = qname.split(".")
    return int(parts[2])


def fused_key(qname: str):
    """``(fused module qname, ordered member qnames)`` or ``None``."""
    prefix, role = qname.rsplit(".", 1)
    for block, fused, members in FUSED:
        if prefix.endswith("." + block) and role in members:
            return f"{prefix}.{fused}", tuple(f"{prefix}.{m}" for m in members)
    return None


def charged_bits(prismaquant: "Path | None", family: str, rung: int, shape) -> "Fraction | None":
    """PrismaQuant's own charged wire bits for this unit, or ``None`` if unavailable.

    The allocator's byte budget is spent in this currency, so it is the number
    an export must reproduce.  Imported from the PrismaQuant tree rather than
    reimplemented: two accountings of one wire is the drift this check exists
    to catch.
    """
    if prismaquant is None:
        return None
    if str(prismaquant) not in sys.path:
        sys.path.insert(0, str(prismaquant))
    try:
        from prismaquant.tessera_formats import artifact_bpp
    except Exception as exc:                                    # pragma: no cover - env
        print(f"  (no PrismaQuant accounting: {exc})", flush=True)
        return None
    rows, cols = shape
    return Fraction(artifact_bpp(family, rung, shape=(rows, cols))) * rows * cols


def uniform_control_block(plan: dict, shapes: dict, *, rule: str = "nearest"):
    """The byte-matched uniform arm this plan has to beat, as a record.

    An allocation over a rate axis is a *claim* -- that choosing rungs beats
    spending the same bytes at one rung -- and on 2026-09-02 that claim was
    false by 2.00x while every other check in the pipeline passed.  So the
    sidecar carries the arm that tests it, priced but not served, next to the
    bpp (tessera#3, principle 12).

    It records rather than refuses: the converter's job is to write the plan it
    was given, and a plan whose control cannot be byte-matched (two families,
    or the 0.239-bpp hole below the E2M1x2 coset cap) is still a plan.  What it
    must never do is stay silent about it, so the reason lands in the block.
    ``experiments/uniform_control.py`` is where the match is *asserted*, because
    that is where the arm you would actually build gets written.
    """
    try:
        units = units_from_plan(plan, shapes)
        control = uniform_control(units, rule=rule, assert_match=False)
    except TesseraError as exc:
        return {"schema": "tessera.uniform_control.v1", "built": False,
                "refusal": str(exc)}
    block = control_block(control)
    block["built"] = True
    return block


def build(config: dict, shapes: dict, *, cover: str, allow_disagreement: bool,
          prismaquant: "Path | None", control_rule: str = "nearest",
          with_control: bool = True):
    meta = config.get("__prismaquant__")
    assignment = {k: v for k, v in config.items() if not k.startswith("__")}
    if not assignment:
        raise PlanError("layer_config names no units")

    tessera, bf16, other = {}, [], {}
    for qname, entry in sorted(assignment.items()):
        kind, payload = parse_entry(qname, entry)
        if kind == "tessera":
            tessera[qname] = payload
        elif kind == "bf16":
            bf16.append(qname)
        else:
            other[qname] = payload
    if other:
        counts = collections.Counter(other.values())
        sample = sorted(other)[:5]
        raise PlanError(
            f"{len(other)} unit(s) carry a non-Tessera QUANTISED choice "
            f"({dict(counts)}); the Tessera plugin serves TESSERA_* wires only, so one "
            f"checkpoint cannot hold these and a Tessera wire at the same time.  Units, "
            f"first five: {sample}.  BF16 is not in this count -- a BF16 choice is a plain "
            f"BF16 module and is planned as one.")

    priced_layers = sorted({layer_of(q) for q in tessera} | {layer_of(q) for q in bf16})
    all_layers = sorted({layer_of(t[: -len(".weight")]) for t in shapes})

    plan, units, broadcast_from = {}, [], None
    if cover == "as-allocated":
        chosen = dict(tessera)
        for qname in bf16:
            plan[qname + ".weight"] = "BF16"
    elif cover == "broadcast-by-role":
        if len(priced_layers) != 1:
            raise PlanError(
                f"--cover broadcast-by-role needs a single-layer allocation to broadcast; this "
                f"one names layers {priced_layers}.  Broadcasting a multi-layer allocation would "
                f"have to invent a rule for which layer's rate wins.")
        broadcast_from = priced_layers[0]
        by_role = {role_of(q): payload for q, payload in tessera.items()}
        bf16_roles = {role_of(q) for q in bf16}
        chosen = {}
        for tensor, shape in shapes.items():
            qname = tensor[: -len(".weight")]
            role = role_of(qname)
            source = f"model.layers.{broadcast_from}.{qname.split('.', 3)[3]}.weight"
            if role in bf16_roles and role not in by_role:
                plan[tensor] = "BF16"
                continue
            if role not in by_role:
                plan[tensor] = "BF16"        # unpriced role: say BF16, do not assume it
                continue
            if source in shapes and shapes[source] != shape:
                raise PlanError(
                    f"{qname} is {shape} but the priced {source} is {shapes[source]}; a rung is "
                    f"a rate on a SHAPE (the CHANNEL plane amortises over rows), so broadcasting "
                    f"it onto a different shape would be a different rate.  Use "
                    f"--cover as-allocated.")
            chosen[qname] = by_role[role]
    else:                                                       # pragma: no cover - argparse
        raise PlanError(f"unknown coverage mode {cover!r}")

    # A tensor the plan does not name is NOT a BF16 module: the exporter falls
    # back to its own --grid/--q256 default, which is E2M1x2 q256=896 unless a
    # caller happens to override it.  A plan that covers seven Linears and
    # leaves 189 to that default is a 4-bit NVFP4 checkpoint with seven Tessera
    # units in it, priced as neither -- so name every remaining body Linear
    # BF16 explicitly and let the plan, not an exporter default, be the record
    # of what the allocation said.
    for tensor in shapes:
        if tensor not in plan and tensor[: -len(".weight")] not in chosen:
            plan[tensor] = "BF16"

    # The fused invariant, checked before the encode rather than after it.
    groups, disagreements = {}, []
    for qname in chosen:
        key = fused_key(qname)
        if key is None:
            continue
        module, members = key
        if module in groups:
            continue
        groups[module] = members
        present = [m for m in members if m in chosen]
        recipes = {(chosen[m][0], chosen[m][1]) for m in present}
        if len(present) != len(members) or len(recipes) != 1:
            disagreements.append({
                "module": module,
                "members": {m: (f"{chosen[m][2]}_R{chosen[m][1]}" if m in chosen else "ABSENT")
                            for m in members},
            })
    if disagreements and not allow_disagreement:
        raise PlanError(
            f"{len(disagreements)} fused module(s) do not share one (grid, q256): "
            f"{json.dumps(disagreements[:3], indent=2)}\n"
            f"vLLM builds ONE quantization method per fused module, so the exporter would pass "
            f"the whole group through as BF16.  That is a finding about the allocation -- it "
            f"chose an assignment this serving path cannot express.  Re-run with "
            f"--allow-fused-disagreement to write the plan anyway and let the exporter's own "
            f"passthrough handle it.")

    # The override writes a plan, and what it must write is the plan that will
    # SERVE.  The exporter's answer to a disagreeing group is to drop every
    # member and pass the module through (``export_tessera_serving``: "roles
    # disagree ...; vLLM builds one method per fused module"), so a plan that
    # still names Tessera rungs for those members describes an encode that will
    # not happen, and the sidecar -- whose whole job is "the bytes served must
    # be the bytes priced" -- reports a rate three to four times below the one
    # the checkpoint will carry.  Recording the demotion is not the converter
    # ROUNDING the allocation to a rate of its own invention: no single rate is
    # derivable from the members' (a mink group exists precisely because their
    # sensitivities differ, and min / bytes-weighted / max are all taste), and
    # inventing one would move the point the DP chose off the frontier it was
    # chosen on.  What exists is the exporter's own resolution, and this makes
    # it visible before the encode instead of after it.
    demoted_params = 0
    for entry in disagreements:
        entry["planned_as"] = "BF16"
        entry["reason"] = (
            "vLLM builds one quantization method per fused module; the "
            "exporter passes the whole group through at source precision"
        )
        members = groups[entry["module"]]
        entry["demoted_params"] = {}
        for member in members:
            chosen.pop(member, None)
            tensor = member + ".weight"
            if tensor in shapes:
                plan[tensor] = "BF16"
                rows, columns = shapes[tensor]
                entry["demoted_params"][member] = rows * columns
                demoted_params += rows * columns

    total_params, total_charged = 0, Fraction(0)
    for qname, (grid, rung, family) in sorted(chosen.items()):
        tensor = qname + ".weight"
        if tensor not in shapes:
            raise PlanError(f"{qname} is not a 2-D body Linear in the model: no tensor {tensor}")
        rows, cols = shapes[tensor]
        plan[tensor] = {"grid": grid, "q256": rung}
        bits = charged_bits(prismaquant, family, rung, (rows, cols))
        total_params += rows * cols
        if bits is not None:
            total_charged += bits
        units.append({
            "tensor": tensor, "qname": qname, "role": role_of(qname), "layer": layer_of(qname),
            "family": family, "grid": grid, "q256": rung, "rows": rows, "columns": cols,
            "params": rows * cols,
            "prismaquant_charged_bits": None if bits is None else float(bits),
            "prismaquant_charged_bits_exact": None if bits is None else [bits.numerator, bits.denominator],
            "prismaquant_charged_bpp": None if bits is None else float(bits / (rows * cols)),
        })

    provenance = {
        "schema": "tessera.plan_from_layer_config.v1",
        "source_layer_config": None,                             # filled by main
        "prismaquant_meta": meta,
        "coverage": {
            "mode": cover,
            "extrapolated": cover == "broadcast-by-role",
            "broadcast_from_layer": broadcast_from,
            "allocation_units": len(tessera) + len(bf16),
            "allocation_layers": priced_layers,
            "model_body_linears": len(shapes),
            "model_layers": len(all_layers),
            "planned_tessera_units": len(chosen),
            "planned_bf16_units": sum(1 for v in plan.values() if v == "BF16"),
            "unplanned_body_linears": len(shapes) - len(plan),
            "note": ("every body Linear the allocation did not name is planned as BF16 "
                     "explicitly, because an unnamed tensor takes the exporter's --grid default, "
                     "not a passthrough" if cover == "as-allocated" else
                     "the allocation's per-ROLE assignment applied at every depth; the allocator "
                     "priced only the layer(s) above and nothing here says the same rate is right "
                     "at another depth"),
        },
        "fused_disagreements": disagreements,
        "fused_disagreement_policy": (
            "refused" if not disagreements else
            "demoted_to_bf16_by_--allow-fused-disagreement"
        ),
        "totals": {
            "tessera_units": len(chosen),
            "quantized_params": total_params,
            # Params the allocation gave a Tessera rung and the plan gives
            # BF16.  Non-zero only under --allow-fused-disagreement, and it is
            # the machine-readable form of "the allocation served is not the
            # allocation that was chosen".
            "demoted_to_bf16_params": demoted_params,
            "prismaquant_charged_bits": float(total_charged) if total_charged else None,
            "prismaquant_charged_bpp": (float(total_charged / total_params)
                                        if total_charged and total_params else None),
        },
        "units": units,
    }
    if with_control:
        provenance["uniform_control"] = uniform_control_block(
            plan, shapes, rule=control_rule)
    return plan, provenance


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("layer_config", type=Path, help="PrismaQuant layer_config.json")
    ap.add_argument("model", type=Path, help="the source checkpoint, read for tensor names and shapes")
    ap.add_argument("out", type=Path, help="the exporter's --plan-json")
    ap.add_argument("--cover", choices=("as-allocated", "broadcast-by-role"), default="as-allocated",
                    help="as-allocated: plan exactly what the allocation names (default).  "
                         "broadcast-by-role: apply its per-role assignment at every depth "
                         "(an EXTRAPOLATION, stamped as one in the sidecar)")
    ap.add_argument("--allow-fused-disagreement", action="store_true",
                    help="write the plan even when a fused module's members disagree.  The whole "
                         "group is then planned BF16 -- which is what the exporter does with it -- "
                         "and the demotion is recorded in the sidecar "
                         "(fused_disagreements[].planned_as, totals.demoted_to_bf16_params), so "
                         "the plan states the allocation that will SERVE and not the one that was "
                         "chosen")
    ap.add_argument("--write-uniform-plan", type=Path, default=None,
                    help="also write the byte-matched UNIFORM control plan here -- the arm "
                         "the candidate has to beat at the same bytes (tessera#3).  The byte "
                         "match is ASSERTED before it is written; a plan whose control cannot "
                         "be matched refuses rather than exporting a second byte budget")
    ap.add_argument("--control-rule", choices=("nearest", "no_larger"), default="nearest",
                    help="nearest: minimise |candidate - control| bytes (default).  no_larger: "
                         "the heaviest rung that does not outweigh the candidate")
    ap.add_argument("--no-uniform-control", action="store_true",
                    help="do not price the uniform control into the sidecar")
    ap.add_argument("--prismaquant", type=Path, default=None,
                    help="a PrismaQuant tree, imported read-only for its own wire accounting "
                         "(prismaquant.tessera_formats.artifact_bpp) so the sidecar carries the "
                         "bits the allocator charged")
    args = ap.parse_args(argv)

    config = json.loads(args.layer_config.read_text())
    shapes = body_weights(args.model)
    plan, provenance = build(config, shapes, cover=args.cover,
                             allow_disagreement=args.allow_fused_disagreement,
                             prismaquant=args.prismaquant,
                             control_rule=args.control_rule,
                             with_control=not args.no_uniform_control)
    provenance["source_layer_config"] = str(args.layer_config.resolve())
    provenance["model"] = str(args.model.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=2, sort_keys=True))
    sidecar = args.out.with_suffix(args.out.suffix + ".provenance.json")
    sidecar.write_text(json.dumps(provenance, indent=2))

    cov = provenance["coverage"]
    print(f"{args.layer_config}")
    print(f"  allocation: {cov['allocation_units']} unit(s) on layer(s) {cov['allocation_layers']}"
          f" of {cov['model_body_linears']} body Linears over {cov['model_layers']} layers")
    print(f"  coverage {cov['mode']}"
          + ("  [EXTRAPOLATED from layer "
             f"{cov['broadcast_from_layer']}]" if cov["extrapolated"] else ""))
    print(f"  planned: {cov['planned_tessera_units']} Tessera, {cov['planned_bf16_units']} BF16, "
          f"{cov['unplanned_body_linears']} unnamed (must be 0: an unnamed tensor takes the "
          f"exporter's --grid default, which is a 4-bit rung, not BF16)")
    by_rung = collections.Counter((u["family"], u["q256"]) for u in provenance["units"])
    for (family, rung), n in sorted(by_rung.items()):
        print(f"    {family}_R{rung}: {n}")
    demoted = provenance["totals"]["demoted_to_bf16_params"]
    if demoted:
        modules = len(provenance["fused_disagreements"])
        print(f"  DEMOTED: {modules} fused module(s) whose members took different rungs are "
              f"planned BF16 ({demoted} params, "
              f"{100.0 * demoted / max(demoted + provenance['totals']['quantized_params'], 1):.1f}% "
              f"of the allocated body).  This plan is NOT the allocation that was chosen; "
              f"--allow-fused-disagreement asked for the exporter's passthrough and this is it.")
    totals = provenance["totals"]
    if totals["prismaquant_charged_bpp"] is not None:
        print(f"  PrismaQuant charges {totals['prismaquant_charged_bits']:.0f} bits over "
              f"{totals['quantized_params']} params = "
              f"{totals['prismaquant_charged_bpp']:.6f} bpp")
    block = provenance.get("uniform_control")
    if block and block.get("built"):
        control = block["control"]
        match = control["match"]
        print(f"  uniform control: {control['grid']} R{control['q256']} at "
              f"{match['control_bpp']:.6f} bpp against the candidate's "
              f"{match['candidate_bpp']:.6f} "
              f"({match['relative_slack_ppm']:.1f} ppm, the {match['fatter_arm']} arm is "
              f"fatter){'' if match['byte_matched'] else '  NOT BYTE-MATCHED'}")
        print("    unserved: this is the arm the candidate has to beat at the same bytes")
    elif block:
        print(f"  uniform control: NOT BUILT -- {block['refusal']}")
    if args.write_uniform_plan is not None:
        units = units_from_plan(plan, shapes)
        control = uniform_control(units, rule=args.control_rule)
        args.write_uniform_plan.parent.mkdir(parents=True, exist_ok=True)
        args.write_uniform_plan.write_text(
            json.dumps(control.plan, indent=2, sort_keys=True))
        print(f"  -> {args.write_uniform_plan}  (uniform {control.grid} "
              f"R{control.q256}, the control arm)")
    print(f"  -> {args.out}\n  -> {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
