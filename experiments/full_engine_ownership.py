"""Replay-time ownership derivation for the full-engine resource ledger (tessera#399).

The census inside the worker owns what it can see at a checkpoint: parameters
and buffers, the KV backings, the runner's persistent roots, the BLAS
workspace and the native boundary tensors. Everything else -- every transient
the engine or the plugin allocates and frees inside a step, and every
resident the census never saw -- reached the ledger with no owner, and a row
with no owner nulls every composition term.

This module assigns those rows an owner from a DECLARED rule applied to the
capture's own evidence, and nothing else:

* the allocation-site frames Torch's allocator history recorded for every
  allocation (``record_memory_history(stacks="all")``), read against the
  run's own package inventories -- the installer's plugin file roster and the
  attested vLLM core manifest -- so a site is a file the run attested, not a
  string;
* the checkpoint order of the same history, so an allocation that precedes
  ``before_model_load`` is known to predate any assignment;
* the plan's canonical roster, so a candidate row resolves to the unit it
  belongs to.

Observed ownership is never rewritten. ``observed_owners`` and
``observed_categories`` stay the census's raw record; the derivation travels
beside them as ``owner_views``, one view per allocation, each naming the rule
that fired or the rule that abstained and why. Unknown stays null: a row no
rule can place is listed, not bucketed, and it keeps every term null.

What this module does NOT decide is whether a census ``shared`` row --
a persistent runner root, the BLAS workspace, a native boundary tensor -- is
assignment-invariant. That is tessera#548's measurement (two captures under
two assignments). The only ``shared`` rows this module classifies are the
ones the capture itself proves predate any assignment by history order. For
the boundary tensors it computes the within-capture cross-family geometry
witness #548 asked for and carries it as an observation.
"""
from collections import Counter
import json
from pathlib import PurePosixPath

OWNER_VIEWS_SCHEMA = "tessera.full_engine_owner_views.v1"
EXTERNAL_RECORDS_SCHEMA = "tessera.full_engine_external_records.v1"
GEOMETRY_WITNESS_SCHEMA = "tessera.full_engine_boundary_geometry_witness.v1"
TRANSIENT_WITNESS_SCHEMA = "tessera.full_engine_transient_gap_witness.v1"
DENSE_STARTUP_CHECK_SCHEMA = "tessera.full_engine_dense_startup_check.v1"

#: The classes a derived view may carry. ``observer`` is the observer's own
#: footprint: named, counted, and charged to no serve term.
VIEW_CLASSES = ("fixed", "candidate", "kv", "observer")

#: The plugin's per-family route modules under ``tessera/serving``. A
#: load-time allocation whose frames pass through one of these files belongs
#: to that family's units; the table is the plugin tree's own module layout,
#: stated here so the rule is readable without importing the serving package.
ROUTE_FAMILIES = {
    "serving/nvfp4_route.py": "TESSERA_NVFP4",
    "serving/fp8_route.py": "TESSERA_FP8",
    "serving/bf16_route.py": "TESSERA_BF16",
}

#: Every rule, stated once. ``basis`` is the invariance claim a class makes and
#: what backs it; ``abstains`` is what the rule leaves null.
RULES = {
    "census": {
        "class": "observed",
        "statement": "the worker census observed this storage at a checkpoint and named its "
                     "owner and category; the derivation repeats the census's class and, for a "
                     "candidate row loaded outside every unit interval, resolves the unit from the "
                     "owner's parameter/buffer path through the plan's roster",
        "basis": "observed ownership; never rewritten here",
    },
    "history_order": {
        "class": "fixed",
        "statement": "a census-shared row allocated before the before_model_load checkpoint "
                     "predates the model, its assignment and every route; it is fixed by the "
                     "capture's own event order",
        "basis": "history order (allocate_index < index(before_model_load)); consumer-recomputable",
    },
    "site:plugin": {
        "class": "candidate",
        "statement": "the innermost non-Torch Python frame of the allocation site is a file of the "
                     "installed Tessera plugin package (the installer's own file roster); the plugin "
                     "allocates only for the units it serves, so the row is a candidate resource of "
                     "the unit in scope",
        "basis": "allocation-site file in per-job-runtime.json plugin_files; unit from the outermost "
                 "scope-stack unit, else the census owner path through the roster, else the "
                 "load-time route module's family when the roster has exactly one unit of it",
    },
    "site:vllm": {
        "class": "fixed",
        "statement": "the innermost non-Torch Python frame is a file of the attested vLLM core "
                     "(runtime-inventory.json): stock engine code outside the quant-method "
                     "boundary, which reads no assignment",
        "basis": "allocation-site file in the vLLM core manifest; per-layer invariance witnessed "
                 "within this capture by transient_gap_witness (the same transient pattern after "
                 "units of different families), never asserted across captures",
    },
    "site:image": {
        "class": "fixed",
        "statement": "the innermost non-Torch Python frame is a file of another package of the "
                     "pinned image's site-packages (neither the plugin nor vLLM core), reached "
                     "from stock engine code",
        "basis": "allocation-site file under the pinned image's site-packages root; the image is "
                 "bound by digest, the package is not separately inventoried",
    },
    "site:observer": {
        "class": "observer",
        "statement": "the innermost non-Torch Python frame is a file of the observer tree "
                     "(the PYTHONPATH roots outside site-packages); the observer's own footprint, "
                     "charged to no serve term and reported as such",
        "basis": "allocation-site file under an observer root from the worker's recorded sys.path",
    },
    "two_capture:agreed": {
        "class": "fixed",
        "statement": "a census-shared site the substitution could have moved -- it names no unit, "
                     "or its unit changed family between the two captures -- whose matched bytes "
                     "are identical in both: the assignment changed and these bytes did not",
        "basis": "equal site bytes across the two captures named in the boundary classification, "
                 "on a site whose unit changed family there or which names no unit; a site whose "
                 "unit kept its family agrees by construction and stays pending_548. Evidence for "
                 "THOSE TWO captures, never a universal invariance, and consumer-recomputable "
                 "from the classification's own byte and family columns",
    },
    "two_capture:moved": {
        "class": "candidate",
        "statement": "a census-shared site whose matched bytes differ between the two captures: "
                     "its bytes move with the selected assignment, so it is a candidate resource "
                     "and owes a unit",
        "basis": "differing site bytes across the two captures named in the boundary "
                 "classification; consumer-recomputable from the same byte columns",
    },
    "pending_548": {
        "class": None,
        "statement": "a census-shared row allocated after before_model_load: a persistent runner "
                     "root, the BLAS workspace, a library cache buffer or a native boundary tensor; "
                     "whether its bytes are assignment-invariant is tessera#548's two-assignment "
                     "measurement, not a rule this capture can state",
        "basis": None,
    },
}


# ------------------------------------------------------------------ sites ---

def allocation_site(frames, torch_root):
    """The innermost Python frame outside the Torch package, or ``None``.

    Torch's history records the C++ unwind first and the Python frames
    innermost-first after it. The Torch package's own Python frames
    (``torch/_tensor.py`` and friends) are the allocator's, not the caller's,
    so the first frame outside them is the site that asked for the bytes.
    """
    torch_root = str(torch_root).rstrip("/") + "/"
    for frame in frames or ():
        filename = frame.get("filename") or ""
        if filename.endswith(".py") and not filename.startswith(torch_root):
            return {"file": filename, "name": frame.get("name"), "line": frame.get("line")}
    return None


def site_package(site, evidence):
    """``(package, relative_path)`` for one allocation site, or ``(None, None)``.

    ``evidence`` carries the run's own inventories: ``plugin_package_path`` and
    ``plugin_files`` (the installer's roster, relative to the package),
    ``vllm_root`` and ``vllm_files`` (the attested core manifest), and
    ``observer_roots`` (the worker's PYTHONPATH roots outside site-packages).
    A file under the plugin or vLLM root that the inventory does not list is
    NOT classified: the run attested a roster, and a file outside it is a file
    nothing attested.
    """
    if site is None:
        return None, None
    path = PurePosixPath(site["file"])
    plugin = PurePosixPath(evidence["plugin_package_path"])
    vllm = PurePosixPath(evidence["vllm_root"])
    if _under(path, plugin):
        relative = str(path.relative_to(plugin))
        return ("plugin", relative) if relative in evidence["plugin_files"] else (None, None)
    if _under(path, vllm):
        relative = str(path.relative_to(vllm))
        return ("vllm", relative) if relative in evidence["vllm_files"] else (None, None)
    site_packages = plugin.parent
    if _under(path, site_packages):
        return "image", str(path.relative_to(site_packages))
    for root in evidence.get("observer_roots", ()):
        root = PurePosixPath(root)
        if _under(path, root):
            return "observer", str(path.relative_to(root))
    return None, None


def _under(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def route_family_of(frames, evidence):
    """The family whose route module the frames pass through, or ``None``.

    Reads every Python frame, not only the innermost: a load-time table is
    allocated by ``tessera/decode.py`` under ``nvfp4_route.process_weights_
    after_loading``, and it is the route frame that names the family.
    Refuses (returns ``None``) when frames pass through more than one route.
    """
    plugin = PurePosixPath(evidence["plugin_package_path"])
    families = set()
    for frame in frames or ():
        filename = frame.get("filename") or ""
        path = PurePosixPath(filename)
        if not _under(path, plugin):
            continue
        family = ROUTE_FAMILIES.get(str(path.relative_to(plugin)))
        if family is not None:
            families.add(family)
    return families.pop() if len(families) == 1 else None


# ------------------------------------------------------------ owner views ---

def _roster_index(roster):
    """``{module_path: unit_id}`` over the roster's modules and member modules."""
    by_module, by_family = {}, {}
    for row in roster:
        by_module[row["module"]] = row["unit_id"]
        for member in row.get("members", ()):
            module, _, _leaf = member.rpartition(".")
            by_module.setdefault(module, row["unit_id"])
        # A census-built roster (capture_full_engine_resources.canonical_roster)
        # carries no family; it groups under None rather than raising into the
        # analyzer's swallowed-issue path.
        by_family.setdefault(row.get("family"), []).append(row["unit_id"])
    return by_module, by_family


def _unit_from_owner_path(owners, by_module):
    """The roster unit a ``model:parameter|buffer:<path>`` owner names, or ``None``."""
    units = set()
    for owner in owners:
        if not owner.startswith("model:"):
            continue
        _kind, _, path = owner.split(":", 2)
        module, _, _leaf = path.rpartition(".")
        # Fused vLLM modules (qkv_proj, gate_up_proj) are roster modules; the
        # manifest's HF-named members (q_proj, gate_proj) are member modules.
        # Walk outward until a roster module matches, so a leaf under a
        # sub-module of a unit still resolves to that unit.
        while module:
            if module in by_module:
                units.add(by_module[module])
                break
            module, _, _ = module.rpartition(".")
    return units.pop() if len(units) == 1 else None


def _two_capture_sites(classification, capture_sha256):
    """``{owner: site}`` from a boundary classification that names this capture.

    A classification of two OTHER captures says nothing about this one's rows,
    so reading it here would be borrowing evidence. It is refused by name
    rather than ignored (tessera#548).
    """
    if classification is None:
        return None
    named = [capture["capture_sha256"] for capture in classification["captures"]]
    if capture_sha256 not in named:
        raise ValueError("the boundary classification names captures " + ", ".join(named)
                         + ", not this one: " + str(capture_sha256))
    return {site["owner"]: site for site in classification["sites"]}, named


def _apply_two_capture(view, owners, sites, named, stack):
    """Classify one ``pending_548`` row from the two captures' matched bytes.

    Every shared site the row carries has to be present in both captures and
    has to agree, or the row keeps the class the measurement did not give it.
    A row whose sites disagree with each other is not averaged: the first
    moving site makes the row a candidate, because bytes that move with the
    assignment are what a candidate charge is.
    """
    both = ", ".join(named)
    matched = [sites[owner] for owner in owners if owner in sites]
    if not matched or any(len(site["present_in"]) != 2 for site in matched):
        view["reason"] = (view["reason"] + "; the two-capture classification of " + both
                          + " matches this site in one capture only, so it is neither agreed "
                            "nor moved")
        return
    moved = [site for site in matched if len(set(site["bytes"].values())) != 1]
    if not moved:
        # Agreement is only evidence where the substitution could have moved
        # the bytes. A per-Linear substitution touches a few units: a boundary
        # tensor of a unit whose family did not change agrees BY CONSTRUCTION,
        # and calling that fixed would charge a serving gate for bytes that
        # move the moment the menu does -- the failure this measurement exists
        # to prevent. A site naming no unit (a runner root, the BLAS
        # workspace) is tested by any change of assignment, so it still counts.
        untested = [site for site in matched
                    if site.get("unit") and not site.get("unit_family_changed")]
        if untested:
            view["reason"] = (view["reason"] + "; the bytes agree across " + both
                              + " but this site's unit kept its family ("
                              + json.dumps(untested[0].get("unit_family"), sort_keys=True)
                              + "), so the substitution never tested it")
            return
        view["class"], view["rule"] = "fixed", "two_capture:agreed"
        view["reason"] = ("the same bytes in both captures " + both
                          + ", on a site the substitution did move or that names no unit; "
                            "evidence for those two captures, not for every assignment")
        return
    view["class"], view["rule"] = "candidate", "two_capture:moved"
    unit = next((site["unit"] for site in moved if site["unit"]), None) or (stack[0] if stack else None)
    view["unit"] = unit
    view["reason"] = ("the bytes moved between the two captures " + both
                      + ": " + json.dumps(moved[0]["bytes"], sort_keys=True)
                      + ("" if unit else "; the owner names no unit, so this candidate still "
                                         "owes a unit"))


def derive_owner_views(rows, frames_by_index, *, checkpoint_index, roster, evidence,
                       boundary_classification=None, capture_sha256=None):
    """One view per replayed allocation, from the declared rules.

    ``rows`` are the replay's allocation rows (observed ownership already
    attached); ``frames_by_index`` maps ``allocate_index`` to the history
    frames; ``checkpoint_index`` maps checkpoint labels to trace indices;
    ``roster`` is the plan's canonical roster; ``evidence`` the inventories
    :func:`site_package` reads. ``boundary_classification`` is tessera#548's
    two-capture comparison, when one exists for this capture: without it every
    census-shared row allocated after the model stays ``pending_548``, which is
    the honest state of a single capture. Returns ``(views, summary)``.
    """
    two_capture = _two_capture_sites(boundary_classification, capture_sha256)
    sites, named = two_capture if two_capture is not None else (None, ())
    by_module, by_family = _roster_index(roster)
    before_load = checkpoint_index.get("before_model_load")
    ready = checkpoint_index.get("ready_for_workload")
    torch_root = str(PurePosixPath(evidence["plugin_package_path"]).parent / "torch")
    views = []
    for row in rows:
        frames = frames_by_index.get(row["allocate_index"])
        site = allocation_site(frames, torch_root)
        package, relative = site_package(site, evidence)
        view = {"allocation_id": row["allocation_id"], "class": None, "unit": None,
                "rule": None, "reason": None,
                "site": None if site is None else dict(site, package=package, relative=relative)}
        categories = list(row["observed_categories"])
        owners = list(row["observed_owners"])
        stack = list(row["scope_stack"])
        if len(categories) > 1:
            view["reason"] = "the census observed more than one ownership category for this storage"
        elif categories and categories[0] in ("fixed", "candidate", "kv"):
            view["class"], view["rule"] = categories[0], "census"
            if view["class"] == "candidate":
                view["unit"] = stack[0] if stack else _unit_from_owner_path(owners, by_module)
                if view["unit"] is None:
                    view["reason"] = ("candidate row whose owner path names no single roster unit")
        elif categories == ["shared"]:
            if before_load is not None and row["allocate_index"] < before_load:
                view["class"], view["rule"] = "fixed", "history_order"
            else:
                view["rule"] = "pending_548"
                kind = ("native boundary tensor" if any(o.startswith("native:") for o in owners)
                        else "persistent runtime root" if any(o.startswith("runner:") for o in owners)
                        else "BLAS workspace" if any(o.startswith("torch.cublas:") for o in owners)
                        else "library cache buffer")
                view["reason"] = (f"census shared ({kind}); assignment invariance is tessera#548's "
                                  "two-assignment measurement")
                if sites is not None:
                    _apply_two_capture(view, owners, sites, named, stack)
        elif not categories:
            if package == "plugin":
                view["class"], view["rule"] = "candidate", "site:plugin"
                if stack:
                    view["unit"] = stack[0]
                elif owners:
                    view["unit"] = _unit_from_owner_path(owners, by_module)
                if view["unit"] is None and row["free_completed_index"] is None and (
                        before_load is not None and ready is not None
                        and before_load <= row["allocate_index"] < ready):
                    family = route_family_of(frames, evidence)
                    units = by_family.get(family, []) if family is not None else []
                    if len(units) == 1:
                        view["unit"] = units[0]
                        view["reason"] = None
                    else:
                        view["reason"] = (
                            "load-time plugin allocation with no unit scope"
                            + (f"; route family {family} has {len(units)} roster units, so the unit "
                               "is not resolvable without a load-time unit scope"
                               if family is not None else
                               "; frames pass through no single route module"))
                elif view["unit"] is None:
                    view["reason"] = "plugin allocation with no unit scope and no owner path"
            elif package in ("vllm", "image"):
                view["class"], view["rule"] = "fixed", "site:" + package
            elif package == "observer":
                view["class"], view["rule"] = "observer", "site:observer"
            elif site is None:
                view["reason"] = "the allocation site carries no Python frame outside Torch"
            else:
                view["reason"] = "the allocation site is a file no run inventory attests: " + site["file"]
        else:
            view["reason"] = "unsupported observed category: " + ",".join(categories)
        views.append(view)
    summary = {
        "by_rule": dict(sorted(Counter(view["rule"] or "none" for view in views).items())),
        "by_class": dict(sorted(Counter(view["class"] or "null" for view in views).items())),
        "null_views": sum(1 for view in views if view["class"] is None),
        "null_bytes": sum(row["bytes"] for row, view in zip(rows, views) if view["class"] is None),
        "candidate_without_unit": sum(1 for view in views
                                      if view["class"] == "candidate" and view["unit"] is None),
    }
    return views, summary


def owner_views_record(views, summary, evidence):
    return {"schema": OWNER_VIEWS_SCHEMA,
            "rules": RULES,
            "evidence": {key: evidence[key] for key in ("plugin_package_path", "vllm_root",
                                                        "observer_roots", "inventory_digests")},
            "summary": summary,
            "views": views,
            "scope": ("one view per replayed allocation; observed ownership is repeated, never "
                      "rewritten; a null class names the rule that abstained and why")}


# ---------------------------------------------------------- witnesses ---

def _parse_boundary_owner(owner):
    """``native:<unit_id>:<invocation>:<kind>`` -> (unit_id, invocation, kind)."""
    if not owner.startswith("native:"):
        return None
    body = owner[len("native:"):]
    unit_id, invocation, kind = body.rsplit(":", 2)
    return unit_id, invocation, kind


#: tessera#548's row rule for the v2 boundary ledger. Every native boundary
#: tensor the worker observes is one allocation row in the ``_ALLOCATION_FIELDS``
#: shape, keyed by the ``(unit_id, invocation, kind)`` its owner string names.
#: Two rows under one key is not a smaller defect than a missing row: the ledger
#: could no longer say which storage that boundary is, and a two-capture
#: comparison would be matching one key against two different byte figures.
def boundary_rows(rows):
    """``{(unit_id, invocation, kind): row}`` over the native boundary owners.

    Raises when one key names two allocations, which refuses the v2 ledger
    rather than publishing an ambiguous boundary row.
    """
    index = {}
    for row in rows:
        for owner in row["observed_owners"]:
            parsed = _parse_boundary_owner(owner)
            if parsed is None:
                continue
            if parsed in index and index[parsed]["allocation_id"] != row["allocation_id"]:
                raise ValueError(
                    "two allocations carry one native boundary (unit, invocation, kind): "
                    + owner)
            index[parsed] = row
    return index


def _layer_of(module):
    parts = module.split(".")
    if "layers" in parts:
        index = parts.index("layers")
        if index + 1 < len(parts) and parts[index + 1].isdigit():
            return int(parts[index + 1])
    return None


def boundary_geometry_witness(rows, roster, steps):
    """tessera#548's within-capture witness for the native boundary tensors.

    For every boundary kind (``input.x`` / ``output``), module role (the
    module path's last component) and engine step, the bytes each family's
    units carried. Under one mixed artifact the same role appears under
    several families in layer 0 and under one family elsewhere; where the
    bytes agree across every family present, the geometry is a function of
    the role and the step alone. This is evidence for #548's question, from
    one capture; it is carried, not used to classify.
    """
    family_of = {row["unit_id"]: row.get("family") for row in roster}
    module_of = {row["unit_id"]: row["module"] for row in roster}
    table = {}
    for row in rows:
        for owner in row["observed_owners"]:
            parsed = _parse_boundary_owner(owner)
            if parsed is None:
                continue
            unit_id, _invocation, kind = parsed
            step = _step_of(row["allocate_index"], steps)
            role = module_of.get(unit_id, unit_id).split(".")[-1]
            key = (kind, role, step)
            table.setdefault(key, {}).setdefault(family_of.get(unit_id, "?"), Counter())[row["bytes"]] += 1
    cells = []
    for (kind, role, step), families in sorted(table.items(), key=lambda item: (item[0][0], item[0][1], str(item[0][2]))):
        sizes = {family: sorted(counter) for family, counter in families.items()}
        agree = len({tuple(size) for size in sizes.values()}) == 1 and all(len(size) == 1 for size in sizes.values())
        cells.append({"kind": kind, "role": role, "step": step,
                      "families": {family: {"bytes": sorted(counter), "rows": sum(counter.values())}
                                   for family, counter in sorted(families.items())},
                      "agree_across_families": agree})
    return {"schema": GEOMETRY_WITNESS_SCHEMA, "cells": cells,
            "cells_with_several_families": sum(1 for cell in cells if len(cell["families"]) > 1),
            "cells_agreeing_across_families": sum(1 for cell in cells
                                                  if len(cell["families"]) > 1 and cell["agree_across_families"]),
            "scope": ("bytes of each native boundary tensor by kind, module role and engine step, "
                      "split by the family of the unit it bounds; a within-capture witness for "
                      "tessera#548, not an ownership rule")}


def _step_of(index, steps):
    for step_id, begin, end in steps:
        if begin <= index < end:
            return step_id
    return None


def transient_gap_witness(rows, views, intervals, roster, steps):
    """Per-layer signature of the stock engine's in-step transients.

    Every ``site:vllm`` row inside a declared step is attributed to the gap
    between consecutive unit intervals it falls in, and the gap to the layer
    of the unit that closed it. Each (step, layer) then has a signature: the
    multiset of ``(site, bytes)`` the engine allocated there. Layer 0 of the
    mixed artifact holds units of three families and layers 1-27 one family;
    equal signatures across layers say the stock engine's transients do not
    depend on the family of the neighbouring Linear, which is the invariance
    ``site:vllm`` claims. Unequal signatures are reported as such.
    """
    module_of = {row["unit_id"]: row["module"] for row in roster}
    ordered = sorted(intervals, key=lambda item: item[0])
    # Unit ends per step: a gap is attributed to the unit that closed before it
    # IN THE SAME STEP, so the rows before a step's first unit are its own
    # bucket rather than the previous step's last layer. The last unit of each
    # step bounds the after-last-unit bucket (final norm, head, sampling).
    ends_by_step, last_end_in_step = {}, {}
    for begin, end, _invocation, unit in ordered:
        step = _step_of(begin, steps)
        if step is not None:
            ends_by_step.setdefault(step, []).append((end, unit))
            last_end_in_step[step] = max(last_end_in_step.get(step, 0), end)
    signatures = {}
    for row, view in zip(rows, views):
        if view["rule"] != "site:vllm":
            continue
        step = _step_of(row["allocate_index"], steps)
        if step is None:
            continue
        previous = None
        for end, unit in ends_by_step.get(step, ()):
            if end <= row["allocate_index"]:
                previous = unit
            else:
                break
        layer = _layer_of(module_of.get(previous, "")) if previous is not None else None
        site = view["site"]
        if layer is None:
            bucket = "pre-first-unit"
        elif row["allocate_index"] >= last_end_in_step.get(step, float("inf")):
            bucket = "after-last-unit"
        else:
            bucket = layer
        key = (step, bucket)
        signatures.setdefault(key, Counter())[(site["relative"], site["name"], row["bytes"])] += 1
    result = {}
    for step in sorted({key[0] for key in signatures}):
        layers = {key[1]: counter for key, counter in signatures.items()
                  if key[0] == step and key[1] not in ("pre-first-unit", "after-last-unit")}
        distinct = {}
        for layer, counter in layers.items():
            distinct.setdefault(tuple(sorted(counter.items())), []).append(layer)
        groups = sorted(distinct.values(), key=lambda group: (-len(group), group))
        outside = {name: {"rows": sum(signatures[(step, name)].values()),
                          "bytes": sum(size * count for (_f, _n, size), count in signatures[(step, name)].items())}
                   for name in ("pre-first-unit", "after-last-unit") if (step, name) in signatures}
        result[step] = {
            "layers": len(layers),
            "distinct_signatures": len(distinct),
            "signature_groups": [{"layers": sorted(group), "rows": sum(layers[group[0]].values()),
                                  "bytes": sum(size * count for (_f, _n, size), count in layers[group[0]].items())}
                                 for group in groups],
            "outside_layers": outside,
            "all_layers_agree": len(distinct) == 1,
        }
    return {"schema": TRANSIENT_WITNESS_SCHEMA, "steps": result,
            "scope": ("stock-engine in-step transients grouped by the layer of the unit interval "
                      "that precedes them; one signature per (step, layer); rows before the first "
                      "unit and after the last unit of a step are bucketed apart; agreement across "
                      "layers whose Linears differ in family is the witness site:vllm rests on")}


# ---------------------------------------------------------- externals ---

#: The four places a device static's source library may live, each attested by
#: the run: the plugin's JIT extension directory (the launcher's
#: ``TESSERA_EXT_DIR`` mount, holding only ``tessera_nvfp4_<sha>.so``), the
#: observer's own collector libraries (the plan names them by path and digest),
#: the pinned image's site-packages (the root the loaded plugin package sits
#: under, bound by image digest), and the stock runtime's JIT caches inside the
#: launcher's cache mount (``TRITON_CACHE_DIR`` / ``TORCH_EXTENSIONS_DIR``).
EXTERNAL_STATIC_CLASSES = ("plugin_jit_static", "observer_static", "image_static",
                           "image_jit_static")


def external_record_views(records, *, markers, unit_windows, evidence):
    """Classify every CUPTI memory record the Torch join left unmatched.

    Mirrors ``experiments/native_operator_resources.py`` so the two lanes price
    the same objects: DEVICE_STATIC (kind 6) records are ``startup_static`` by
    their CUPTI ``source`` library; DEVICE (kind 3) records outside the Torch
    allocator are external device allocations whose ``source`` names the
    allocating library, swept for the peak of new live bytes inside unit
    windows (``external_native_peak_bytes``) and for what stays resident.
    A record with no source, or from a library no run evidence places, is
    unresolved and named.
    """
    plugin_jit = evidence.get("plugin_jit_prefix")
    observer_libraries = set(evidence.get("observer_libraries", ()))
    site_packages = str(PurePosixPath(evidence["plugin_package_path"]).parent).rstrip("/") + "/"
    jit_caches = tuple(str(prefix).rstrip("/") + "/" for prefix in evidence.get("jit_cache_prefixes", ()))
    views, live3, live6 = [], {}, {}
    windows = sorted(unit_windows)
    in_window_live, external_peak = {}, 0
    for record in sorted(records, key=lambda row: row["timestamp_ns"]):
        source = record.get("source")
        key = (record["device_id"], record["context_id"], record["address"])
        phase = _phase_of(record["timestamp_ns"], markers)
        window = _window_of(record["timestamp_ns"], windows)
        view = {"memory_kind": record["memory_kind"], "operation": record["operation"],
                "bytes": record["bytes"], "address": record["address"],
                "context_id": record["context_id"], "source": source,
                "phase": phase, "unit_window": window, "class": None, "reason": None}
        if record["memory_kind"] == 6:
            if not isinstance(source, str) or not source:
                view["reason"] = "device static with no source library"
            elif plugin_jit and source.startswith(plugin_jit):
                view["class"] = "plugin_jit_static"
            elif source in observer_libraries:
                view["class"] = "observer_static"
            elif source.startswith(site_packages):
                view["class"] = "image_static"
            elif jit_caches and source.startswith(jit_caches):
                view["class"] = "image_jit_static"
            else:
                view["reason"] = ("device static from a library outside the image's site-packages, "
                                  "its JIT caches, the plugin JIT and the observer")
            if record["operation"] == "allocate":
                live6[key] = (record["bytes"], source, view["class"])
            else:
                live6.pop(key, None)
        elif record["memory_kind"] == 3:
            if not isinstance(source, str) or not source:
                view["reason"] = "device allocation outside the Torch allocator with no source library"
            elif window is not None:
                view["class"] = "unit_window_external"
            else:
                view["class"] = "library_external"
            if record["operation"] == "allocate":
                live3[key] = (record["bytes"], source, window)
                if window is not None:
                    in_window_live[key] = record["bytes"]
                    external_peak = max(external_peak, sum(in_window_live.values()))
            else:
                live3.pop(key, None)
                in_window_live.pop(key, None)
        else:
            view["reason"] = f"unsupported memory kind {record['memory_kind']}"
        views.append(view)
    startup_sources = {}
    for size, source, _class in live6.values():
        entry = startup_sources.setdefault(source, {"source": source, "live_bytes": 0, "live_allocation_count": 0})
        entry["live_bytes"] += size
        entry["live_allocation_count"] += 1
    resident_external = [{"bytes": size, "source": source, "unit_window": window}
                         for size, source, window in live3.values()]
    unresolved = [view for view in views if view["class"] is None]
    retained_in_window = [row for row in resident_external if row["unit_window"] is not None]
    return {
        "schema": EXTERNAL_RECORDS_SCHEMA,
        "records": views,
        "record_count": len(views),
        "startup_static": {
            "scope": "observed_live_device_static_at_capture_end",
            "live_bytes": sum(size for size, _s, _c in live6.values()),
            "live_allocation_count": len(live6),
            # An unresolved static keeps source None and class None; sort with
            # a None-safe key so the record that most needs naming cannot
            # take the whole derivation down with a TypeError.
            "sources": [startup_sources[source] for source in sorted(startup_sources, key=_none_last)],
            "by_class": dict(sorted(Counter(cls for _s, _src, cls in live6.values()).items(),
                                    key=lambda item: _none_last(item[0]))),
        },
        "external_native_peak_bytes": external_peak,
        "external_native_peak_scope": ("peak of new live non-Torch device allocations inside unit "
                                       "windows, swept over CUPTI timestamps; the native lane's "
                                       "external_native_peak_bytes"),
        "external_resident": {
            "scope": "non-Torch device allocations live at capture end, by source library",
            "live_bytes": sum(row["bytes"] for row in resident_external),
            "records": sorted(resident_external, key=lambda row: (_none_last(row["source"]), row["bytes"])),
            "retained_from_unit_windows": retained_in_window,
        },
        "unresolved": unresolved,
        "scope": ("every CUPTI memory record the Torch-segment join left unmatched, classified by "
                  "its own source library and by where in the capture it fell; a record with no "
                  "class is named here and keeps external_closure open"),
    }


def _none_last(value):
    """A sort key that orders ``None`` after every string instead of raising."""
    return (value is None, value or "")


def _phase_of(timestamp, markers):
    """The last checkpoint marker at or before ``timestamp``."""
    label = None
    for stamp, name in markers:
        if stamp <= timestamp:
            label = name
        else:
            break
    return label


def _window_of(timestamp, windows):
    for begin, end, unit in windows:
        if begin <= timestamp <= end:
            return unit
    return None


# ------------------------------------------------------ dense startup ---

def dense_startup_check(rows, views, dense, *, ready_index):
    """The dense artifact's ``worker_startup`` closure, from two independent numbers.

    Per unit, the ledger's candidate-owned rows that are still resident must
    sum to the manifest's own ``resident_bytes_resident_mode`` for that unit;
    and the allocator's ``memory_allocated()`` sampled at arm must bound every
    row live at ``ready_for_workload``. Any per-unit disagreement refuses the
    domain with the two numbers side by side.
    """
    if not isinstance(dense, dict) or dense.get("schema") != "tessera.full_engine_dense_startup_observation.v1":
        return None
    per_unit = Counter()
    resident_rows = {}
    for row, view in zip(rows, views):
        if view["class"] == "candidate" and row["free_completed_index"] is None and view["unit"] is not None:
            per_unit[view["unit"]] += row["bytes"]
            resident_rows.setdefault(view["unit"], []).append(_resident_row(row, view))
    units = {}
    for unit_id, entry in dense["units"].items():
        ledger_bytes = per_unit.get(unit_id, 0)
        manifest_bytes = entry["manifest_resident_bytes_resident_mode"]
        agree = ledger_bytes == manifest_bytes
        units[unit_id] = {"family": entry["family"], "ledger_candidate_resident_bytes": ledger_bytes,
                          "manifest_resident_bytes_resident_mode": manifest_bytes,
                          "difference_bytes": ledger_bytes - manifest_bytes,
                          "agree": agree,
                          # On disagreement every resident row the ledger charges
                          # to the unit, by its census owner or allocation site,
                          # so the reader sees which bytes the manifest's figure
                          # does not price (or which priced bytes are absent).
                          # Enumeration is disclosure; it never closes the cell.
                          "resident_rows": None if agree else sorted(
                              resident_rows.get(unit_id, []), key=lambda r: (-r["bytes"], r["owner"] or ""))}
    unpriced = sum(cell["difference_bytes"] for cell in units.values() if cell["difference_bytes"] > 0)
    live_at_ready = sum(row["bytes"] for row in rows
                        if ready_index is not None and row["allocate_index"] < ready_index
                        and (row["free_completed_index"] is None or row["free_completed_index"] >= ready_index))
    extra_units = sorted(set(per_unit) - set(dense["units"]))
    disagreeing = sorted(unit_id for unit_id, cell in units.items() if not cell["agree"])
    bounded = dense["memory_allocated_bytes"] >= live_at_ready
    return {
        "schema": DENSE_STARTUP_CHECK_SCHEMA,
        "units": units,
        "units_checked": len(units),
        "units_disagreeing": disagreeing,
        "manifest_unpriced_resident_bytes": unpriced,
        "candidate_units_outside_manifest": extra_units,
        "memory_allocated_bytes": dense["memory_allocated_bytes"],
        "ledger_live_bytes_at_ready_for_workload": live_at_ready,
        "allocator_sample_bounds_ledger": bounded,
        "closed": not disagreeing and not extra_units and bounded,
        "scope": ("per unit: ledger candidate-owned resident rows against the artifact manifest's own "
                  "resident_bytes_resident_mode; allocator sample at arm against the ledger's live "
                  "bytes at ready_for_workload; closed only on exact per-unit agreement, and a "
                  "disagreeing unit lists its resident rows by census owner or allocation site so the "
                  "difference is named, never absorbed"),
    }


def _resident_row(row, view):
    owners = row.get("observed_owners") or []
    site = view.get("site") or {}
    return {"bytes": row["bytes"], "allocation_id": row["allocation_id"],
            "owner": owners[0] if len(owners) == 1 else (owners or None),
            "site": (f"{site.get('relative')}:{site.get('name')}" if site.get("relative") else None),
            "rule": view.get("rule")}


# -------------------------------------------------------- observation ---

OWNERSHIP_OBSERVATION_SCHEMA = "tessera.full_engine_ownership_observation.v1"


def ownership_observation(*, views, summary, evidence, external_records, geometry_witness,
                          transient_witness, dense_startup_check, boundary_classification=None):
    """The one ``owner_views`` observation the report carries.

    Everything the derivation produced travels under the observation name the
    report schema already owes (``owner_views``), so the observation field set
    the consumer reads does not grow: the per-allocation views and their
    summary, the rules and the evidence they were applied to, the external
    record classification, the two within-capture witnesses and the dense
    startup check. Each member keeps its own schema string.
    """
    return {
        "schema": OWNERSHIP_OBSERVATION_SCHEMA,
        "views": owner_views_record(views, summary, evidence),
        "external_records": external_records,
        "boundary_geometry_witness": geometry_witness,
        "transient_gap_witness": transient_witness,
        "dense_startup_check": dense_startup_check,
        # tessera#548's comparison travels INSIDE this observation rather than
        # beside it: the report's `observations` key set is what the consumer
        # pins, and the numbers a consumer must recompute the two_capture rule
        # from have to arrive with the views that rule wrote. Null when this
        # capture has no pair, which is the honest state of one capture.
        "boundary_classification": boundary_classification,
        "scope": ("replay-time ownership derivation over one capture: derived views beside the "
                  "census's observed ownership, external CUDA records by source library, and "
                  "the witnesses the rules rest on; nothing here rewrites an observed field"),
    }


def views_by_allocation(ledger):
    """``{allocation_id: view}`` from a ledger's ownership observation, or ``{}``."""
    observation = ledger.get("owner_views")
    if not isinstance(observation, dict) or observation.get("schema") != OWNERSHIP_OBSERVATION_SCHEMA:
        return {}
    return {view["allocation_id"]: view for view in observation["views"]["views"]}
