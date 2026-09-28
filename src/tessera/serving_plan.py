"""The serving plan one producer writes and the exporter holds to.

``tessera.serving_plan.v1`` is the published shape of the ``--plan-json``
object (tessera#687): the plan is Tessera-defined, so the producer that
allocates a checkpoint writes it directly instead of a translation layer in
this repository parsing the producer's own records (#599).  The schema:

* a JSON OBJECT mapping tensor or stack names to entries;
* per TENSOR: ``"PASSTHROUGH"`` or ``"BF16"`` (the negative spellings -- copy
  the source tensor, quantise nothing), or an object ``{"grid", "q256"}``;
* per ``<moe>.experts`` STACK: an object ``{"grid", "q256", "source_layout"}``
  giving every expert of the stack one rung and naming the source tensor
  layout the packed conventions decode.

``grid`` is one of the four grid names (``E2M1``, ``E2M1x2``, ``E4M3``,
``BF16``); ``q256`` is an integer rung; ``source_layout`` is OPTIONAL, one of
the three closed conventions when present, and meaningful only on a stack
entry.  One field name -- ``producer_annotations`` -- is the PRODUCER
ANNOTATION: a JSON object the exporter copies through to the published plan
verbatim and never reads (#691 item 6).  It carries whatever sidecar the
producer's own accounting already knows, under the producer's own names, so
this schema names no producer and no producer field: nothing here imports or
reads PrismaQuant (#599).

A plan may declare its schema under the reserved top-level ``"schema"`` key
(#691 item 5).  The key is not an entry -- a tensor is never named
``"schema"`` -- and its value must be ``SERVING_PLAN_SCHEMA``; a plan
declaring anything else is refused.  The key is optional, so plans written
before it existed still validate; the exporter records what the plan
declared (or nothing) in the manifest beside the published plan.

This module also owns the fused-group rule ``module_scheme_key`` -- moved out
of the exporter (#687) so a producer can check the groups it allocates BEFORE
writing a plan, without importing an entry point -- and ``family_for``, the
grid-to-family statement that rule and the exporter share.
"""

from tessera.serving.scheme import (
    STRUCTURE_DENSE, TESSERA_BF16, TESSERA_FP8, TESSERA_NVFP4)

#: The schema identity a plan object declares, or is validated against.
SERVING_PLAN_SCHEMA = "tessera.serving_plan.v1"

#: The reserved top-level key a plan declares its schema under (#691 item
#: 5).  Not an entry: no tensor or stack is ever named ``"schema"``.
SCHEMA_KEY = "schema"

#: The neutral producer-annotation field an entry may carry (#691 item 6).
#: A JSON object, copied through verbatim and never read.
PRODUCER_ANNOTATIONS = "producer_annotations"

#: The plan key suffix that names a routed expert STACK rather than a tensor.
STACK_SUFFIX = ".experts"

_PASSTHROUGH = ("PASSTHROUGH", "BF16")
_OWNED_FIELDS = ("grid", "q256", "source_layout")
_SOURCE_LAYOUTS = ("unpacked_per_expert", "out_first_chunked", "in_first_interleaved")


def validate_serving_plan(entries) -> None:
    """Refuse a plan outside ``tessera.serving_plan.v1``, naming the entry.

    This is the argument-time half of the plan contract; the publication gate
    (``tessera.serving_parts.validate_explicit_plan``) holds the plan to the
    same entry shape again when it checks the export's obligations against
    it.  A refusal here names the offending entry (and field), because a
    malformed entry is a typo in the run's instructions and finding it after
    the first encode costs the encode.
    """
    if not isinstance(entries, dict):
        raise ValueError(
            "the serving plan must be a JSON object mapping tensor or stack "
            "names to entries")
    declared = entries.get(SCHEMA_KEY)
    if declared is not None and declared != SERVING_PLAN_SCHEMA:
        raise ValueError(
            f"has an invalid {SCHEMA_KEY!r} {declared!r}: this validator reads "
            f"{SERVING_PLAN_SCHEMA}")
    for name, spec in entries.items():
        if name == SCHEMA_KEY:
            continue
        if spec in _PASSTHROUGH:
            continue
        if not isinstance(spec, dict):
            raise ValueError(
                f"has an invalid entry {name!r}: an entry is "
                f'"PASSTHROUGH"/"BF16" or an object carrying "grid" and "q256"')
        unknown = sorted(set(spec) - set(_OWNED_FIELDS) - {PRODUCER_ANNOTATIONS})
        if unknown:
            raise ValueError(
                f"has an invalid entry {name!r}: field(s) {unknown} are not in "
                f"the {SERVING_PLAN_SCHEMA} schema")
        annotations = spec.get(PRODUCER_ANNOTATIONS)
        if annotations is not None and not isinstance(annotations, dict):
            raise ValueError(
                f"has an invalid entry {name!r}: {PRODUCER_ANNOTATIONS!r} "
                f"must be an object, got {annotations!r}")
        missing = [field for field in ("grid", "q256") if field not in spec]
        if missing:
            raise ValueError(
                f"has an invalid entry {name!r}: missing {missing}")
        if type(spec["q256"]) is not int:
            raise ValueError(
                f"has an invalid entry {name!r}: q256 must be an integer, "
                f"got {spec['q256']!r}")
        if not isinstance(spec["grid"], str):
            raise ValueError(
                f"has an invalid entry {name!r}: grid must be one of the four "
                f"grid names, got {spec['grid']!r}")
        from tessera.control import grid_for_name, GrammarError
        try:
            grid_for_name(spec["grid"])
        except GrammarError as exc:
            raise ValueError(
                f"has an invalid entry {name!r}: {exc}") from exc
        layout = spec.get("source_layout")
        if layout is not None:
            if not isinstance(name, str) or not name.endswith(STACK_SUFFIX):
                raise ValueError(
                    f"has an invalid entry {name!r}: source_layout is a stack "
                    f"field, carried only on a {STACK_SUFFIX} entry")
            if layout not in _SOURCE_LAYOUTS:
                raise ValueError(
                    f"has an invalid entry {name!r}: source_layout "
                    f"{layout!r} is not one of {_SOURCE_LAYOUTS}")


def family_for(grid) -> str:
    """The stock tile family a grid decodes to."""
    if grid.name == "BF16":
        return TESSERA_BF16
    return TESSERA_FP8 if grid.name == "E4M3" else TESSERA_NVFP4


def module_scheme_key(grid, q256: int, structure: str = STRUCTURE_DENSE) -> tuple:
    """The facts every role of one vLLM-fused module must agree on.

    NOT ``(grid, q256)`` (#37).  vLLM builds one quant method per module, so
    what has to be shared is what that method is built from -- the family, and
    with it the grid, body and scale plane its route decodes to one tile.  The
    RATE is not one of them: every decoder in ``tessera.serving`` reads a role
    from that role's OWN manifest, and the runtime says so in a value
    (``runtime_contract.json``'s ``fused_module.fields``, checked against
    ``scheme.FUSED_MODULE_FIELDS``).  The old key compared the rate too, so a
    group whose members took different rungs -- which is exactly what a
    producer's group knapsack allocates, folded ON by default -- was passed
    through at source precision by the exporter, and the allocator's chosen
    point had no Tessera export at all.

    The body and the plane are DERIVED from the rung here rather than assumed
    constant, because the served recipe picks them per rung: ``served_recipe``
    promotes every NVFP4 rung to the span-2 TCQ body the routed path decodes,
    for ``STRUCTURE_ROUTED_MOE`` (the research ``wire_recipe`` default below
    the coset cap is still the window body, which is why this key reads the
    served spelling rather than the research one).  Two members resolving to
    different served bodies would decode on two different decoders, and this
    key separates them, so the relaxation cannot let a mixed-body group
    through the back door.  (A rung no served spelling covers -- a dense
    sub-cap rung keeps WINDOW and is refused -- is refused by ``check_recipe``
    before this anyway; the key does not rely on that.)
    """
    # Lazy: ``tessera.export`` imports torch, and this module stays on the
    # torch-free side of that boundary (CI ``pure``) -- the key needs the
    # served recipe only here, never at import time.
    from tessera.export import served_recipe
    recipe = served_recipe(grid, q256, structure)
    return (family_for(grid), grid.name, recipe.body.name, recipe.scale_plane.name)
