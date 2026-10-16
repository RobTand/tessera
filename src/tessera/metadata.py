"""Pure producer metadata, ownership rules, smoke derivations and lane decisions.

The public API reads JSON-shaped metadata without loading serving or numerical
runtimes. Serving imports these same owners. The API marker grants no admission.
"""
from __future__ import annotations

import re
from typing import Any, Mapping

from .structure import STRUCTURES, STRUCTURE_ROUTED_MOE

METADATA_API_VERSION = "tessera.producer-metadata.v1"

__all__ = [
    "fused_module", "role_name", "glm_fused_groups",
    "METADATA_API_VERSION", "parse_smoke_record", "smoke_status_is_derived",
    "derive_smoke_status", "derive_smoke_attribution", "decide_lane_requirements",
    "route_family_for_grid", "direct_consumer_activation_contract",
    "direct_consumer_resident_bytes",
]

EVIDENCE_SMOKE_STATUSES = ("recorded", "repetitive", "not_recorded")


#: A smoke's CONTROL (schema v7, #195): the reference the same prompt was run
#: against, so a consumer can tell "this route degenerates" from "this prompt
#: degenerates on this model".  ``status`` says what came back and is accurate;
#: it is not the field that decides admission, because a degenerate completion
#: the unquantised source produces character for character is a property of the
#: model and the prompt.  That distinction was measured
#: (``docs/measurements/moe-evidence-debt-2026-09-04.md`` section 7) and lived
#: in prose while a producer-side gate refused the whole routed-MoE lane on
#: ``status``: the machine-readable field was true about the observation and
#: misleading about its meaning, which is principle 14 one level up.
#:
#: ``control`` is ``{reference, outcome, receipt}`` or ``null``, and
#: ``attribution`` is DERIVED from it and checked, the way ``grade`` is derived
#: from ``kl``.  What ``outcome`` establishes, exactly: ``identical_completion``
#: means the reference returned the SAME completion for the same prompt under
#: the smoke's own runtime and execution mode; ``different_completion`` means
#: the completions differ -- NOT that the reference was healthy, which no string
#: comparison can decide.  There is no ``scope`` field: a smoke record carries no
#: scope of its own, a control cannot be better scoped than the thing it
#: controls, and that the arms match is what the receipt attests, exactly as for
#: every other receipt-backed field here.  One control, not a list: one is what
#: has been run, and a second reference is a bump rather than a surprise.
EVIDENCE_CONTROL_REFERENCES = ("bf16_source",)


#: A smoke's RECORD (schema v9, #327): the machine-readable scoring that
#: ``status`` and ``attribution`` are DERIVED from, the way ``grade`` is derived
#: from ``kl``.  Until v9 the step from a set of completions to the cell's one
#: published word was undeclared, underived and untested: the per-completion
#: rule was stated and pinned, but the AGGREGATION -- which set of completions
#: makes a cell read ``recorded`` -- was stated once, in a dated measurement
#: file, in the existential form "at least one prompt reads ``recorded`` on both
#: arms", with nothing in the repository saying why "some" rather than "all",
#: nothing deriving it, and nothing that would notice if the smoke were re-run
#: with a different result.  A rule that decides a field a serving gate reads
#: does not live in append-only history (AGENTS.md rules 2, 4 and 10).
#:
#: THE AGGREGATION, in one place, and it is :func:`derive_smoke_status`:
#:
#: * ``repetitive`` when some row cycles on this cell's arm and the reference
#:   does not.  A degeneration the reference does not share is the route's, and
#:   that is the whole content of the word.
#: * else ``recorded`` when some row reads ``recorded`` on this arm AND on the
#:   reference (or on this arm alone when no reference was run): a positive
#:   record, which is what principle 5's "generates correctly" leg asks for.
#: * else ``repetitive`` when any row cycles at all -- every cycle is shared
#:   with the reference, but nothing positive was recorded, so there is no
#:   record to publish.
#: * else ``not_recorded``: nothing came back.
#:
#: The threshold is not "some prompt" and not "all prompts"; it is derived from
#: what the word is a claim ABOUT.  ``status`` is a claim about this route, so
#: an observation the unquantised reference reproduces is not this route's --
#: which is the same principle schema v7's ``control`` was added for, applied
#: per (prompt, form) instead of once per cell.  What that leaves behind is
#: carried by ``attribution``, still derived and still checked, now off the
#: record: a residual cycle every reference row shares reads
#: ``shared_with_reference``, so a consumer at the pin can tell a cell whose
#: smoke cycled on an interface from one whose smoke never cycled at all.
#:
#: A row is one (prompt, form) pair: ``{prompt, form, interface, status,
#: reference_status}``, each verdict from :data:`EVIDENCE_SMOKE_STATUSES` under
#: the instrument's per-completion rule, which the record quotes verbatim.
#: ``interface`` is the model interface the prompt exercised, and it is the axis
#: the 2026-09-05 routed-MoE measurement actually turns on: through the
#: checkpoint's own chat template both arms answer, under raw continuation both
#: arms -- student and BF16 source alike -- end in a cycle.  No number lives in
#: a cell here either: the token count, the period and the periodic suffix are
#: the receipt's, and the record carries the verdicts read off them.
#:
#: AN ASSERTED STATUS.  ``record`` is nullable, and a cell whose record is
#: ``null`` publishes an **asserted status** -- a word a person read off a
#: receipt, which no validator can check and which :func:`smoke_status_is_derived`
#: reports ``False`` for.  At schema v9, two dense BF16 cells were in that state: their
#: receipt (``tessera-bf16-route-served-2026-09-02.md`` :83) predates this
#: instrument and records one greedy continuation from four census arms of the
#: same route -- one prompt, no per-completion scoring and no reference arm --
#: so there is no per-(prompt, form) table to transcribe and nothing here
#: invents one.  What would give them a derived status is a measurement, not an
#: edit: ``experiments/moe_greedy_smoke.py`` run against a serve of that
#: artifact with its own tokenizer, and a BF16 reference arm beside it.  At schema v9, the six
#: ``not_recorded`` cells had run no smoke and had nothing to record.
EVIDENCE_SMOKE_INTERFACES = ("raw_completion", "chat_template")


EVIDENCE_SMOKE_FORMS = ("campaign", "pure_greedy")


EVIDENCE_SMOKE_RECORD_KEYS = frozenset({"instrument", "rule", "reference", "rows"})


EVIDENCE_SMOKE_ROW_KEYS = frozenset(
    {"prompt", "form", "interface", "status", "reference_status"})


def _require_keys(payload: Mapping[str, Any], where: str, required: set[str],
                  optional: set[str] = frozenset()) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{where} must be a JSON object")
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"{where} is missing {missing}")
    unknown = sorted(set(payload) - required - set(optional))
    if unknown:
        raise ValueError(f"{where} carries unknown field(s) {unknown}")


def smoke_status_is_derived(smoke: Mapping[str, object]) -> bool:
    """Whether this smoke's ``status`` is derived or ASSERTED (schema v9, #327).

    ``True`` when the smoke carries a ``record``, in which case
    :func:`derive_smoke_status` computes the word and the validator refuses any
    other.  ``False`` when ``record`` is ``null``: the word is one a person read
    off a receipt, nothing here can check it, and the header comment on
    :data:`EVIDENCE_SMOKE_RECORD_KEYS` says which cells are in that state and
    what would have to be measured to leave it.
    """
    return smoke.get("record") is not None


def derive_smoke_status(smoke: Mapping[str, object]) -> str:
    """The status word a smoke's ``record`` DERIVES (schema v9, #327).

    ``smoke`` is a validated smoke mapping with a string ``status``. A null
    record returns that asserted value unchanged, as the historical owner did;
    this derivation does not validate a caller's assertion.

    The aggregation from a set of scored completions to the cell's one
    published word, in its only home.  Each row is one (prompt, form) pair
    already scored under the instrument's per-completion rule, on this cell's
    arm and -- when a reference arm was run -- on the reference:

    * some row cycles on this arm and the reference does not: ``repetitive``.
      A degeneration the reference does not share is this route's, and that is
      what the word claims.
    * else some row reads ``recorded`` on this arm and on the reference (on
      this arm alone when there is no reference): ``recorded``.  That is the
      positive record principle 5's "generates correctly" leg asks for.
    * else some row cycles: ``repetitive``.  Every cycle is shared, but nothing
      positive was recorded, so there is no record to publish.
    * else ``not_recorded``: nothing came back.

    A row whose arm reads ``recorded`` while the reference cycles is neither: the
    route is not degenerating and the pair is not a two-arm positive record.
    An empty completion scores ``not_recorded`` at the instrument
    (``experiments/moe_greedy_smoke.py``), so it can never be or create one.

    With no record there is no derivation: the smoke's own asserted ``status``
    is returned unchanged, and :func:`smoke_status_is_derived` is how a caller
    tells that case apart rather than reading a word back from itself.
    """
    record = smoke.get("record")
    if record is None:
        return smoke.get("status")
    reference = record.get("reference")
    positive = unshared = cycled = False
    for row in record.get("rows") or []:
        status, against = row.get("status"), row.get("reference_status")
        if status == "recorded" and (reference is None or against == "recorded"):
            positive = True
        elif status == "repetitive":
            cycled = True
            if reference is None or against != "repetitive":
                unshared = True
    if unshared:
        return "repetitive"
    if positive:
        return "recorded"
    return "repetitive" if cycled else "not_recorded"


def derive_smoke_attribution(smoke: Mapping[str, object]) -> str:
    """What a smoke's ``record`` -- or, without one, its ``control`` -- DERIVES
    about where the symptom lives.

    No reference arm, ``unattributed`` -- nobody ran one, so the status is an
    observation and not an attribution.  A reference that shares every cycle,
    ``shared_with_reference``: the model and the prompt produce it, not this
    route.  One that does not share some cycle, ``not_shared_with_reference``:
    the reference does not explain it, which is weaker than "the route is at
    fault" and is deliberately not spelled that way.  A record with a reference
    arm and no cycle anywhere is ``unattributed`` too: there is no symptom to
    attribute.

    Schema v9 moves the source from ``control`` to ``record`` for a smoke that
    carries one, and the two may not both be present (:func:`cell_evidence`
    refuses that by name): one derivation, one home.  The v7 ``control`` branch
    below is unchanged and is what a smoke with no record still reads.
    """
    if smoke.get("record") is not None:
        record = smoke["record"]
        if record.get("reference") is None:
            return "unattributed"
        cycles = [row for row in (record.get("rows") or [])
                  if row.get("status") == "repetitive"]
        if not cycles:
            return "unattributed"
        if all(row.get("reference_status") == "repetitive" for row in cycles):
            return "shared_with_reference"
        return "not_shared_with_reference"
    control = smoke.get("control")
    if control is None:
        return "unattributed"
    if control.get("outcome") == "identical_completion":
        return "shared_with_reference"
    return "not_shared_with_reference"


def _require_repository_path(value: Any, where: str, field: str) -> str:
    """A path inside this repository, or raise.

    Not under ``docs/measurements/`` -- an instrument is code, not a receipt --
    but relative and without traversal, as ``test_contract_is_portable.py``
    requires. This checks path grammar; tree tests check file existence.
    """
    if (not isinstance(value, str) or not value or value.startswith("/")
            or "\\" in value or any(c.isspace() for c in value)
            or any(part in ("", ".", "..") for part in value.split("/"))):
        raise ValueError(
            f"{where}.{field} must be a repository path to the code that owns the rule "
            f"(relative, no traversal), got {value!r}")
    return value


def parse_smoke_record(record: object, *, control: object,
                       where: str) -> dict[str, object] | None:
    """The per-(prompt, form) scoring a smoke's word is DERIVED from (v9, #327).

    ``null`` when the smoke has no machine-readable record and its ``status`` is
    therefore ASSERTED (see :data:`EVIDENCE_SMOKE_RECORD_KEYS`).  Otherwise the
    closed ``{instrument, rule, reference, rows}``: ``instrument`` the
    repository path to the code that owns the per-completion rule, ``rule`` that
    code's own statement of it quoted verbatim so the rule is readable at the
    pin (the wheel does not ship ``experiments/``; a tree test pins the two
    equal), ``reference`` the arm the completions were compared against or
    ``null``, and ``rows`` a non-empty set of distinct (prompt, form) pairs.
    """
    payload = record
    smoke_at = where.removesuffix(".record")
    if payload is None:
        return None
    if control is not None:
        raise ValueError(
            f"{smoke_at} carries BOTH a control and a record; the attribution is derived from "
            "one of them, and two homes for one derivation is exactly how they drift. A record "
            "names its reference arm and every row's verdict on it, so it says everything the "
            "v7 control says and says it per (prompt, form): drop the control")
    _require_keys(payload, where, required=set(EVIDENCE_SMOKE_RECORD_KEYS))
    _require_repository_path(payload["instrument"], where, "instrument")
    rule = payload["rule"]
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError(
            f"{where}.rule must state the per-completion rule the rows were scored under, "
            f"verbatim from the instrument, got {rule!r}")
    reference = payload["reference"]
    if reference is not None and reference not in EVIDENCE_CONTROL_REFERENCES:
        raise ValueError(
            f"{where}.reference {reference!r} is not one of {list(EVIDENCE_CONTROL_REFERENCES)}; "
            "a reference this package cannot name is prose, and the whole point of the record "
            "is that a gate reads it")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows or any(not isinstance(r, Mapping) for r in rows):
        raise ValueError(
            f"{where}.rows must be a non-empty JSON array of {{prompt, form, interface, status, "
            f"reference_status}} objects; a record with no row scores nothing, got {rows!r}")
    seen: set[tuple[str, str]] = set()
    parsed_rows = []
    for i, row in enumerate(rows):
        spot = f"{where}.rows[{i}]"
        _require_keys(row, spot, required=set(EVIDENCE_SMOKE_ROW_KEYS))
        prompt = row["prompt"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{spot}.prompt must name the prompt the receipt names, "
                             f"got {prompt!r}")
        form, interface = row["form"], row["interface"]
        if form not in EVIDENCE_SMOKE_FORMS:
            raise ValueError(f"{spot}.form {form!r} is not one of {list(EVIDENCE_SMOKE_FORMS)}")
        if interface not in EVIDENCE_SMOKE_INTERFACES:
            raise ValueError(
                f"{spot}.interface {interface!r} is not one of {list(EVIDENCE_SMOKE_INTERFACES)}; "
                "the interface a prompt exercised is what separates a raw continuation from the "
                "checkpoint's own chat template, and a reader cannot infer it from the prompt id")
        status = row["status"]
        if status not in EVIDENCE_SMOKE_STATUSES:
            raise ValueError(
                f"{spot}.status {status!r} is not one of {list(EVIDENCE_SMOKE_STATUSES)}")
        against = row["reference_status"]
        if reference is None:
            if against is not None:
                raise ValueError(
                    f"{spot}.reference_status is {against!r} but the record names no reference "
                    "arm; a verdict on an arm nobody ran is a claim without a measurement")
        elif against not in EVIDENCE_SMOKE_STATUSES:
            raise ValueError(
                f"{spot}.reference_status {against!r} is not one of "
                f"{list(EVIDENCE_SMOKE_STATUSES)}; the record names reference {reference!r}, so "
                "every row states what that arm returned for the same (prompt, form)")
        key = (prompt, form)
        if key in seen:
            raise ValueError(
                f"{spot} repeats (prompt, form) {key}; the rows are a set of pairs, and a "
                "duplicate would weight one completion twice in the aggregation")
        seen.add(key)
        parsed_rows.append({"prompt": prompt, "form": form, "interface": interface,
                            "status": status, "reference_status": against})
    return {"instrument": payload["instrument"], "rule": rule,
            "reference": reference, "rows": parsed_rows}


#: The checkpoint's spelling of a body/plane beside the route's spelling of
#: the same fact.  ``attested_wire`` speaks ``wire.recipes`` -- the
#: vocabulary a checkpoint's config records per rung
#: (``export._BODY_NAMES`` / ``export._PLANE_NAMES``) -- while the authority
#: it is checked against speaks ``scheme.ROUTES``.  The map between the two
#: dialects is written once here, and
#: ``tests/test_serving_attested_wire.py`` compares both ends against their
#: sources, so a rename on either side fails there rather than drifting here.
_ATTESTED_WIRE_BODY = {"tcq": "TCQ", "window": "WINDOW"}


_ATTESTED_WIRE_PLANE = {"s6b": "S6B", "lut16": "LUT", "channel": "CHANNEL"}


#: ``manifest.RotationState`` names, in the checkpoint dialect a lane
#: predicate speaks (``lane.requires.rotation``, #264).
_ATTESTED_WIRE_ROTATION = {"none": "NONE", "r_in_only": "R_IN_ONLY"}


def route_wire_spelling(field: str, value: str) -> str:
    """``scheme.ROUTES``' spelling of a checkpoint-dialect wire fact name.

    The contract states a wire in the vocabulary a checkpoint's own config
    records (``window``, ``channel``, ``none``); the tables a loader gates on
    spell the same facts ``WINDOW``, ``CHANNEL`` and ``RotationState.NONE``.
    One map, the one ``attested_wire`` already uses for body and plane, so a
    caller comparing a recipe against a published predicate does not carry a
    second copy of it.
    """
    dialect = {"body": _ATTESTED_WIRE_BODY, "plane": _ATTESTED_WIRE_PLANE,
               "rotation": _ATTESTED_WIRE_ROTATION}.get(field)
    if dialect is None:
        raise ValueError(
            f"no wire dialect for field {field!r}; known: ['body', 'plane', 'rotation']")
    if value not in dialect:
        raise ValueError(f"{value!r} is not a {field} this package names ({sorted(dialect)})")
    return dialect[value]


#: How each PUBLISHED lane requirement is decided against ONE parsed unit's
#: facts: the requirement's own name -> (the fact it reads, how it is read).
#: A requirement this table has not learned is REFUSED rather than skipped,
#: which is the whole of #206 -- an offline checker that quietly evaluated the
#: subset it knew published ``READABLE`` for wire its own loader rejects, and
#: was therefore weaker than the plan-time gate it exists to double-check.
#:
#: This table and :func:`decide_lane_requirements` below are ONE HOME (#264):
#: the byte-time report (:func:`lane_wire_report`), the plan-time gate
#: (:func:`refuse_unreachable_lane`), the loader
#: (``kernel_window_gemv.prepare_from_parsed``) and the route-side gate
#: (``bf16_route.gemv_refusal_for_unit``) all decide a unit through the same
#: function over the same published requirements.  Before that, the contract
#: published four conditions while the loader refused nine, so a TP row shard
#: -- ``layout.slice_unit`` stamps ``initial_state`` -- read READABLE at
#: preflight and fell back module by module at load.
_LANE_WIRE_CHECKS = {
    "column_rates": ("rates", "every_in"),
    # Structure-scoped (tessera#694): the rates the lane's ROUTED-EXPERT launch
    # reaches on the target, decided over ``facts["rates"]`` only when
    # ``facts["structure"]`` is the routed-MoE structure; a caller that did
    # not state the structure is refused by name, and any other structure is
    # decided as not binding.  The fused window lane reads every rate of the
    # wire (``column_rates``) on its one-table dense launch, but its two-table
    # gate/up launch holds fewer word-stage words in sm_121's opt-in shared
    # memory, and the value library's dense launch reaches rates (9..14,
    # tessera#750 item 4) its routed launches are not built for, so a stack
    # above this set keeps the compact adapter.
    "column_rates_routed_moe": ("rates", "every_in_routed_moe"),
    "window_bits": ("window_bits", "one_of"),
    "body": ("body", "wire_spelling"),
    "plane": ("plane", "wire_spelling"),
    "release_overrides": ("release_overrides", "carries"),
    "diagonals": ("diagonals", "carries"),
    "start_state": ("start_state", "carries"),
    "rotation": ("rotation", "state_in"),
    "grid_arities": ("grid_arity", "one_of"),
}


#: What a ``carries``-mode refusal says, per requirement.  ``{n}`` is the
#: fact's count where the fact carries one (RELEASE overrides), else elided.
_CARRIES_REFUSALS = {
    "release_overrides": (
        "the unit carries {n}RELEASE override(s), and the lane decodes straight "
        "off the wire with no override read"),
    "diagonals": (
        "the unit carries segment-2a diagonals, and the lane applies none"),
    "start_state": (
        "the unit carries a shard start state (INITIAL_STATE, layout.slice_unit), "
        "and the lane supplies state_{{-1}} = 0 itself -- the L-bit pad that opens "
        "a wire column is not stored, so a shard packed against the pinned start "
        "would decode its first rows to plausible wrong weights; a lane that "
        "threads the state (tessera.kernel_window) or the materialised path "
        "serves this shard"),
}


def decide_lane_requirements(lane: str, requires: Mapping[str, object],
                             facts: Mapping[str, object]) -> list[str]:
    """Refusals, by name, for ONE unit's facts against a lane's requirements.

    THE HOME of the lane-readability decision (#264).  ``requires`` is the
    published predicate (``native_extensions[].lane.requires``, or the
    build's own copy ``ext.WINDOW_GEMV_LANE["requires"]`` -- the contract
    validator pins the two equal); ``facts`` is one unit's wire facts in
    :func:`wire_facts_of_parsed`'s vocabulary.  Every caller that answers
    "can this lane read this unit" calls this: the byte-time and plan-time
    gates, the loader and the bf16 route's gate, so the published predicate
    and the loader agree by construction rather than by three restatements.

    A requirement this function has not learned RAISES rather than being
    skipped (#206), and a fact the caller did not supply is a refusal, never
    a pass -- absent evidence is not evidence of absence.
    """
    unknown = sorted(set(requires) - set(_LANE_WIRE_CHECKS))
    if unknown:
        raise ValueError(
            f"lane {lane!r} publishes requirement(s) {unknown} that this reader cannot "
            f"decide (it decides {sorted(_LANE_WIRE_CHECKS)}). A checker that skipped a "
            "published condition would call a unit readable that the loader refuses, so "
            "it refuses to answer instead. Teach scheme._LANE_WIRE_CHECKS to read the "
            "fact, and give it a case in tests/test_lane_reachability.py.")

    refusals: "list[str]" = []
    for name in sorted(requires):
        fact_name, how = _LANE_WIRE_CHECKS[name]
        if how == "every_in_routed_moe":
            structure = facts.get("structure")
            if structure is None:
                refusals.append(
                    f"the unit's structure was not read, so the lane's {name} requirement "
                    f"({requires[name]!r}) cannot be decided; absent evidence is not a pass")
                continue
            if structure not in STRUCTURES:
                refusals.append(
                    f"the unit's structure {structure!r} is not one this build serves "
                    f"({list(STRUCTURES)}), so the lane's {name} requirement cannot be decided")
                continue
            if structure != STRUCTURE_ROUTED_MOE:
                continue                     # decided: the requirement binds expert stacks only
            how = "every_in"
        value = facts.get(fact_name)
        if value is None or (how == "every_in" and not tuple(value)):
            refusals.append(
                f"the unit's {fact_name} was not read, so the lane's {name} requirement "
                f"({requires[name]!r}) cannot be decided; absent evidence is not a pass")
            continue
        if how == "every_in":
            supported = [int(r) for r in requires[name]]
            offending = sorted({int(r) for r in value} - set(supported))
            if offending and name == "column_rates_routed_moe":
                refusals.append(
                    f"{name} {offending} are outside the rates this lane's routed-expert "
                    f"launches reach ({supported}); the lane reads the wire at these rates "
                    "but its routed launches do not reach them on the target (the gate/up "
                    "launch's two tables and word stages exceed the target's shared memory "
                    "above that set), so the stack keeps the compact adapter")
            elif offending:
                refusals.append(
                    f"{name} {offending} are outside the rates this lane reads "
                    f"({supported}); the lane repacks each column at its own rate, "
                    "so one column out of range refuses the whole unit")
        elif how == "one_of":
            allowed = [int(v) for v in requires[name]]
            if int(value) not in allowed:
                refusals.append(
                    f"{name} {int(value)} is outside the {allowed} this lane reads")
        elif how == "carries":
            if bool(value) != bool(requires[name]):
                if bool(requires[name]):
                    refusals.append(
                        f"{name}: the lane reads only units that carry this, "
                        "and the unit does not")
                else:
                    count = "" if isinstance(value, bool) else f"{int(value)} "
                    refusals.append(
                        f"{name}: " + _CARRIES_REFUSALS[name].format(n=count))
        elif how == "state_in":
            allowed = [route_wire_spelling("rotation", v) for v in requires[name]]
            if str(value) not in allowed:
                refusals.append(
                    f"{name} {value} is not among the {allowed} states this lane reads; "
                    "the kernel decodes the wire without applying a rotation")
        else:
            wanted = route_wire_spelling(name, requires[name])
            if str(value) != wanted:
                refusals.append(
                    f"{name} {value} is not the {wanted} {name} this lane reads")
    return refusals


NVFP4_ACTIVATION_CONTRACT = "e2m1_group16_ue4m3_static"

FP8_ACTIVATION_CONTRACT = "fp8_per_token_dynamic"

BF16_ACTIVATION_CONTRACT = "bf16_unquantized"

TESSERA_NVFP4 = "TESSERA_NVFP4"

TESSERA_FP8 = "TESSERA_FP8"

TESSERA_BF16 = "TESSERA_BF16"

ROUTES: dict[str, dict] = {
    TESSERA_NVFP4: {
        "grids": ("E2M1", "E2M1x2"), "plane": "LUT",
        "body": "TCQ", "span": 2,
        "short": "NVFP4",
        "grid_kind": "an E2M1-based",
        "builder": ("tessera.serving.nvfp4_route", "build_tessera_nvfp4_method"),
        "tile": "nvfp4 (packed E2M1 codes, group-16 ue4m3 block scales, one global)",
        "columns_multiple": 16,
        "activation_contract": NVFP4_ACTIVATION_CONTRACT,
        "gemm_symbol": "torch._scaled_mm",
    },
    TESSERA_FP8: {
        "grids": ("E4M3",), "plane": "CHANNEL",
        "body": "WINDOW", "span": 1,
        "short": "FP8",
        "grid_kind": "the scalar E4M3",
        "builder": ("tessera.serving.fp8_route", "build_tessera_fp8_method"),
        "tile": "fp8 per-channel (E4M3 bytes, one fp32 scale per row)",
        "columns_multiple": 16,
        "activation_contract": FP8_ACTIVATION_CONTRACT,
        "gemm_symbol": "torch._scaled_mm",
    },
    TESSERA_BF16: {
        "grids": ("BF16",), "plane": "CHANNEL",
        # The SAME body and span the E4M3 route reads -- the 16-bit family is
        # the window recipe with a wider alphabet, not a different trellis --
        # and ``bf16_route._parse_unit`` refuses anything else by name at load.
        # What differs between the two routes is the alphabet the window table
        # snaps to and the tile the decode lands in, and neither is a body
        # fact -- which is why this row is FP8's row here and in
        # ``sharding.ROUTE_TP_AXES``, for the same reason in both places.
        "body": "WINDOW", "span": 1,
        "short": "BF16",
        "grid_kind": "the scalar BF16",
        "builder": ("tessera.serving.bf16_route", "build_tessera_bf16_method"),
        "tile": "bf16 (the raw table values, one fp32 scale per row applied on the GEMM output)",
        # No K quantum.  The other two routes decode to a PACKED tile whose
        # mainloop reads groups -- a nibble pair, a group-16 block scale -- and
        # 16 is that group.  A bf16 tile is one word per weight, so the GEMM
        # takes any K, and asserting a quantum here would refuse a geometry
        # this route serves.
        "columns_multiple": 1,
        "activation_contract": BF16_ACTIVATION_CONTRACT,
        # NOT ``torch._scaled_mm``: there is no scale to hand a scaled GEMM.
        # The row scale is an fp32 epilogue, so this route calls the stock
        # matmul and says so.
        "gemm_symbol": "torch.mm",
    },
}

TESSERA_FAMILIES = tuple(ROUTES)


def route_family_for_grid(grid: str) -> str:
    """Return the first declared route family that holds a producer grid."""
    for family, route in ROUTES.items():
        if grid in route["grids"]:
            return family
    raise ValueError(f"no route family holds grid {grid!r}")


def _direct_consumer_kind(module: str) -> str | None:
    if module.endswith(".indexer.wk_weights_proj"):
        return "indexer"
    if module.endswith(".kv_b_proj"):
        return "mla"
    return None


def direct_consumer_resident_bytes(module, family, rows, columns, roles):
    """Return the extra resident bytes that the selected direct consumer needs."""
    if family not in TESSERA_FAMILIES:
        raise ValueError(f"{module}: no projection decoder exists for {family!r}")
    consumer = _direct_consumer_kind(module)
    extra = 0
    if consumer == "indexer":
        tail = [int(size) for name, size in roles if name == "weights_proj"]
        if len(tail) != 1:
            raise ValueError(f"{module}: the direct consumer needs one weights_proj role")
        extra = tail[0] * int(columns) * 4
    elif consumer == "mla":
        extra = int(rows) * int(columns) * 2
    return {"resident_bytes_resident_mode": extra, "resident_bytes_stock": extra}


def direct_consumer_activation_contract(module, role):
    """Return the direct role's arithmetic, not its unused dense output."""
    consumer = _direct_consumer_kind(module)
    if consumer == "indexer" and role == "weights_proj":
        return "a32"
    if consumer == "mla" and role == "kv_b_proj":
        return "a16"
    return None


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
    # Lfm2MoeMlp uses w1/w3 source leaves for its dense gate/up pair and
    # constructs one w13 Linear. Routed experts have an intervening
    # .experts.INDEX segment and remain owned by the expert stack rule.
    (re.compile(r"^(.*\.feed_forward\.)(w1|w3)\.weight$"), "w13", ("w1", "w3")),
)

# These architectures have no generic three-member qkv_proj owner.
# GLM KDA uses its six-member input owner; MLA can keep a standalone query.
SEPARATE_QKV_ARCHITECTURES = frozenset({"Glm5NextForConditionalGeneration"})


# GLM member order follows the current loader's stacked_params_mapping.
GLM_FUSED = (
    (re.compile(r"^(.*\.self_attn\.)(q_proj|k_proj|v_proj|b_proj|f_a_proj|g_a_proj)\.weight$"),
     "in_proj_qkvbfg_a", ("q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj", "g_a_proj")),
    (re.compile(r"^(.*\.self_attn\.)(q_a_proj|kv_a_proj_with_mqa)\.weight$"),
     "fused_qkv_a_proj", ("q_a_proj", "kv_a_proj_with_mqa")),
    (re.compile(r"^(.*\.self_attn\.indexer\.)(wk|weights_proj)\.weight$"),
     "wk_weights_proj", ("wk", "weights_proj")),
)


def glm_attention_kind(tensor_name: str, config: Mapping | None) -> str | None:
    """Read the layer kind from the checkpoint construction config."""
    if config is None:
        return None
    text = config.get("text_config", config)
    match = re.search(r"(?:^|\.)layers\.(\d+)\.self_attn\.", tensor_name)
    kinds = text.get("layer_types", ())
    if match and int(match.group(1)) < len(kinds):
        return kinds[int(match.group(1))]
    return None


def fused_module(tensor_name: str, architecture: str | None = None, *,
                 config: Mapping | None = None, tensor_names=None
                 ) -> tuple[str, tuple[str, ...]] | None:
    """Return the dense owner and its ordered checkpoint members.

    GLM q_proj can be a KDA member or a standalone MLA projection.
    The construction config identifies the layer; source names are the fallback.
    Neither rule determines partition sizes.
    """
    separate_qkv = architecture in SEPARATE_QKV_ARCHITECTURES
    if separate_qkv:
        for pattern, fused, members in GLM_FUSED:
            match = pattern.match(tensor_name)
            if match is None:
                continue
            prefix = match.group(1)
            if fused == "in_proj_qkvbfg_a":
                kind = glm_attention_kind(tensor_name, config)
                if kind != "linear_attention":
                    if kind is not None or tensor_names is None or prefix + "b_proj.weight" not in tensor_names:
                        continue
            if fused == "fused_qkv_a_proj" and config is not None:
                text = config.get("text_config", config)
                if "q_lora_rank" in text and text["q_lora_rank"] is None:
                    continue
            return prefix + fused, tuple(prefix + m + ".weight" for m in members)
    for pattern, fused, members in FUSED:
        if separate_qkv and fused == "qkv_proj":
            continue
        match = pattern.match(tensor_name)
        if match:
            return match.group(1) + fused, tuple(match.group(1) + m + ".weight" for m in members)
    return None


def role_name(tensor_name: str) -> str:
    """The scheme role a whole checkpoint tensor is declared under: its leaf."""
    module = tensor_name[: -len(".weight")] if tensor_name.endswith(".weight") else tensor_name
    return module.rsplit(".", 1)[-1]


def glm_fused_groups() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return GLM target leaves and their ordered source member leaves."""
    return tuple((target, members) for _pattern, target, members in GLM_FUSED)
