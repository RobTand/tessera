"""Derive the full-engine resource partition from an already-replayed ledger.

The ledger produced by :func:`analyze_engine_resource_ledger` is a raw replay:
it retains every allocation generation, its lifetime scope, its observed owners
and the domains its CUDA records fall in. This module turns that replay into the
partition and the composition terms the consumer recomputes, and it emits a term
only when every domain that term depends on is closed.

Nothing here fills an unknown. A term whose domains are not all closed is
``None``; no whole-engine residual, independent-median subtraction or tolerance
becomes a fixed charge. The numbers this module declares are a claim the
consumer reproduces from the same raw observations without running this code.

See ``docs/design/full_engine_resource_report.md`` for the frozen schema, and
PrismaQuant ``docs/design/runtime_fixed_resource_admission.md`` for what the
consumer independently checks.
"""
from __future__ import annotations

PARTITION_SCHEMA = "tessera.full_engine_resource_partition.v1"

# The six domains are the six qualification gaps the raw ledger names, stated as
# what each must establish rather than as what is absent.
DOMAIN_NAMES = ("worker_startup", "history_join", "external_closure",
                "provenance_admission", "cache_capacity", "timing_partition")

# Which domains each composition term depends on. A term is emitted only when
# every domain it names is closed.
TERM_DOMAINS = {
    "fixed_resident": ("worker_startup", "history_join", "external_closure"),
    "candidate_resident": ("worker_startup", "history_join", "external_closure"),
    "fixed_activation": ("worker_startup", "history_join"),
    "candidate_activation": ("worker_startup", "history_join"),
    "fixed_scratch": ("history_join", "external_closure"),
    "candidate_scratch": ("history_join", "external_closure"),
    "fixed_kv": ("cache_capacity",),
}

# The off-step transient peak is not one of the seven terms and does not share
# their gate. It needs the ownership join and the external closure, exactly as
# a scratch term does, and nothing else: ``cache_capacity`` prices KV and has no
# bearing on it. ``worker_startup`` does bear on it -- an unobserved startup
# prefix can only hide off-step bytes -- but it bounds the number from below
# rather than invalidating it, so the peak is published as a floor and says so
# in ``non_step_transient_peak_scope`` instead of going null.
NON_STEP_PEAK_DOMAINS = ("history_join", "external_closure")

# An allocation's owner class must be exactly one of these. ``shared`` and
# ``unknown`` supply neither a classification nor an invariance, so a row
# carrying either is unclassified and blocks the partition rather than
# defaulting anywhere.
OWNER_CLASSES = ("fixed", "candidate", "kv")

# The (owner, lifetime) cells the seven composition terms actually charge. The
# classifier can produce nine; these are seven. A KV backing with a transient
# lifetime is therefore classified and billed to nothing, and so is a candidate
# allocation carrying no unit, because every candidate term is keyed by unit.
CHARGED_CELLS = (("fixed", "resident"), ("candidate", "resident"),
                 ("fixed", "activation"), ("candidate", "activation"),
                 ("fixed", "scratch"), ("candidate", "scratch"),
                 ("kv", "resident"))

# Lifetime classes, derived from the replay's own ``lifetime_scope``, whether
# the allocation was ever freed inside the captured interval, and -- for a row
# outside every unit -- where its lifetime sits relative to the declared engine
# steps. ``non_step`` is the one class no composition term charges: a row live
# during no declared step is outside the scope of a per-step budget, which is
# what the seven terms compose.
LIFETIME_CLASSES = ("resident", "activation", "scratch", "non_step")

# The two domains whose closure check is implemented here. The other four are
# stated as what is missing, never closed by the presence of an argument: a
# domain that closes because a caller passed a truthy object is ``qualified:
# true`` spelled differently, and this schema does not have that field.
IMPLEMENTED_DOMAINS = ("history_join", "external_closure")

_UNIMPLEMENTED_DOMAINS = {
    "worker_startup":
        "the recorder is proven to attach before CUDA initialization (the replay "
        "refuses a capture that did not), but no capture yet runs inside the "
        "engine's own worker process, so startup is not covered end to end",
    "provenance_admission":
        "no check here recomputes the runtime provenance relation; the "
        "consumer's admission is what closes this domain, and it refuses today",
    "cache_capacity":
        "no check here recomputes pool sizes and resolved limits from the raw "
        "worker records with views deduplicated by physical backing generation",
    "timing_partition":
        "no check here recomposes the measured step from ordered native apply "
        "intervals, and it needs a device with no other work on it, which no "
        "capture has had",
}

_ISSUES_REASON = "the raw ledger carries unresolved issues"

# The fields each declared member carries, and the one execution coordinate this
# producer can stamp a partition scope for. Restated here rather than imported:
# the consumer is another repository and this module may not depend on it. Two
# copies of one contract drift, which is exactly what the consumer's independent
# recomputation exists to catch -- so they are stated, not inferred.
DECLARED_MEMBER_FIELDS = {
    "reference": ("canonical_census", "runtime_binding", "selected_rows"),
    "workload": ("calibration", "prompt_ids", "sampling"),
    "execution": ("graph_mode", "residency", "topology"),
}
SUPPORTED_EXECUTION = {"graph_mode": "eager", "residency": "resident", "topology": "tp1"}

# The scope every partition declares. It is the composite spelling of
# SUPPORTED_EXECUTION, and the report assembler refuses an execution coordinate
# that does not match it rather than stamping this string over the caller's.
SCOPE_TOPOLOGY = "tp1_single_device_resident_eager"


def _owner_class(row):
    """The one owner class this allocation carries, or ``None``.

    ``observed_categories`` is already sorted and deduplicated by the replay. A
    row with no category, more than one, or a ``shared``/``unknown`` category is
    unclassified: ownership is never inferred from a pointer or from what is
    left over.
    """
    categories = row["observed_categories"]
    if len(categories) != 1:
        return None
    category = categories[0]
    return category if category in OWNER_CLASSES else None


def _declared_steps(ledger):
    """The engine steps this classification may rely on, or ``None``.

    Index pairs come back only when the capture declared a step interval for
    **every** engine step it executed. Partial coverage is not simply a smaller
    set of steps: an allocation from an undeclared step is live during a step
    nobody declared, so "live during no declared step" stops being a proof and
    becomes a guess. Partial and unobserved therefore behave identically, which
    is how this module behaved before any step could be declared at all.

    This reads the ledger and nothing else. A caller cannot supply a step table
    for the same reason it cannot supply a domain table.
    """
    intervals, coverage = ledger.get("step_intervals"), ledger.get("step_coverage")
    if not intervals or type(coverage) is not dict or coverage.get("state") != "complete":
        return None
    return [(row["begin_index"], row["end_index"]) for row in intervals]


def _lifetime_class(row, steps):
    """Persistent, carried across a boundary, invocation-local, or off-step."""
    if row["free_completed_index"] is None:
        return "resident"
    if row["lifetime_scope"] == "escapes_unit":
        return "activation"
    if row["lifetime_scope"] == "inside_unit":
        return "scratch"
    # No unit interval contains this allocation. On a live engine attention,
    # norms, routing, sampling and every startup transient land here. Charging
    # one needs a declared step boundary saying how often it recurs: as scratch
    # it would be assumed once per step, as startup never again, and without a
    # boundary both are fills. So without one it stays unclassified and named.
    if steps is None:
        return None
    # With one, the question is answered by liveness rather than by the
    # allocation index alone. A buffer allocated before a step and freed inside
    # it is live during that step and has to be charged; a buffer whose whole
    # lifetime sits between steps is live during none of them.
    begin, end = row["allocate_index"], row["free_completed_index"]
    if any(step_begin <= begin and end <= step_end for step_begin, step_end in steps):
        return "scratch"
    if any(step_begin < end and begin < step_end for step_begin, step_end in steps):
        # Live across a step boundary: carried, exactly as a row that outlives
        # its unit is carried. ``fixed_activation`` and ``fixed_scratch`` are
        # separate additive terms, so this neither double-counts nor drops it.
        return "activation"
    return "non_step"


def _simultaneous_peak(rows, terminal_index):
    """Maximum simultaneous live sum over the rows' own lifetimes.

    Sweeps allocation and free events in history order. A sum of per-allocation
    maxima is not a peak, and neither is the difference of two independent
    peaks; only this sweep derives one.
    """
    events = []
    for row in rows:
        end = row["free_completed_index"]
        events.append((row["allocate_index"], 1, row["bytes"]))
        events.append((terminal_index if end is None else end, 0, row["bytes"]))
    # Frees at an index settle before allocations at the same index, so a
    # reused extent is never counted twice.
    events.sort(key=lambda event: (event[0], event[1]))
    live = peak = 0
    for _, is_allocation, size in events:
        live += size if is_allocation else -size
        peak = max(peak, live)
    return peak


def _unit_of(row):
    """The candidate unit this allocation is charged to, if any.

    The **outermost** unit on the scope stack, not the innermost. Three things
    have to name one interval or the composition is wrong. The replay reads
    ``unit_invocation`` from the outermost containing interval, and it decides
    ``lifetime_scope`` against that same interval. Charging the innermost splits
    a row's lifetime basis from its charge -- and because unit intervals may
    nest, with only *crossing* refused, it puts rows that are simultaneously
    live into two per-unit buckets that ``compose_scalar_budget`` then takes a
    maximum between. Outermost intervals cannot overlap each other, so a maximum
    over them is a maximum over genuine alternatives.
    """
    stack = row["scope_stack"]
    return stack[0] if stack else None


def _checked_allocation_rows(ledger):
    """The replayed allocations, or a refusal, before any arithmetic runs.

    ``analyze_engine_resource_ledger`` cannot emit any of these: it builds every
    size through ``_int(..., minimum=1)`` and a free always follows its own
    allocation. But ``derive_partition`` is a public entry point that takes a
    dict, and the report schema requires a reader that refuses nonfinite values,
    negative sizes and booleans where an integer is declared *before* arithmetic
    rather than after. A free ordered before its allocation is the one that
    matters most: the sweep in :func:`_simultaneous_peak` would settle that free
    first, drive the running sum negative, and return a peak that hides real
    bytes instead of inflating them.
    """
    rows = ledger["torch_allocations"]
    for row in rows:
        where = row["allocation_id"]
        size, begin, end = row["bytes"], row["allocate_index"], row["free_completed_index"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"allocation carries no positive integer size: {where}")
        if isinstance(begin, bool) or not isinstance(begin, int) or begin < 0:
            raise ValueError(f"allocation carries no history index: {where}")
        if end is None:
            continue
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError(f"allocation carries a non-integer free index: {where}")
        if end < begin:
            raise ValueError(f"allocation is freed before it is allocated: {where}")
    return rows


def classify_allocations(ledger):
    """Split every replayed allocation into (owner class, lifetime class).

    Returns ``(classified, unclassified, non_step)``. ``unclassified`` rows are
    the reason a domain stays open; they are named, never dropped and never
    bucketed. ``non_step`` rows are classified and deliberately outside the
    composition: each one is proven, from the capture's own declared step
    intervals, to be live during no engine step, and the seven terms compose a
    per-step budget. They are named too, and priced separately, because an
    engine still has to fit its startup peak even when no step ever reaches it.
    """
    steps = _declared_steps(ledger)
    classified, unclassified, non_step = [], [], []
    for row in _checked_allocation_rows(ledger):
        owner, lifetime = _owner_class(row), _lifetime_class(row, steps)
        if owner is None or lifetime is None:
            unclassified.append({
                "allocation_id": row["allocation_id"], "bytes": row["bytes"],
                "observed_categories": row["observed_categories"],
                "lifetime_scope": row["lifetime_scope"],
                "reason": ("no single supported owner category" if owner is None
                           else "freed outside every unit interval, and no complete "
                                "declared step boundary covers this capture, so "
                                "charging it would assume either once per step or "
                                "never again"),
            })
            continue
        entry = {
            "allocation_id": row["allocation_id"], "bytes": row["bytes"],
            "owner_class": owner, "lifetime_class": lifetime,
            "unit": _unit_of(row), "allocate_index": row["allocate_index"],
            "free_completed_index": row["free_completed_index"],
        }
        (non_step if lifetime == "non_step" else classified).append(entry)
    return classified, unclassified, non_step


def uncharged_allocations(classified):
    """Classified rows that no composition term charges.

    This is the one error direction that must never happen silently. An
    overcount wastes headroom; an undercount hands a serving gate a budget
    smaller than the engine needs, and on a unified-memory box that is an OOM
    that kills the job. Two rows fall through the seven terms: a KV backing
    whose lifetime is transient rather than resident, and a candidate
    allocation with no unit in its scope stack.

    They are named and they null every term, exactly like an unclassified row.
    Charging them somewhere would be inventing a rule the composition does not
    have, and the composition is the consumer's to reproduce, not ours to
    extend.
    """
    uncharged = []
    for row in classified:
        cell = (row["owner_class"], row["lifetime_class"])
        if cell not in CHARGED_CELLS:
            reason = f"no composition term charges a {cell[0]} {cell[1]} allocation"
        elif row["owner_class"] == "candidate" and row["unit"] is None:
            reason = "a candidate allocation carrying no unit is charged by no per-unit term"
        else:
            continue
        uncharged.append({"allocation_id": row["allocation_id"],
                          "bytes": row["bytes"], "owner_class": row["owner_class"],
                          "lifetime_class": row["lifetime_class"],
                          "unit": row["unit"], "reason": reason})
    return uncharged


def qualify_domains(ledger):
    """State each of the six domains from the evidence actually present.

    ``closed`` means the evidence is present and consistent. ``refused`` means
    the evidence is present and contradicts the model. ``open`` means it was
    never observed, or no check that could close it exists yet. The gate blocks
    on ``refused`` and ``open`` alike; the distinction is for the person reading
    the report.

    This function takes the ledger and nothing else. Four of the six domains
    have no implemented closure check, and they say so rather than closing when
    a caller supplies an artifact — an unread argument is not evidence.
    """
    issues = ledger["issues"]
    domains = {}

    # History join: every Torch event has its reciprocal CUPTI record and back.
    unattributed = ledger.get("unattributed_external_records")
    if issues:
        domains["history_join"] = _domain(False, [], _ISSUES_REASON, refused=True)
    elif unattributed is None:
        # Absent and null are "never observed", exactly as they are eleven lines
        # below for external_closure. An empty list is the observation that
        # closes this domain; no list at all is no observation, and a domain
        # that closes on evidence it did not read cites evidence that is not
        # there.
        domains["history_join"] = _domain(
            False, [], "no external CUDA memory record join was observed")
    elif unattributed:
        domains["history_join"] = _domain(
            False, [], "unattributed external CUDA memory records remain",
            refused=True)
    else:
        domains["history_join"] = _domain(
            True, ["unattributed_external_records", "checkpoints"], None)

    # External/context/host closure: the one quantity that closes this domain is
    # a disjoint observed charge, never a residual.
    external = ledger.get("external_native_peak_bytes")
    if issues:
        domains["external_closure"] = _domain(False, [], _ISSUES_REASON, refused=True)
    elif external is None:
        domains["external_closure"] = _domain(
            False, [],
            "external/static/context backings carry no disjoint observed charge")
    else:
        domains["external_closure"] = _domain(
            True, ["external_native_peak_bytes", "cuda_argument_domains"], None)

    for name, reason in _UNIMPLEMENTED_DOMAINS.items():
        domains[name] = _domain(False, [], reason)

    return {name: domains[name] for name in DOMAIN_NAMES}


def _domain(closed, evidence, reason, refused=False):
    state = "closed" if closed else ("refused" if refused else "open")
    return {"state": state,
            "evidence": list(evidence) if closed else [],
            "reason": None if closed else reason}


def _term_available(term, domains):
    return all(domains[name]["state"] == "closed" for name in TERM_DOMAINS[term])


def _compose_terms(classified, units, terminal):
    """The seven composition terms' values, before any availability check.

    Factored out so the arithmetic stays testable without a caller being able to
    hand :func:`derive_partition` a table of closed domains. Recording that a
    caller supplied one was not enough: the populated partition was the object
    that escaped, and the only refusal lived in an assembler that path never
    reached. A test seam that produces a shippable artifact is not a seam.
    """
    def select(owner, lifetime, unit=None):
        return [row for row in classified
                if row["owner_class"] == owner and row["lifetime_class"] == lifetime
                and (unit is None or row["unit"] == unit)]

    return {
        # Resident bytes add; they are live at the terminal boundary by definition.
        "fixed_resident": sum(row["bytes"] for row in select("fixed", "resident")),
        "candidate_resident":
            {unit: sum(row["bytes"] for row in select("candidate", "resident", unit))
             for unit in units},
        # Transient maxima come from the simultaneous sweep, never from a sum of
        # per-allocation maxima and never from a difference of peaks.
        "fixed_activation": _simultaneous_peak(select("fixed", "activation"), terminal),
        "candidate_activation":
            {unit: _simultaneous_peak(select("candidate", "activation", unit), terminal)
             for unit in units},
        "fixed_scratch": _simultaneous_peak(select("fixed", "scratch"), terminal),
        "candidate_scratch":
            {unit: _simultaneous_peak(select("candidate", "scratch", unit), terminal)
             for unit in units},
        "fixed_kv": sum(row["bytes"] for row in select("kv", "resident")),
    }


def derive_partition(ledger):
    """Build the partition and the composition terms from one replayed ledger.

    Every number is derived from the ledger's own classified event lifetimes.
    A term is ``None``, and named in ``scope.unavailable_terms``, when a domain
    it depends on is not closed *or* when any allocation is unclassified or
    charged by no term -- such a row could belong to any term, and no term can
    be complete while one exists.

    A row live during no declared engine step is split out instead: the terms
    compose one step, so no term charges it, and it is named, counted and priced
    separately rather than nulling anything. That exemption is a proof from the
    capture's own declared step intervals, not a cell the composition forgot, and
    it holds only while ``step_coverage`` is complete.

    There is no way to hand this function a domain table. Four domains can never
    close from a real ledger, so the arithmetic is exercised through
    :func:`_compose_terms`, which returns values and never an artifact.

    ``scope.topology`` states the execution coordinate this partition is valid
    in. The raw ledger does not carry one, so this function cannot check it;
    :func:`assemble_full_engine_resource_report` refuses a declared coordinate
    that disagrees rather than stamping this one over it.
    """
    if ledger["schema"] != "tessera.full_engine_raw_resource_ledger.v1":
        raise ValueError("unsupported raw ledger schema")
    domains = qualify_domains(ledger)
    classified, unclassified, non_step = classify_allocations(ledger)
    terminal = 1 + max((row["allocate_index"] for row in ledger["torch_allocations"]),
                       default=0)
    units = sorted({row["unit"] for row in classified if row["unit"] is not None})
    uncharged = uncharged_allocations(classified)
    raw_terms = _compose_terms(classified, units, terminal)
    # Priced by the same sweep as every other transient maximum, and never
    # folded into a term: the seven terms compose one engine step, and these
    # bytes are live during none of them. The placement obligation is
    # max(scalar_budget_bytes, non_step_transient_peak_bytes), which is the
    # consumer's to apply -- hiding this number inside a term would invent a
    # rule the composition does not have, and dropping it would let a startup
    # peak above the per-step budget pass a gate that never saw it.
    non_step_peak = _simultaneous_peak(non_step, terminal)

    if not unclassified and not uncharged:
        # Every charged byte is now inside some term, so the composition has to
        # cover the largest simultaneous live sum the replay actually saw. Below
        # it is not disclosed conservatism -- it is a contradiction, and the
        # error direction that kills a job: an undercount hands a serving gate a
        # budget smaller than the engine needs, and on unified memory that is an
        # OOM rather than a spill. Both sides are the same quantity, requested
        # allocation bytes excluding allocator rounding, which is what the
        # observation's own scope field says. Each of this module's three silent
        # undercounts -- a cell no term charged, a candidate row with no unit,
        # and a per-unit maximum taken over units that can be live at once --
        # would have been caught here. It raises rather than nulling a term,
        # because a composition that contradicts its own observations is a
        # defect in this code, not a property of the capture.
        observed_peak = ledger.get("torch_observed_live_peak_bytes")
        # The floor is the peak over the rows the terms actually charge. When
        # nothing was excluded as off-step that is the whole capture, so the
        # ledger's own number is used and the two are then required to agree --
        # a disagreement means the two modules replayed different rows, which is
        # a defect in one of them rather than a property of the capture.
        charged_peak = _simultaneous_peak(classified, terminal)
        floor = observed_peak if (not non_step and observed_peak is not None) else charged_peak
        composed = _compose(raw_terms)
        if composed < floor:
            raise ValueError(
                f"composed budget {composed} is below the observed simultaneous "
                f"live peak {floor}; the composition omits live bytes")
        if not non_step and observed_peak is not None and observed_peak != charged_peak:
            raise ValueError(
                f"the ledger's observed live peak {observed_peak} and this "
                f"partition's sweep over the same allocations {charged_peak} "
                f"disagree; one of the two replayed different rows")

    peak_available = (not unclassified and not uncharged
                      and all(domains[name]["state"] == "closed"
                              for name in NON_STEP_PEAK_DOMAINS))
    peak_scope = ("off-step transient bytes observed in this capture"
                  if domains["worker_startup"]["state"] == "closed" else
                  "off-step transient bytes observed in this capture; a floor rather "
                  "than the startup peak, because worker_startup is open and the "
                  "prefix before the recorder attached is unobserved")

    terms, unavailable = {}, []
    for name, value in raw_terms.items():
        if not unclassified and not uncharged and _term_available(name, domains):
            terms[name] = value
        else:
            terms[name] = None
            unavailable.append(name)

    return {
        "schema": PARTITION_SCHEMA,
        "identity": ledger.get("identity"),
        "capture_sha256": ledger.get("capture_sha256"),
        "domains": domains,
        # Always "derived", because there is no other way to reach this function.
        # The field stays because the consumer reads it and refuses anything
        # else: it is the assertion that crosses the repository boundary, where
        # a hand-written artifact is the only thing that could claim otherwise.
        "domains_source": "derived",
        "membership": classified,
        "unclassified_allocations": unclassified,
        "uncharged_allocations": uncharged,
        # Named in full, always, whether or not its price is expressible. A
        # consumer reproduces this split from observations.step_intervals and
        # observations.step_coverage without running this code.
        "non_step_allocations": non_step,
        "non_step_transient_peak_bytes": non_step_peak if peak_available else None,
        "non_step_transient_peak_scope": peak_scope,
        "units": units,
        "terms": terms,
        "scope": {
            "topology": SCOPE_TOPOLOGY,
            "allocation_scope": "gpu_allocations_only",
            "unavailable_terms": sorted(unavailable),
            "expressible": not unavailable,
            "unclassified_allocation_count": len(unclassified),
            "uncharged_allocation_count": len(uncharged),
            # A count, not a charge: it stays readable even when every term is
            # null, so a reader can see the hazard before there is a price on it.
            "non_step_allocation_count": len(non_step),
            "step_coverage": (ledger.get("step_coverage") or {}).get("state"),
            "invariance": "one complete assignment, one row per unit",
        },
    }


def _compose(terms):
    """The scalar composition over term values that are all present."""
    return (terms["fixed_resident"]
            + sum(terms["candidate_resident"].values())
            + terms["fixed_activation"]
            + max(terms["candidate_activation"].values(), default=0)
            + terms["fixed_scratch"]
            + max(terms["candidate_scratch"].values(), default=0)
            + terms["fixed_kv"])


def compose_scalar_budget(partition):
    """The consumer's conservative scalar composition, or ``None``.

    ``fixed_resident + sum(candidate_resident) + fixed_activation +
    max(candidate_activation) + fixed_scratch + max(candidate_scratch) +
    fixed_KV``.

    Returns ``None`` when any term is unavailable. The composition can exceed
    the measured instantaneous peak because independent maxima need not
    coincide; that is disclosed conservatism, not permission to charge one
    physical extent twice.
    """
    terms = partition["terms"]
    if any(value is None for value in terms.values()):
        return None
    return _compose(terms)


REPORT_SCHEMA = "tessera.full_engine_resource_report.v1"

# The seven envelope members, in the frozen order.
REPORT_MEMBERS = ("identity", "reference", "workload", "execution",
                  "observations", "partition", "derived")

# The three the raw ledger cannot supply. The producer refuses each by name
# rather than defaulting it: a report that invents its own reference row or its
# own workload digest is exactly the failure the consumer's independent
# recomputation exists to catch.
_DECLARED_MEMBERS = ("reference", "workload", "execution")


def assemble_full_engine_resource_report(ledger, *, reference, workload,
                                         execution, artifacts=()):
    """Assemble the frozen seven-member report envelope for one capture.

    ``identity``, ``observations``, ``partition`` and ``derived`` are read or
    derived from the ledger. ``reference``, ``workload`` and ``execution`` are
    the caller's declarations of coordinates the raw ledger does not carry —
    the census and assignment, the workload and its token rows, and the
    execution coordinate — and each is refused when absent.

    ``derived`` is a claim, never an input. The consumer recomputes every
    number in it from ``partition`` and ``observations`` and admits nothing on
    disagreement, so nothing here is authoritative because the producer said
    it.
    """
    declared = {"reference": reference, "workload": workload, "execution": execution}
    for name in _DECLARED_MEMBERS:
        value = declared[name]
        if not isinstance(value, dict):
            raise ValueError(f"report member is missing and is never defaulted: {name}")
        fields = DECLARED_MEMBER_FIELDS[name]
        if set(value) != set(fields):
            raise ValueError(
                f"report member {name} declares {sorted(value)}; this schema's "
                f"fields are {sorted(fields)}")
        # A dict of nulls is truthy, so "is it empty" was never the question. A
        # member every one of whose fields is null declares nothing, and a
        # report that carries it says it has a reference row when it has none.
        if all(field_value is None for field_value in value.values()):
            raise ValueError(f"report member declares every field null: {name}")

    # The scope this producer stamps is the composite spelling of one execution
    # coordinate. A caller declaring another one gets a refusal, not a partition
    # whose scope contradicts the declaration inside the same envelope: the
    # schema says a report outside that scope refuses rather than projecting,
    # and a stamped constant is exactly the projection it forbids.
    if execution != SUPPORTED_EXECUTION:
        raise ValueError(
            f"execution coordinate {sorted(execution.items())} is outside this "
            f"schema's scope {sorted(SUPPORTED_EXECUTION.items())}, and the "
            f"scope is never projected over it")

    partition = derive_partition(ledger)
    report = {
        "schema": REPORT_SCHEMA,
        "identity": {
            "run": ledger.get("identity"),
            "capture_sha256": ledger.get("capture_sha256"),
            # Carried, not dropped: a capture that says on its face that it is
            # synthetic must keep saying so in every artifact derived from it.
            "fixture_provenance": ledger.get("fixture_provenance"),
        },
        "reference": reference,
        "workload": workload,
        "execution": execution,
        "observations": {
            "capture_sha256": ledger.get("capture_sha256"),
            "torch_allocations": ledger["torch_allocations"],
            "checkpoints": ledger.get("checkpoints"),
            "cuda_argument_domains": ledger.get("cuda_argument_domains"),
            "unattributed_external_records": ledger.get("unattributed_external_records"),
            "external_native_peak_bytes": ledger.get("external_native_peak_bytes"),
            "torch_observed_live_peak_bytes": ledger.get("torch_observed_live_peak_bytes"),
            "torch_observed_live_peak_scope": ledger.get("torch_observed_live_peak_scope"),
            # The declared engine steps and the coverage claim over them. These
            # are what a consumer reads to reproduce the off-step filter; without
            # them in the envelope the filter would be a producer rule nobody
            # else could check, which is the shape this schema exists to refuse.
            "step_intervals": ledger.get("step_intervals"),
            "step_coverage": ledger.get("step_coverage"),
            "issues": ledger["issues"],
            # Named and null, never absent. A consumer must be able to tell
            # "this capture did not observe it" from "the producer forgot to
            # carry it", and a missing key says neither. Each member below is
            # what its domain would close on, and #399 owes every one of them:
            # without them the four unclosable domains are unclosable *from the
            # artifact* too, so the scalar composition cannot complete even on a
            # perfect capture. That is the gap, stated where a reader sees it.
            "worker_startup_records": ledger.get("worker_startup_records"),
            "runtime_provenance_relation": ledger.get("runtime_provenance_relation"),
            "kv_observations": ledger.get("kv_observations"),
            "timing_captures": ledger.get("timing_captures"),
            "owner_views": ledger.get("owner_views"),
            "observer_qualification": ledger.get("observer_qualification"),
            "artifacts": list(artifacts),
        },
        "partition": partition,
        # ``derived`` carries the recomputed numbers and their scope. It does
        # not restate ``partition["domains"]``: two copies of one claim invite
        # drift, and the consumer would have to compare them to find out which
        # is authoritative.
        "derived": {
            "terms": partition["terms"],
            "scalar_budget_bytes": compose_scalar_budget(partition),
            # First-class, beside the budget it does not belong to. The seven
            # terms price one engine step; this prices what the engine still
            # holds when no step is running. A reader who takes the budget alone
            # takes the smaller of two numbers the box has to satisfy.
            "non_step_transient_peak_bytes": partition["non_step_transient_peak_bytes"],
            "non_step_transient_peak_scope": partition["non_step_transient_peak_scope"],
            "placement_obligation": "max(scalar_budget_bytes, non_step_transient_peak_bytes)",
            "scope": partition["scope"],
        },
    }
    assert set(report) == set(REPORT_MEMBERS) | {"schema"}
    # Evidence that points at nothing cannot be checked. Every id a closed
    # domain cites has to name an observation this envelope carries.
    for name, domain in partition["domains"].items():
        for observation in domain["evidence"]:
            if observation not in report["observations"]:
                raise ValueError(
                    f"domain {name} cites an observation the report does not "
                    f"carry: {observation}")
    return report
