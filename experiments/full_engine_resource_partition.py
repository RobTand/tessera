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

# An allocation's owner class must be exactly one of these. ``shared`` and
# ``unknown`` supply neither a classification nor an invariance, so a row
# carrying either is unclassified and blocks the partition rather than
# defaulting anywhere.
OWNER_CLASSES = ("fixed", "candidate", "kv")

# Lifetime classes, derived from the replay's own ``lifetime_scope`` plus
# whether the allocation was ever freed inside the captured interval.
LIFETIME_CLASSES = ("resident", "activation", "scratch")

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


def _lifetime_class(row):
    """Persistent, carried across a unit boundary, or invocation-local."""
    if row["free_completed_index"] is None:
        return "resident"
    if row["lifetime_scope"] == "escapes_unit":
        return "activation"
    if row["lifetime_scope"] == "inside_unit":
        return "scratch"
    # Allocated and freed with no unit interval containing either end. It is a
    # real transient — on a live engine attention, norms, routing, sampling and
    # every startup transient land here — but charging it needs a declared step
    # boundary saying how often it recurs, and the capture emits unit intervals
    # only. Calling it fixed scratch would assume once per step; calling it
    # startup would assume never again. Both are fills, so it stays
    # unclassified and named. See the report schema's missing-input list.
    return None


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
    """The candidate unit this allocation is charged to, if any."""
    stack = row["scope_stack"]
    return stack[-1] if stack else None


def classify_allocations(ledger):
    """Split every replayed allocation into (owner class, lifetime class).

    Returns ``(classified, unclassified)``. ``unclassified`` rows are the reason
    a domain stays open; they are named, never dropped and never bucketed.
    """
    classified, unclassified = [], []
    for row in ledger["torch_allocations"]:
        owner, lifetime = _owner_class(row), _lifetime_class(row)
        if owner is None or lifetime is None:
            unclassified.append({
                "allocation_id": row["allocation_id"], "bytes": row["bytes"],
                "observed_categories": row["observed_categories"],
                "lifetime_scope": row["lifetime_scope"],
                "reason": ("no single supported owner category" if owner is None
                           else "freed outside every unit interval; charging it "
                                "needs a declared step boundary the capture does "
                                "not emit"),
            })
            continue
        classified.append({
            "allocation_id": row["allocation_id"], "bytes": row["bytes"],
            "owner_class": owner, "lifetime_class": lifetime,
            "unit": _unit_of(row), "allocate_index": row["allocate_index"],
            "free_completed_index": row["free_completed_index"],
        })
    return classified, unclassified


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
    elif unattributed:
        domains["history_join"] = _domain(
            False, [], "unattributed external CUDA memory records remain",
            refused=True)
    else:
        domains["history_join"] = _domain(
            True, ["unattributed_external_records", "history_join"], None)

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


def derive_partition(ledger, domains=None):
    """Build the partition and the composition terms from one replayed ledger.

    Every number is derived from the ledger's own classified event lifetimes.
    A term is ``None``, and named in ``scope.unavailable_terms``, when a domain
    it depends on is not closed *or* when any allocation is unclassified — an
    unclassified row has neither owner nor lifetime, so it could belong to any
    term, and no term can be complete while one exists.
    """
    if ledger["schema"] != "tessera.full_engine_raw_resource_ledger.v1":
        raise ValueError("unsupported raw ledger schema")
    if domains is None:
        domains = qualify_domains(ledger)
    classified, unclassified = classify_allocations(ledger)
    terminal = 1 + max((row["allocate_index"] for row in ledger["torch_allocations"]),
                       default=0)

    def select(owner, lifetime, unit=None):
        return [row for row in classified
                if row["owner_class"] == owner and row["lifetime_class"] == lifetime
                and (unit is None or row["unit"] == unit)]

    units = sorted({row["unit"] for row in classified if row["unit"] is not None})
    terms, unavailable = {}, []

    def emit(name, value):
        if not unclassified and _term_available(name, domains):
            terms[name] = value
        else:
            terms[name] = None
            unavailable.append(name)

    # Resident bytes add; they are live at the terminal boundary by definition.
    emit("fixed_resident", sum(row["bytes"] for row in select("fixed", "resident")))
    emit("candidate_resident",
         {unit: sum(row["bytes"] for row in select("candidate", "resident", unit))
          for unit in units})
    # Transient maxima come from the simultaneous sweep, never from a sum of
    # per-allocation maxima and never from a difference of peaks.
    emit("fixed_activation", _simultaneous_peak(select("fixed", "activation"), terminal))
    emit("candidate_activation",
         {unit: _simultaneous_peak(select("candidate", "activation", unit), terminal)
          for unit in units})
    emit("fixed_scratch", _simultaneous_peak(select("fixed", "scratch"), terminal))
    emit("candidate_scratch",
         {unit: _simultaneous_peak(select("candidate", "scratch", unit), terminal)
          for unit in units})
    emit("fixed_kv", sum(row["bytes"] for row in select("kv", "resident")))

    return {
        "schema": PARTITION_SCHEMA,
        "identity": ledger.get("identity"),
        "capture_sha256": ledger.get("capture_sha256"),
        "domains": domains,
        "membership": classified,
        "unclassified_allocations": unclassified,
        "units": units,
        "terms": terms,
        "scope": {
            "topology": "tp1_single_device_resident_eager",
            "allocation_scope": "gpu_allocations_only",
            "unavailable_terms": sorted(unavailable),
            "expressible": not unavailable,
            "unclassified_allocation_count": len(unclassified),
            "invariance": "one complete assignment, one row per unit",
        },
    }


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
    return (terms["fixed_resident"]
            + sum(terms["candidate_resident"].values())
            + terms["fixed_activation"]
            + max(terms["candidate_activation"].values(), default=0)
            + terms["fixed_scratch"]
            + max(terms["candidate_scratch"].values(), default=0)
            + terms["fixed_kv"])


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
        if not isinstance(value, dict) or not value:
            raise ValueError(f"report member is missing and is never defaulted: {name}")

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
            "issues": ledger["issues"],
            "artifacts": list(artifacts),
        },
        "partition": partition,
        "derived": {
            "terms": partition["terms"],
            "scalar_budget_bytes": compose_scalar_budget(partition),
            "scope": partition["scope"],
            "domains": partition["domains"],
        },
    }
    assert set(report) == set(REPORT_MEMBERS) | {"schema"}
    return report
