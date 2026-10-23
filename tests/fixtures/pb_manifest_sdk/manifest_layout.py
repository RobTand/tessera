"""Manifest read order, phase frontiers and entry-aligned movement ranges."""
from __future__ import annotations

from collections.abc import Iterable, Mapping


class EntryExceedsWindow(ValueError):
    """A manifest entry larger than the window of the tier it moves into (#965).

    A mover stages whole entries, so no chunk can be smaller than the entry
    it holds.  An entry larger than the tier's window would need a chunk the
    window can never admit, and its mover would wait for capacity that never
    comes.  The seal refuses it instead, naming the entry.
    """

    def __init__(self, *, entry_index: int, start_bytes: int, end_bytes: int,
                 window_bytes: int) -> None:
        self.entry_index = entry_index
        self.start_bytes = start_bytes
        self.end_bytes = end_bytes
        self.window_bytes = window_bytes
        super().__init__(
            f"read-order entry {entry_index} at [{start_bytes}, {end_bytes}) "
            f"is {end_bytes - start_bytes} bytes, past the {window_bytes}-byte "
            f"window; a mover stages whole entries, so no chunk can hold it")


def split_range_into_chunks(start_bytes: int, end_bytes: int,
                            chunk_bytes: int, *,
                            entry_bytes: Iterable[int],
                            window_bytes: int | None = None,
                            ) -> list[tuple[int, int]]:
    """One phase's ``[start, end)`` as entry-aligned chunk ranges, in read order.

    ``entry_bytes`` is the size of every manifest entry in the order the
    action reads them (``manifest_read_entries``), and the range must start
    and end on those entries' boundaries, as every phase from
    ``manifest_phase_ranges`` does.  A mover stages whole entries: it covers
    every entry its range touches.  A cut inside an entry would hand both
    neighbouring movers more bytes than their tokens reserved, and each
    would refuse ``residency_overran_reservation`` on every attempt (#965).

    Cuts therefore fall only between entries.  Whole entries are packed in
    read order while the chunk stays within ``chunk_bytes``.  An entry larger
    than ``chunk_bytes`` becomes a chunk of its own.  A range that fits is one
    chunk over the whole range.

    The chunks tile the range, so their bytes sum to the phase's entry bytes
    and each entry lands in exactly one chunk.  A chunk's reservation is its
    bytes rounded up to whole tokens, like every reservation, so a phase's
    chunks can reserve up to one token per extra chunk more than the whole
    phase would.  That is rounding, not bytes counted twice.

    ``window_bytes`` is the most the tier ever holds.  An entry in the range
    larger than it raises ``EntryExceedsWindow``.  ``None`` checks nothing.
    """

    if (isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int)
            or chunk_bytes <= 0):
        raise ValueError("chunk_bytes must be a positive whole number of bytes")
    for name, value in (("start_bytes", start_bytes),
                        ("end_bytes", end_bytes)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative whole number of bytes")
    if end_bytes <= start_bytes:
        raise ValueError("a chunked range must be non-empty and half-open")
    if window_bytes is not None and (
            isinstance(window_bytes, bool) or not isinstance(window_bytes, int)
            or window_bytes <= 0):
        raise ValueError("window_bytes must be a positive whole number of bytes")
    chunks: list[tuple[int, int]] = []
    cut: int | None = None
    reached: int | None = None
    position = 0
    for index, size in enumerate(entry_bytes):
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"entry {index} must be a non-negative whole number of bytes")
        entry_start, entry_end = position, position + size
        position = entry_end
        if entry_end <= start_bytes:
            continue
        if entry_start >= end_bytes:
            break
        if entry_start < start_bytes:
            raise ValueError(
                f"start_bytes {start_bytes} is not an entry boundary: entry "
                f"{index} spans [{entry_start}, {entry_end})")
        if entry_end > end_bytes:
            raise ValueError(
                f"end_bytes {end_bytes} is not an entry boundary: entry "
                f"{index} spans [{entry_start}, {entry_end})")
        if window_bytes is not None and size > window_bytes:
            raise EntryExceedsWindow(
                entry_index=index, start_bytes=entry_start,
                end_bytes=entry_end, window_bytes=window_bytes)
        if cut is None:
            cut = entry_start
        if entry_end - cut > chunk_bytes and entry_start > cut:
            chunks.append((cut, entry_start))
            cut = entry_start
        if entry_end - cut > chunk_bytes:
            chunks.append((cut, entry_end))
            cut = entry_end
        reached = entry_end
    if cut is None or reached != end_bytes:
        raise ValueError(
            f"end_bytes {end_bytes} is not an entry boundary of the read "
            f"order, which reaches {position} bytes")
    if cut < end_bytes:
        chunks.append((cut, end_bytes))
    return chunks


def _checked_phase_boundaries(
    phases: list, *, total: int, entry_boundaries: set[int],
) -> list[tuple[str, int]] | None:
    """One refusal rule for both manifest schemas' phase tables (#594).

    A phase that is not a mapping, a name that is missing, empty, repeated
    or not a string, a cumulative that is not an integer (bools and floats
    included -- ``int()`` accepts ``True`` and truncates, and neither is a
    byte count), a table that steps back, overruns its total, cuts an entry,
    or ends anywhere but the total is not a description of this manifest,
    and a mover priced off it would reserve for bytes nobody reads.  ``None``
    is that refusal; the caller answers it with ``[]``.
    """

    boundaries: list[tuple[str, int]] = []
    previous = 0
    seen: set[str] = set()
    for phase in phases:
        if not isinstance(phase, Mapping):
            return None
        name = phase.get("name")
        cumulative = phase.get("cumulative_bytes")
        if not isinstance(name, str) or not name or name in seen:
            return None
        if isinstance(cumulative, bool) or not isinstance(cumulative, int):
            return None
        if (cumulative < previous or cumulative > total
                or cumulative not in entry_boundaries):
            return None
        seen.add(name)
        boundaries.append((name, cumulative))
        previous = cumulative
    if previous != total:
        return None
    return boundaries


def _read_order_indices(phases: list, entry_count: int) -> list[int] | None:
    """The entry indices of the v2 read plan's consumption order, or ``None``.

    v1 consumes ``entries`` in list order, so its boundaries are the list's
    own prefix sums.  v2 consumes them in ``read_plan`` order, which exists
    to differ -- so the boundaries a v2 table is checked against are the
    prefix sums of the plan's own ``entry_indices`` expansion, not of the
    list.  Checking a reordered plan against list-order sums would refuse a
    table the core validator accepted, which is the two-readers disagreement
    this shared rule removes.  A plan that does not say what it reads, or
    names entries it cannot have, refuses.
    """

    order: list[int] = []
    for phase in phases:
        indices = phase.get("entry_indices")
        if not isinstance(indices, list):
            return None
        seen: set[int] = set()
        for ref in indices:
            if isinstance(ref, bool) or not isinstance(ref, int):
                return None
            if ref < 0 or ref >= entry_count:
                return None
            if ref in seen:
                return None
            seen.add(ref)
            order.append(ref)
    return order


def _manifest_read_layout(
    manifest: Mapping[str, object],
) -> tuple[list[tuple[str, int]], list[Mapping[str, object]]] | None:
    """The validated phase boundaries and the entries in read order, or ``None``.

    One reading of the manifest for both of the things a seal cuts on: the
    phase table (``manifest_phase_ranges``) and the entry boundaries inside
    it (``manifest_read_entries``, which ``split_range_into_chunks`` cuts
    at).  A table this refuses describes no read order at all.
    """

    schema = manifest.get("schema")
    if schema == "prismaquant.prismabuild.data_manifest.v2":
        plan = manifest.get("read_plan")
        if not isinstance(plan, Mapping):
            return None
        phases = plan.get("phases")
        if not isinstance(phases, list) or not phases:
            return None
        for phase in phases:
            if not isinstance(phase, Mapping):
                return None
        entries = manifest.get("entries", []) or []
        entry_sizes: list[int] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                return None
            entry_sizes.append(int(entry.get("bytes", 0) or 0))
        total = int(manifest.get("total_bytes", 0) or 0)
        if sum(entry_sizes) != total:
            return None
        read = plan.get("read_bytes")
        if isinstance(read, bool) or not isinstance(read, int):
            return None
        order = _read_order_indices(phases, len(entry_sizes))
        if order is None:
            return None
        read_sizes = [entry_sizes[index] for index in order]
        if sum(read_sizes) != read:
            return None
        entry_boundaries: set[int] = set()
        running = 0
        for size in read_sizes:
            running += size
            entry_boundaries.add(running)
        checked = _checked_phase_boundaries(
            phases, total=read, entry_boundaries=entry_boundaries)
        if checked is None:
            return None
        return checked, read_entries(manifest)
    annotations = manifest.get("annotations")
    if not isinstance(annotations, Mapping):
        return None
    declared = annotations.get("phases")
    if not isinstance(declared, list) or not declared:
        return None
    total = int(manifest.get("total_bytes", 0) or 0)
    entries = manifest.get("entries", []) or []
    entry_boundaries = set()
    running = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            return None
        running += int(entry.get("bytes", 0) or 0)
        entry_boundaries.add(running)
    if running != total:
        return None
    checked = _checked_phase_boundaries(
        declared, total=total, entry_boundaries=entry_boundaries)
    if checked is None:
        return None
    return checked, list(entries)


def manifest_phase_ranges(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    """The manifest's read order as half-open byte ranges, or ``[]``.

    Both manifest schemas already declare the same thing: a running byte sum
    over the entries in the order the action consumes them.  v2 carries it in
    ``read_plan.phases`` (validated entry by entry in ``core``), v1 in
    ``annotations.phases``.  This turns either into the ranges a movement node
    names, so that the range in a mover's demand is a quotation from the
    consumer's own manifest rather than a number somebody chose.

    A v1 table that does not describe this manifest -- a boundary that cuts an
    entry in half, a sum that does not end at ``total_bytes`` -- yields ``[]``,
    the same refusal ``manifest_phases`` makes, because windowing
    on the wrong boundaries reserves for bytes nobody will read.  The v2 table
    in ``read_plan.phases`` is held to the same rule, against the plan's own
    consumption order: both schemas declare a running byte sum over the
    entries in the order the action consumes them, so one validator reads
    both.
    """

    layout = _manifest_read_layout(manifest)
    if layout is None:
        return []
    ranges: list[dict[str, object]] = []
    previous = 0
    for name, cumulative in layout[0]:
        if cumulative > previous:
            ranges.append({"name": name, "start_bytes": previous, "end_bytes": cumulative})
        previous = cumulative
    return ranges


def manifest_prelaunch_phases(
        manifest: Mapping[str, object]) -> list[str]:
    """The declared prelaunch-resident phases of the manifest, in order, or ``[]``.

    A phase declares with a literal ``resident_before_launch: true`` (#1594):
    on a v1 ``annotations.phases`` entry, on a v2 ``read_plan.phases``
    entry.  The declaration is part of the content hash.  Only a literal
    boolean true declares.  Other present values refuse.  A declared set
    past the prefix refuses.  A manifest with no read order declares
    nothing.
    """

    layout = _manifest_read_layout(manifest)
    if layout is None:
        return []
    names = [name for name, _ in layout[0]]
    if manifest.get("schema") == "prismaquant.prismabuild.data_manifest.v2":
        plan = manifest.get("read_plan")
        raw_phases = (plan.get("phases") if isinstance(plan, Mapping)
                      else None)
    else:
        annotations = manifest.get("annotations")
        raw_phases = (annotations.get("phases")
                      if isinstance(annotations, Mapping) else None)
    if not isinstance(raw_phases, list):
        return []
    declared: list[str] = []
    for index, phase in enumerate(raw_phases):
        if not isinstance(phase, Mapping):
            return []
        if "resident_before_launch" not in phase:
            continue
        value = phase["resident_before_launch"]
        if type(value) is not bool:
            raise ValueError(
                f"data manifest phase {index} resident_before_launch "
                f"must be a boolean, got {value!r}")
        if value:
            declared.append(str(phase.get("name")))
    if declared != names[:len(declared)]:
        raise ValueError(
            f"data manifest prelaunch phases {declared} are not a "
            f"contiguous prefix of the read order {names}")
    return declared


def manifest_read_entries(
        manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    """The manifest's entries in the order its action reads them, or ``[]``.

    The order both movers walk through :func:`read_entries`:
    list order for v1, the ``read_plan`` expansion for v2.  A seal cuts
    chunks at these entries' boundaries, so the seal and the movers must
    read one order.  A manifest whose phase table ``manifest_phase_ranges``
    refuses yields ``[]``, because it describes no read order to cut.
    """

    layout = _manifest_read_layout(manifest)
    return [] if layout is None else list(layout[1])


def manifest_phases(manifest: dict) -> list[dict[str, object]]:
    """Phase frontiers; unphased v1 manifests have no progress frontier."""
    if manifest.get("schema") == "prismaquant.prismabuild.data_manifest.v2":
        return [{"name": phase["name"],
                 "cumulative_bytes": phase["cumulative_bytes"]}
                for phase in manifest["read_plan"]["phases"]]
    layout = _manifest_read_layout(manifest)
    return ([] if layout is None else
            [{"name": name, "cumulative_bytes": cumulative}
             for name, cumulative in layout[0]])


def read_entries(manifest: dict) -> list[dict[str, object]]:
    """Actual read timeline; v1 keeps its original consumption-order list."""

    entries = manifest["entries"]
    if manifest.get("schema") != "prismaquant.prismabuild.data_manifest.v2":
        return list(entries)
    return [entries[index]
            for phase in manifest["read_plan"]["phases"]
            for index in phase["entry_indices"]]


def manifest_read_bytes(manifest: dict) -> int:
    if manifest.get("schema") == "prismaquant.prismabuild.data_manifest.v2":
        return int(manifest["read_plan"]["read_bytes"])
    return int(manifest["total_bytes"])


def phase_index(phase_names: Iterable[object], phase_name: str | None) -> int | None:
    """The first position matching a reported phase, or ``None`` if unknown."""

    return next((index for index, name in enumerate(phase_names)
                 if name == phase_name), None)


def consumed_through(phases: list[dict[str, object]], phase_name: str) -> int:
    """Bytes an action reporting ``phase_name`` has finished reading.

    The phase it names is the one it is reading now, so what it is done with
    is everything before that phase began.  Counting the named phase's own
    bytes as consumed would release a window the action is still inside.
    A name this manifest does not carry releases nothing.
    """

    index = phase_index((phase["name"] for phase in phases), phase_name)
    return 0 if index is None or index == 0 else int(
        phases[index - 1]["cumulative_bytes"])


def window_target(phases: list[dict[str, object]], entries: list[dict[str, object]],
                  *, consumed: int, budget: int) -> tuple[int, str]:
    """The furthest entry boundary that stays within ``budget`` of the reader.

    Phase boundaries are accepted-progress frontiers: only a matching worker
    observation can move ``consumed`` to one.  They are not a minimum warm
    unit.  An oversized phase can contain many manifest entries, and any
    entry-aligned prefix inside it is safe to fault into ARC without claiming
    that the action has consumed those bytes.  Return the target boundary and
    its phase name when it happens to end a phase; an in-phase target has no
    phase name.
    """

    phase_names = {int(phase["cumulative_bytes"]): str(phase["name"])
                   for phase in phases}
    target = 0
    cumulative = 0
    for entry in entries:
        cumulative += int(entry["bytes"])
        if cumulative <= consumed:
            continue
        if cumulative - consumed > budget:
            break
        target = cumulative
    return target, phase_names.get(target, "")


def entries_between(entries: list[dict[str, object]], start: int,
                    end: int) -> list[dict[str, object]]:
    """The entries covering ``[start, end)`` of the manifest's own byte order.

    ``cumulative_bytes`` is a sum over these same entries, so a phase boundary
    falls between two of them and the slice is exact.  An entry straddling the
    start is included whole, because a warm reads files, not byte ranges.
    """

    chosen: list[dict[str, object]] = []
    position = 0
    for entry in entries:
        size = int(entry.get("bytes", 0))
        if position >= end:
            break
        if position + size > start:
            chosen.append(entry)
        position += size
    return chosen
