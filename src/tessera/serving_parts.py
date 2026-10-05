"""Whole-layer export ownership and checked assembly of serving checkpoints.

A part has no config.json, so it cannot masquerade as a loadable checkpoint.
The encoded containers are copied unchanged; only their index and summaries
are assembled. This module deliberately needs no tensor runtime.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCHEMA = "tessera.serving-part.v1"
BODY_LAYER = re.compile(r"^model\.(?:[^.]+\.)*layers\.(\d+)\.")


def write_serving_manifest(path: Path, manifest: dict) -> None:
    """Write compact UTF-8 JSON, preserving the manifest's insertion order.

    This is a serialization choice for new artifacts, not a canonical identity:
    file bindings still hash the bytes on disk. Never reserialize an already
    bound artifact just to change its whitespace.
    """
    path.write_bytes(json.dumps(manifest, separators=(",", ":")).encode("utf-8"))


def require_json(value, what: str) -> None:
    """Refuse ``value`` unless :func:`write_serving_manifest` can serialize it.

    A driver calls this on a record it will write at the end of a long run,
    so a non-JSON value (an argparse ``Path``, say) refuses in seconds and by
    its key, not after the encode. Raises ``ValueError`` naming ``what`` and
    the dotted key of the first offending value.
    """
    try:
        json.dumps(value, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{what} is not JSON: {_first_non_json(value, what) or exc}") from exc


def _first_non_json(value, where):
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, (str, int, float, bool)) and key is not None:
                return f"{where} has a {type(key).__name__} key"
            found = _first_non_json(item, f"{where}.{key}")
            if found:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _first_non_json(item, f"{where}[{index}]")
            if found:
                return found
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return None
    return f"{where} is {type(value).__name__}"


def unique_json_pairs(pairs):
    """Object-pairs hook refusing duplicate keys (tessera#703).

    Single home for the strict-load discipline: ``cached_unit.read_manifest``
    uses this rather than keeping its own copy, so the byte layer stays
    torch-free and every strict reader refuses the same way.
    """
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def read_serving_manifest(path: "str | Path") -> dict:
    """Load ``tessera_serving_manifest.json``, refusing duplicate keys.

    The strict counterpart to :func:`write_serving_manifest`: every reader
    of a serving manifest goes through here so a duplicate-keyed sidecar is
    refused at read instead of being admitted silently.
    """
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_json_pairs)


def parse_partition(value: str) -> tuple[int, int]:
    try:
        index, count = map(int, value.split("/"))
        if count < 1 or not 0 <= index < count:
            raise ValueError
    except (ValueError, TypeError):
        raise ValueError("partition must be INDEX/COUNT with 0 <= INDEX < COUNT") from None
    return index, count


def partition_owner(name: str, count: int) -> int:
    """A fused dense module and every expert of a stack share one owner."""
    match = BODY_LAYER.match(name)
    return int(match.group(1)) % count if match else 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _affinity_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, os.cpu_count() or 1)


def sha256_files(paths, workers=None, hash_file=None) -> list:
    """:func:`sha256_file` of each path, in the order given.

    Runs one file per thread, at most ``workers`` at a time (the CPUs this
    process may run on when ``None``); ``hashlib`` and file reads release the
    GIL, so a multi-shard pass is bounded by cores rather than by one core
    (tessera#499). The result is what the serial loop returns, and a refusal
    is the one the serial loop would raise first: results are collected in
    ``paths`` order, so an earlier file's exception surfaces before a later
    file's. ``workers=1`` is the serial loop itself. ``hash_file`` replaces
    :func:`sha256_file` per path, e.g. a source digest cache's lookup.
    """
    paths = list(paths)
    hash_file = hash_file or sha256_file
    workers = min(len(paths), workers if workers is not None else _affinity_cpus())
    if workers <= 1:
        return [hash_file(path) for path in paths]
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-sha256") as pool:
        futures = [pool.submit(hash_file, path) for path in paths]
        try:
            return [future.result() for future in futures]
        finally:
            for future in futures:
                future.cancel()


class _HashAhead:
    """:func:`sha256_file` of known paths, started ahead on a bounded thread pool.

    :meth:`sha256` takes one path's digest at the point the serial loop hashed
    it, re-raising that file's exception there, so a check that reads digests
    one at a time still refuses the file it refused serially (tessera#499). A
    path that was not started ahead is hashed on the spot. Leaving the block
    cancels hashes that have not started and waits for running ones.
    """

    def __init__(self, paths, workers=None):
        paths = list(dict.fromkeys(paths))
        workers = min(len(paths), workers if workers is not None else _affinity_cpus())
        self._pool = (ThreadPoolExecutor(max_workers=workers, thread_name_prefix="output-sha256")
                      if workers > 1 else None)
        self._futures = ({path: self._pool.submit(sha256_file, path) for path in paths}
                         if self._pool is not None else {})

    def sha256(self, path: Path) -> str:
        future = self._futures.pop(path, None)
        return future.result() if future is not None else sha256_file(path)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
        return False


def _output_payloads(loaded) -> list:
    """Each part's output shards, in the order the merge checks them.

    A part whose index cannot name them contributes nothing here; the merge
    refuses it at its own check.
    """
    payloads = []
    for _rank, path, _part, _manifest, _config, index in loaded:
        try:
            payloads.extend([path / _leaf(name) for name in sorted(set(index["weight_map"].values()))])
        except Exception:
            continue
    return payloads


def make_artifact_readable(path: Path) -> None:
    """Give one newly produced shard the serving handoff's read permissions.

    Add no write or execute bits. Call only on the output being published,
    never on a source checkpoint or a copy-mode source part.
    """
    path.chmod(stat.S_IMODE(path.stat().st_mode)
               | stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _leaf(name: str) -> str:
    if not isinstance(name, str) or Path(name).name != name or name in ("", ".", ".."):
        raise ValueError(f"shard must be a local filename: {name!r}")
    return name


def tensor_names(path: Path) -> set[str]:
    """Read the safetensors header without materializing its tensor payload."""
    with path.open("rb") as handle:
        length = struct.unpack("<Q", handle.read(8))[0]
        if length > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header length: {path}")
        header = json.loads(handle.read(length))
    return set(header) - {"__metadata__"}


def source_identity(source: Path, *, workers=None, digest_cache=None) -> dict:
    """Exact whole-source identity, with the same fenced shard reuse as parts.

    Config, auxiliaries and complete header coverage are always read. Only
    shard payload hashes use the optional owning SourceDigestCache; the caller
    must retain its receipt. Parallel hashing preserves path/error order.
    """
    source = Path(source)
    index_path = source / "model.safetensors.index.json"
    if index_path.exists():
        tensors = json.loads(index_path.read_text())["weight_map"]
        files = sorted({_leaf(name) for name in tensors.values()})
        actual = {}
        for name in files:
            for tensor in tensor_names(source / name):
                if tensor in actual:
                    raise ValueError(f"source tensor appears twice: {tensor}")
                actual[tensor] = name
        if actual != tensors:
            raise ValueError("source index does not exactly cover the source tensor files")
    else:
        files = sorted(path.name for path in source.glob("*.safetensors"))
        tensors = {}
        for name in files:
            for tensor in tensor_names(source / name):
                if tensor in tensors:
                    raise ValueError(f"source tensor appears twice: {tensor}")
                tensors[tensor] = name
    if not tensors:
        raise ValueError("source has no safetensors tensors")
    return {"config_sha256": sha256_file(source / "config.json"),
            "auxiliary_sha256": _auxiliary_sha256(source),
            "files": dict(zip(files, sha256_files([source / name for name in files], workers,
                digest_cache.sha256 if digest_cache is not None else None))),
            "tensors": tensors}


def _auxiliary_sha256(source: Path) -> dict:
    """The small non-tensor files every source identity binds, by name."""
    auxiliary = sorted({p for pattern in ("*.json", "*.txt", "*.jinja", "*.model")
                        for p in source.glob(pattern)})
    return {p.name: sha256_file(p) for p in auxiliary}


#: The :func:`source_identity` fields that bind a checkpoint's configuration
#: and tensor roster without reading a shard payload (tessera#523).
SOURCE_ROSTER_FIELDS = ("config_sha256", "auxiliary_sha256", "tensors")


def source_roster_identity(source: Path) -> dict:
    """:func:`source_identity` restricted to :data:`SOURCE_ROSTER_FIELDS`.

    Reads the safetensors headers, ``config.json`` and the auxiliary files,
    never the tensor payloads, so it costs seconds where the whole identity
    reads the entire checkpoint.  Use it where every decision depends only on
    names, shapes and configuration -- the planner's carried-projection check
    -- and leave byte binding to the checks that read the bytes: the
    partition stamps and their merge, and the cached-unit intake.  A source
    with no ``config.json`` records ``config_sha256: None``, which never
    equals a whole identity's digest.
    """
    source = Path(source)
    tensors = source_inventory(source)
    config_path = source / "config.json"
    return {"config_sha256": sha256_file(config_path) if config_path.exists() else None,
            "auxiliary_sha256": _auxiliary_sha256(source),
            "tensors": tensors}


def exporter_code_root() -> Path:
    """The tree :func:`export_identity` digests for this running exporter.

    A checkout's root (holding ``src/`` and ``experiments/``) when the
    module resolves from a checkout; the directory holding the ``tessera``
    package (a checkout's ``src/``, or ``site-packages``) when it resolves
    from an install -- the same derivation as
    ``tessera.serving.source_identity._default_root`` (#691 item 4).
    Anything else is refused with its location, not hashed as whatever
    two parents up happens to be.
    """
    holder = Path(__file__).resolve().parents[1]
    checkout = holder.parent
    if holder.name == "src" and (checkout / "experiments").is_dir():
        return checkout
    if (holder / "tessera" / "serving").is_dir():
        return holder
    raise SystemExit(
        f"cannot locate the Tessera code root from {__file__}: {holder} "
        "is neither a checkout src/ with an experiments/ sibling nor a "
        "directory holding an installed tessera package.")


def _installed_commit_id() -> "str | None":
    """The commit the running Tessera was installed from, if recorded.

    A non-editable pip install from git records ``direct_url.json`` with the
    commit in its dist-info; an editable install is a checkout, so git
    answers before this is ever asked.  The search is restricted to the
    directory holding the imported ``tessera`` package: a global scan would
    happily return some OTHER environment's install of Tessera, which is
    the same provenance hole stamped as a value (#691 item 4).
    """
    holder = Path(__file__).resolve().parents[1]
    try:
        from importlib.metadata import distributions
    except ImportError:
        return None
    for dist in distributions(path=[str(holder)]):
        try:
            text = dist.read_text("direct_url.json")
        except Exception:
            continue
        if not text:
            continue
        try:
            info = json.loads(text)
        except ValueError:
            continue
        commit = (info.get("vcs_info") or {}).get("commit_id")
        if commit:
            return commit
    return None


def git_hash() -> str:
    """The commit this build came from -- git, the environment, or the install.

    A build that runs on a synced copy of the tree has no ``.git`` and used to
    stamp ``unknown``, which is a provenance hole in an artifact whose whole
    claim is that the surrogate, the KL and the bytes are one rendering.
    ``TESSERA_GIT`` is how the caller supplies it when git cannot; a
    non-editable install from git stamps its ``direct_url.json`` commit
    (#691 item 4).  When none of the three answers, this refuses instead of
    stamping ``unknown``.
    """
    import os

    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True).strip()
    except Exception:
        pass
    caller = os.environ.get("TESSERA_GIT")
    if caller:
        return caller
    commit = _installed_commit_id()
    if commit:
        return commit
    raise SystemExit(
        "cannot stamp the Tessera commit: no git checkout above the running "
        "module, no TESSERA_GIT in the environment, and the installed Tessera "
        "records no commit in its direct_url.json (not a pip install from "
        "git). Set TESSERA_GIT to the commit this code came from.")


def export_identity(source: Path, options: dict, runtime_image: str, root: Path,
                    shards=None, *, digest_cache=None) -> dict:
    """What every serving part of one export must agree on, plus its own input.

    ``source`` is :func:`source_part_identity` over ``shards`` -- the shards
    this part reads encoded tensors from, every shard when ``None`` -- so a
    part costs one pass over its own input rather than over the whole
    checkpoint (tessera#495). :func:`merge_serving_parts` takes the whole
    source once and proves every part's stamp against that pass.
    """
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", runtime_image or ""):
        raise ValueError("partition runtime image must be an exact repository@sha256 digest")
    # ``root`` is a checkout root (holding ``src/`` and ``experiments/``)
    # or an installed holder (holding the ``tessera`` package, #691 item
    # 4): the digest covers the code the exporter runs from either way, and
    # anything else is refused with its location.
    if (root / "experiments").is_dir() and (root / "src").is_dir():
        paths = sorted([*(p for p in root.joinpath("src").rglob("*")
                           if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}),
                        *root.joinpath("experiments").glob("*.py"),
                        root / "src/tessera/serving/runtime_contract.json"])
    elif (root / "tessera" / "serving").is_dir():
        package = root / "tessera"
        paths = sorted([*(p for p in package.rglob("*")
                           if p.suffix in {".py", ".cu", ".cuh", ".cpp", ".h"}
                           and p.is_file() and "__pycache__" not in p.parts),
                        package / "serving" / "runtime_contract.json"])
    else:
        raise FileNotFoundError(
            f"{root} is neither a Tessera checkout (src/ + experiments/) "
            "nor a directory holding an installed tessera package")
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode() + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"source": source_part_identity(source, shards, digest_cache=digest_cache),
            "code_sha256": digest.hexdigest(),
            # The dispatch command pins this image as the READER/runtime the
            # parts are destined to serve on. It is never a claim about the
            # process that WROTE the part: a selected host producer's own
            # environment is its authenticated receipt
            # (tessera.export_serving.authenticate_producer_python, sealed in
            # identity["producer"]), and that producer does not execute this
            # image. The image's observed runtime identity belongs to the
            # PrismaBuild receipt, not a self-attestation here.
            "runtime_image": runtime_image, "options": options}


def dense_resident_bytes_resident_mode(family: str, rows: int, cols: int,
                                     *, trellis_table_bytes: int = 0,
                                     native_roles=None, decoder: str = "native") -> int:
    """Persistent tensors of the selected dense decoder, including row scales.

    Native window roles carry their verified wire rates/window width and the
    runtime tile geometry. Count each fused role separately: padding, tables,
    permutations and run tables are owned per role. The route retains a second
    concatenated fp32 row-scale buffer beside the frozen bundles.

    The old expanded tensor formula is available only for an explicitly named
    reference decoder; missing native layout must never silently price it.
    """
    if family == "TESSERA_NVFP4":
        if decoder == "torch_window":
            return rows * cols // 2 + rows * cols // 16 + 4 + trellis_table_bytes
        if decoder != "native" or not native_roles:
            raise ValueError("native A4 resident accounting requires per-role layout")
        total, role_rows = 4 + trellis_table_bytes, 0
        for role in native_roles:
            count, width = int(role["rows"]), int(role["cols"])
            rates = tuple(int(rate) for rate in role["rates"])
            arity, memory, half = (int(role[key]) for key in ("arity", "memory", "half"))
            if (arity != 2 or half != 16 or count <= 0 or count % (2 * arity * 8)
                    or width != cols or width % half or len(rates) != width
                    or len(set(rates)) != 1 or not 1 <= rates[0] <= 8 or memory < 1):
                raise ValueError("invalid native A4 role layout")
            steps, pairs = count // arity, count // (2 * arity)
            points = 1 << (rates[0] - 1)
            select = width * (pairs // 8 + 1) + 8
            label = width * (pairs // 4)
            point = width * (steps * (rates[0] - 1) // 8)
            nibbles = count * width // (half * 2)
            lut = int(role["lut_entries"]) if len(native_roles) > 1 else 16
            if not 1 <= lut <= 16:
                raise ValueError("invalid native A4 scale table")
            tables = lut + (1 << (memory + 1)) * 4 + 4 * points * (arity + 1)
            total += select + label + point + nibbles + tables + 4  # per-role epilogue
            role_rows += count
        if role_rows != rows:
            raise ValueError("native A4 role rows do not cover the fused module")
        return total
    if family not in ("TESSERA_BF16", "TESSERA_FP8"):
        raise ValueError(f"no resident-mode accounting for family {family!r}")
    if decoder == "torch_window":
        return rows * cols * (2 if family == "TESSERA_BF16" else 1) + rows * 4
    if decoder != "native":
        raise ValueError(f"unknown dense resident decoder {decoder!r}")
    if not native_roles:
        raise ValueError("native dense resident accounting requires per-role layout")
    total, role_rows = rows * 4, 0
    for role in native_roles:
        count, width = int(role["rows"]), int(role["cols"])
        rates = tuple(int(rate) for rate in role["rates"])
        bits, tile = int(role["window_bits"]), int(role["tile_rows"])
        if count <= 0 or width != cols or len(rates) != width or tile <= 0 or tile % 8:
            raise ValueError("invalid native dense role geometry")
        if bits <= 0 or any(rate < 1 or rate > min(bits, NATIVE_DENSE_WINDOW_RATE_MAX)
                            for rate in rates):
            raise ValueError("invalid native dense window layout")
        padded = -(-count // tile) * tile
        words = padded * sum(rates) // 8
        tables = (1 << bits) * (2 if family == "TESSERA_BF16" else 1)
        if family == "TESSERA_FP8":
            tables += 256
        total += words + tables + count * 4 + len(set(rates)) * 16 + width * 8
        role_rows += count
    if role_rows != rows:
        raise ValueError("native dense role rows do not cover the fused module")
    return total


def routed_window_unit_resident_bytes(family: str, rows: int, cols: int, rates,
                                      *, window_bits: int, tile_rows: int) -> int:
    """One expert projection's slot in the compact routed lane (tessera#624).

    The runtime's compact window lane (``serving.moe_route`` with
    ``compact_window_lane`` true, every TESSERA_FP8/TESSERA_BF16 stack) never
    allocates a decoded tile: ``native_window_moe.WindowUnitAxis._alloc``
    holds, per part and per expert, the repacked BODY words (rows padded to
    the 512-row tile, ``sum(rates)`` bits per padded row), one ``[rate, col0,
    n, word0]`` int32 run per distinct rate, the fp32 row scales, the int32
    column permutation and initial-state register (``cols`` each), the
    family's state table (bf16 ``2^L`` values for the value family; uint8
    ``2^L`` codes plus the 256-byte native grid for e4m3), and four int32
    per-expert scalars (``tile_words``, ``total_words``, ``has_init`` and
    ``finish``'s ``word_off``).  This prices exactly those tensors from the
    unit's own verified manifest (rates, window bits, geometry); the
    per-part ``run_off`` is :func:`routed_window_part_resident_bytes` and
    the fused lane's tables :func:`routed_fused_unit_bytes`.

    Whole unit at TP1: a tensor-parallel cut prices the rank-local rows
    (w13) or columns and their rate slice (w2) through the same function.
    """
    if family not in ("TESSERA_BF16", "TESSERA_FP8"):
        raise ValueError(f"no compact routed accounting for family {family!r}")
    rows, cols, bits, tile = int(rows), int(cols), int(window_bits), int(tile_rows)
    rates = tuple(int(rate) for rate in rates)
    if rows <= 0 or cols <= 0 or len(rates) != cols or tile <= 0 or tile % 8:
        raise ValueError("invalid compact routed unit geometry")
    if bits <= 0 or any(rate < 1 or rate > 8 for rate in rates):
        raise ValueError("invalid compact routed window layout")
    padded = -(-rows // tile) * tile
    words = padded * sum(rates) // 8
    tables = (1 << bits) * 2 if family == "TESSERA_BF16" else (1 << bits) + 256
    runs = len(set(rates)) * 16
    per_expert_scalars = 4 * 4  # tile_words, total_words, has_init, word_off (int32)
    return words + tables + rows * 4 + runs + cols * 8 + per_expert_scalars


def routed_window_part_resident_bytes(experts: int) -> int:
    """The per-part ``run_off`` the axis adds at ``finish``: ``[E + 1]``
    int64, since ``torch.cumsum`` promotes the int32 run counts it sums
    (``WindowUnitAxis.finish`` and ``prepare_grouped_window_gemm`` alike)."""
    if int(experts) <= 0:
        raise ValueError("a routed stack needs at least one expert")
    return 8 * (int(experts) + 1)


def routed_fused_table_bytes(window_bits: int) -> int:
    """One composed 16-bit lookup table the fused routed lane (tessera#685)
    holds per expert projection beside the bundle's planes, as the runtime's
    ``FusedRoutedWindowMoE.resident_bytes`` publishes it (both families; the
    value family's table is a view the self-report still counts)."""
    return 2 * (1 << int(window_bits))


#: The dense loader's column-rate bound (tessera#750 item 4: every rate a
#: 14-bit window holds), restated from ``compact_prep.DENSE_WINDOW_RATE_MAX``
#: so this module stays torch-free (``tests/test_export_routed_resident_
#: pricing.py`` pins them equal).  The routed stacks' bound stays 8.
NATIVE_DENSE_WINDOW_RATE_MAX = 14


#: The fused routed lane's per-projection launch tables since contract v45
#: (tessera#694), restated from ``tessera.routed_fused`` so this module stays
#: torch-free (``tests/test_export_routed_resident_pricing.py`` pins them
#: equal): the run pair is int32 ``[8]`` per expert, and each ``BK``-column
#: block carries an int32 ``[BDESC_INTS]`` descriptor.
ROUTED_FUSED_RUN_PAIR_INTS = 8
ROUTED_FUSED_BLOCK_COLS = 32
ROUTED_FUSED_BDESC_INTS = 12


def routed_fused_unit_bytes(window_bits: int, cols: int) -> int:
    """What the fused routed lane holds per expert projection beside the
    bundle's planes, as ``FusedRoutedWindowMoE.resident_bytes`` publishes it:
    the composed table (:func:`routed_fused_table_bytes`, tessera#685) and,
    since contract v45 (tessera#694), the projection's run pair and its
    block descriptors (``routed_fused.projection_tables``).  ``cols`` is the
    unit's rank-local column count, which sets the descriptor count."""
    cols = int(cols)
    if cols <= 0 or cols % ROUTED_FUSED_BLOCK_COLS:
        raise ValueError(f"the fused routed lane reads whole {ROUTED_FUSED_BLOCK_COLS}-column "
                         f"blocks; {cols} columns are not")
    return (routed_fused_table_bytes(window_bits) + 4 * ROUTED_FUSED_RUN_PAIR_INTS
            + 4 * ROUTED_FUSED_BDESC_INTS * (cols // ROUTED_FUSED_BLOCK_COLS))


def vocab_parallel_rows(rows: int, tp_size: int, *, padding: int = 64) -> int:
    """Rows one rank holds of a vocab-parallel table: the runtime pads the
    vocabulary to a multiple of ``padding`` before cutting it ``tp_size`` ways."""
    rows, tp_size, padding = int(rows), int(tp_size), int(padding)
    if rows <= 0 or tp_size <= 0 or padding <= 0:
        raise ValueError("invalid vocab-parallel geometry")
    padded = -(-rows // padding) * padding
    return -(-padded // tp_size)


def mtp_draft_embed_head_duplicate_bytes(vocab_tensors, tp_size: int) -> int:
    """Conservative legacy MTP ``embed_tokens`` + ``lm_head`` allowance (#645).

    The legacy stock draft allocates a vocab-parallel embedding and head before
    the loader points them at the target's. The source-guarded Tessera draft
    lifetime hook avoids these allocations; this fit allowance stays unchanged
    until full-model/image qualification, not inferred from CPU stand-ins.  ``vocab_tensors`` is
    ``[(rows, cols, element_bytes), ...]`` for the target's passthrough
    embedding and head tensors; the duplicate is their rank-local cut.
    """
    total = 0
    for rows, cols, element_bytes in vocab_tensors:
        total += vocab_parallel_rows(rows, tp_size) * int(cols) * int(element_bytes)
    return total


def per_rank_fit_items(*, tp_size: int, routed_bytes_by_rank, mtp_duplicate_bytes: int,
                       mtp_layers: int) -> dict:
    """The per-rank fit block: named line items and their sum, one row per rank.

    Every item is one named line whose sum is ``total_bytes``; a reader adding
    a budget adds an item rather than folding it into another.  Only the
    compact routed stacks and the MTP draft duplicate are priced here; dense
    modules and passthrough tensors are not, and the note says so.
    """
    tp_size = int(tp_size)
    ranks = []
    for rank, routed in enumerate(routed_bytes_by_rank):
        items = {"routed_moe_resident_mode_bytes": int(routed),
                 "mtp_draft_embed_head_duplicate_bytes": int(mtp_duplicate_bytes)}
        ranks.append({"rank": rank, "items": items, "total_bytes": sum(items.values())})
    if len(ranks) != tp_size:
        raise ValueError(f"{len(ranks)} routed rank figures for tp_size {tp_size}")
    return {
        "tp_size": tp_size,
        "ranks": ranks,
        "mtp_draft_layers": int(mtp_layers),
        "note": ("routed_moe_resident_mode_bytes is each rank's cut of every routed stack "
                 "(packed planes, tables, per-expert bookkeeping and the fused lane's "
                 "composed tables where its shape admits the stack); "
                 "mtp_draft_embed_head_duplicate_bytes is the MTP draft's own rank-local "
                 "embed_tokens + lm_head (0 when the config declares no draft layers). "
                 "Dense modules and passthrough tensors are not priced per rank here."),
    }


def shared_candidate_pricing(modules: dict, module_groups: dict) -> dict:
    """Opt-in additive resident composition; preserve every legacy unit price.

    ``module_groups`` comes from the parsed wire's exact replay table specs,
    not from module geometry or an assumed default trellis.
    """
    from .shared_candidate import build_pricing
    return build_pricing(modules, module_groups)


def summarize_modules(modules: dict, passthrough_bytes: int, checkpoint_bytes: int) -> dict:
    roles = [role for module in modules.values() for role in module["roles"]]
    params = sum(r["rows"] * r["cols"] for r in roles)
    wire = sum(m["wire_bytes"] for m in modules.values())
    containers = sum(m["container_bytes"] for m in modules.values())
    resident = sum(m["resident_bytes_resident_mode"] for m in modules.values())
    families = {}
    for family in sorted({m["family"] for m in modules.values()}):
        members = [m for m in modules.values() if m["family"] == family]
        family_roles = [r for m in members for r in m["roles"]]
        p = sum(r["rows"] * r["cols"] for r in family_roles)
        w = sum(m["wire_bytes"] for m in members)
        r = sum(m["resident_bytes_resident_mode"] for m in members)
        families[family] = {"modules": len(members), "units": len(family_roles),
                            "quantized_params": p, "wire_bytes": w,
                            "wire_bpp": w * 8 / p, "resident_mode_bytes": r,
                            "resident_mode_bpp": r * 8 / p}
    return {"quantized_params": params, "modules": len(modules), "units": len(roles),
            "wire_bytes": wire, "wire_bpp": wire * 8 / params if params else None,
            "on_disk_bytes": containers, "on_disk_bpp": containers * 8 / params if params else None,
            "resident_mode_bytes": resident,
            "resident_mode_bpp": resident * 8 / params if params else None,
            "streamed_mode_note": "the prepared planes (~wire bytes + per-unit tables) plus one transient decoded tile per forward",
            "by_family": families, "passthrough_bytes": passthrough_bytes,
            "checkpoint_bytes": checkpoint_bytes}


def _expected_outputs(owned: set[str], modules: dict) -> set[str]:
    consumed, outputs = set(), set()
    for name, module in modules.items():
        for role in module["roles"]:
            consumed.add(role.get("source_tensor", role["tensor"]))
        if module.get("structure") == "routed_moe":
            for role in module["roles"]:
                stem = role["tensor"].removesuffix(".weight")
                outputs.add(stem + ".wire")
                # The NVFP4 routed builder reads one A-side scale per expert
                # projection, ``experts.{e}.{proj}.input_global_scale`` -- the
                # same quantity the dense route reads as
                # ``trellis_input_global_scale`` (``nvfp4_moe_route``
                # :107-109, :328-331), written beside each wire by
                # ``tessera.export_serving`` (:2413-2429).  The role declares
                # the scale exactly when the export wrote one, so expect it
                # from the declaration rather than from the family: a role
                # that declares a scale and wrote none is as broken as a
                # missing wire, and this set is what proves it.
                if "input_global_scale" in role:
                    outputs.add(stem + ".input_global_scale")
        else:
            outputs.add(name + ".wire_bytes")
            if module["family"] == "TESSERA_NVFP4":
                outputs.add(name + ".trellis_input_global_scale")
    if not consumed <= owned:
        raise ValueError(f"module consumes source tensors outside its partition: {sorted(consumed - owned)[:3]}")
    return (owned - consumed) | outputs


def validate_explicit_plan(plan, modules: dict, config_groups: dict, *, source_tensors=None) -> None:
    """An explicit rung is an obligation, including a whole expert population.

    Implicit/default dense planning has no complete plan roster to compare.
    Unplanned expert stacks, in contrast, always pass through in the exporter,
    so the explicit stack set must equal both emitted and declared stack sets.

    An explicit ``"PASSTHROUGH"``/``"BF16"`` entry is an obligation too -- a
    NEGATIVE one, and it used to be the single plan statement nothing checked:
    the loop below skipped those entries, so a quantized wire emitted for a
    tensor the published plan says was passed through was accepted by this
    gate (#301).  A plan is complete in both directions or it is prose.
    """
    if plan is None:
        return
    if not isinstance(plan, dict):
        raise ValueError("explicit export plan must be an object")
    # The entry SHAPE is the published schema's one implementation (#691
    # item 3): this gate used to repeat a weaker check ("grid" and "q256"
    # present), so a sidecar field the schema accepts was refused downstream
    # of the argument-time gate that had accepted it.  Imported lazily: this
    # module sits under the validator in some import orders. Relative, so a
    # sealed historical copy of this file stays importable: an absolute
    # ``tessera.*`` import reads as an escape into the current producer
    # (historical_producer._SealedLoader) and refuses the whole package.
    from .serving_plan import SCHEMA_KEY, validate_serving_plan
    try:
        validate_serving_plan(plan)
    except ValueError as exc:
        raise ValueError(f"explicit export plan {exc}") from exc
    requested, passthrough = {}, set()
    for name, spec in plan.items():
        if name == SCHEMA_KEY:
            continue
        if spec in ("PASSTHROUGH", "BF16"):
            passthrough.add(name)
            continue
        requested[name] = spec
    planned_stacks = {name for name in requested if name.endswith(".experts")}
    emitted_stacks = {name for name, module in modules.items()
                      if module.get("structure") == "routed_moe"}
    all_roles = [role for module in modules.values() for role in module.get("roles", ())]
    # THE NEGATIVE OBLIGATIONS, before the positive ones: a contradiction here
    # names the tensor and the entry that was contradicted, which the stack
    # coverage difference below cannot.
    for name in sorted(passthrough):
        spelling = plan[name]
        if name in emitted_stacks:
            raise ValueError(
                f"explicit plan stack {name}: planned {spelling} but a routed_moe "
                "module was emitted for it")
        emitted = [role for role in all_roles
                   if name in (role.get("tensor"), role.get("source_tensor"))]
        if emitted:
            raise ValueError(
                f"explicit plan tensor {name}: planned {spelling} but {len(emitted)} "
                f"quantized role(s) were emitted for it, e.g. {emitted[0].get('grid')} "
                f"q256={emitted[0].get('q256')}")
    stack_schemes = {}
    for group in config_groups.values():
        scheme = group.get("scheme", {})
        if scheme.get("structure") != "routed_moe":
            continue
        for name in group.get("targets", ()):
            if name in stack_schemes:
                raise ValueError(f"explicit plan stack {name}: declared twice")
            stack_schemes[name] = scheme
    if planned_stacks != emitted_stacks or planned_stacks != set(stack_schemes):
        raise ValueError(
            "explicit plan stack coverage differs: "
            f"missing emitted={sorted(planned_stacks - emitted_stacks)}, "
            f"extra emitted={sorted(emitted_stacks - planned_stacks)}, "
            f"missing declared={sorted(planned_stacks - set(stack_schemes))}, "
            f"extra declared={sorted(set(stack_schemes) - planned_stacks)}")
    from .serving.scheme import MOE_GROUP_PROJECTIONS, validate_tessera_moe_scheme

    for name, spec in requested.items():
        wanted_grid, wanted_rung = spec["grid"], int(spec["q256"])
        if wanted_grid == "E2M1x1":  # tuple_grid's arity-one spelling
            wanted_grid = "E2M1"
        if name in planned_stacks:
            record, scheme = modules[name], stack_schemes[name]
            roles = record.get("roles", ())
            declared = validate_tessera_moe_scheme(scheme, f"explicit plan {name}")
            experts = declared["experts"]
            expected_roles = {(expert, projection) for expert in range(experts)
                              for projections in MOE_GROUP_PROJECTIONS.values()
                              for projection in projections}
            got_roles = [(r.get("expert"), r.get("role")) for r in roles]
            if len(got_roles) != len(expected_roles) or set(got_roles) != expected_roles:
                raise ValueError(f"explicit plan stack {name}: emitted expert/projection coverage differs")
            if record.get("experts") != experts:
                raise ValueError(f"explicit plan stack {name}: manifest expert count differs from config")
            if source_tensors is not None:
                expected_sources = {n for n in source_tensors if n.startswith(name + ".")}
                consumed = {r.get("source_tensor", r["tensor"]) for r in roles}
                if consumed != expected_sources:
                    raise ValueError(f"explicit plan stack {name}: source projection coverage differs")
            if record.get("grid") != wanted_grid or record.get("q256") != wanted_rung:
                raise ValueError(f"explicit plan stack {name}: manifest grid/rung differs from plan")
            if scheme.get("grid") != wanted_grid or any(
                    any(rung != wanted_rung for rung in group["role_q256"])
                    for group in declared["groups"].values()):
                raise ValueError(f"explicit plan stack {name}: declared grid/rung differs from plan")
        else:
            roles = [r for r in all_roles if r.get("tensor") == name]
            if not roles:
                raise ValueError(f"explicit plan tensor {name}: expected one emitted role, got 0")
            if len(roles) != 1:
                # tessera#377: a MergedColumnParallelLinear built from ONE source
                # tensor is emitted as one role per attested output partition,
                # each a row window of that tensor.  Several roles for one
                # tensor are accepted only when the windows tile it exactly.
                windows = sorted((int(r.get("row_offset", 0)), int(r.get("rows", 0))) for r in roles)
                totals = {int(r.get("source_rows", 0)) for r in roles}
                covered, tiled = 0, len(totals) == 1
                for offset, rows in windows:
                    tiled = tiled and offset == covered and rows > 0
                    covered += rows
                if not tiled or covered != next(iter(totals)):
                    raise ValueError(
                        f"explicit plan tensor {name}: {len(roles)} emitted roles do not tile "
                        f"the tensor by row (windows {windows}, source rows {sorted(totals)})")
        for role in roles:
            if role.get("grid") != wanted_grid or role.get("q256") != wanted_rung:
                raise ValueError(f"explicit plan {name}: emitted role grid/rung differs from plan")

def _require_part_carriers(part, manifest):
    rank, count = part.get("index"), part.get("count")
    if type(rank) is not int or type(count) is not int or count < 1 or not 0 <= rank < count:
        raise ValueError("partition index/count must be integer carriers with 0 <= INDEX < COUNT")
    identity = part["identity"]
    if "producer" in manifest or "producer" in identity:
        if not isinstance(identity.get("producer"), dict) or manifest.get("producer") != identity["producer"]:
            raise ValueError(f"partition {rank}: producer disagrees with sealed export identity")
    options = identity["options"]
    if "encode_batch" in manifest or "encode_batch" in options:
        batch = manifest.get("encode_batch")
        sealed = options.get("encode_batch")
        if type(batch) is not int or type(sealed) is not int or batch < 1 or batch != sealed:
            raise ValueError(f"partition {rank}: encode_batch disagrees with integer sealed export identity")


def validate_serving_part(path, source, *, partition=None, producer=None, encode_batch=None,
                          source_digest_cache=None):
    """Prove one completed part with the same contents owner as assembly.

    Source inventory/config/auxiliary coverage is whole; only the owned input
    shards need body hashes. A reused digest is exposed in source_proof.
    """
    path, source = Path(path), Path(source)
    manifest = read_serving_manifest(path / "tessera_serving_manifest.json")
    part = manifest.get("export_partition") or {}
    if part.get("schema") != SCHEMA:
        raise ValueError(f"{path}: unsupported serving partition schema")
    _require_part_carriers(part, manifest)
    rank, count = part["index"], part["count"]
    if partition is not None and (rank, count) != partition:
        raise ValueError("manifest membership differs from requested partition")
    if producer is not None and manifest.get("producer") != producer:
        raise ValueError("manifest producer receipt differs from fresh authentication")
    if encode_batch is not None and manifest.get("encode_batch") != encode_batch:
        raise ValueError("manifest encode_batch differs from requested batch")
    stamped_files = part["identity"]["source"]["files"]
    whole = source_part_identity(source, stamped_files, digest_cache=source_digest_cache)
    prove_source_part(part["identity"]["source"], whole, f"partition {rank}")
    index = read_serving_manifest(path / "model.safetensors.index.json")
    _prove_part_contents(rank, count, path, part, manifest, index, whole, sha256_file)
    config = read_serving_manifest(path / "tessera_part_config.json")
    source_config = read_serving_manifest(source / "config.json")
    source_config.pop("quantization_config", None)
    if {k: v for k, v in config.items() if k != "quantization_config"} != source_config:
        raise ValueError(f"partition {rank}: model config disagrees with source identity")
    qconfig = config["quantization_config"]
    declarations = {target for group in qconfig["config_groups"].values() for target in group["targets"]}
    if qconfig["quant_method"] != "tessera" or declarations != set(manifest["modules"]):
        raise ValueError(f"partition {rank}: config targets disagree with manifest modules")
    return {"manifest_sha256": sha256_file(path / "tessera_serving_manifest.json"),
            "source_proof": source_digest_cache.receipt() if source_digest_cache is not None else None}


def consumed_input_content(path, plan, hessian, authority, scales=None):
    """Current input snapshots must equal the identities consumed by this part."""
    manifest = read_serving_manifest(Path(path) / "tessera_serving_manifest.json")
    options = manifest["export_partition"]["identity"]["options"]
    plan_bytes = Path(plan).read_bytes()
    plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
    if manifest.get("plan_sha256") != plan_sha256 or options.get("plan_sha256") != plan_sha256:
        raise ValueError("consumed.plan_sha256: current bytes differ from the manifest's consumed plan snapshot")
    if json.loads(plan_bytes, object_pairs_hook=unique_json_pairs) != options.get("plan"):
        raise ValueError("consumed.plan: current allocation differs from the manifest's consumed plan")
    current = {"plan_sha256": plan_sha256,
               "hessian_sha256": sha256_file(Path(hessian)),
               "authority_sha256": sha256_file(Path(authority)),
               "input_scales_sha256": sha256_file(Path(scales)) if scales else None}
    for field, sealed in (("hessian_sha256", "hessian_sha256"),
                          ("authority_sha256", "producer_authority_sha256"),
                          ("input_scales_sha256", "input_scales_sha256")):
        if sealed not in options or current[field] != options[sealed]:
            raise ValueError(f"consumed.{field}: current bytes differ from the manifest's consumed identity")
    return current


def _prove_part_contents(rank, count, path, part, manifest, index, whole, hash_file):
    """The single-part ownership and written-payload proof used by assembly."""
    _require_part_carriers(part, manifest)
    owned = set(part["source_tensors"])
    expected = {n for n in whole["tensors"] if partition_owner(n, count) == rank}
    if owned != expected or len(owned) != len(part["source_tensors"]):
        raise ValueError(f"partition {rank}: source tensor coverage disagrees with ownership")
    read = {whole["tensors"][name] for name in owned}
    if read != set(part["identity"]["source"]["files"]):
        raise ValueError(f"partition {rank}: source stamp coverage disagrees with the shards "
                         f"its source tensors live in: stamped "
                         f"{sorted(part['identity']['source']['files'])[:5]}, read {sorted(read)[:5]}")
    local_map = index["weight_map"]
    if set(local_map) != _expected_outputs(owned, manifest["modules"]):
        raise ValueError(f"partition {rank}: written tensor coverage disagrees with source and encoded modules")
    files = set(local_map.values())
    if files != set(part["output_sha256"]):
        raise ValueError(f"partition {rank}: output sha256 coverage disagrees with index")

    actual, payloads = {}, []
    for filename in sorted(files):
        filename = _leaf(filename)
        payload = path / filename
        if hash_file(payload) != part["output_sha256"][filename]:
            raise ValueError(f"partition {rank}: output sha256 mismatch: {filename}")
        for name in tensor_names(payload):
            if name in actual:
                raise ValueError(f"tensor appears in two files: {name}")
            actual[name] = filename
        payloads.append((payload, filename))
    if files != {p.name for p in path.glob("*.safetensors")} or (owned and not files):
        raise ValueError(f"partition {rank}: output shard coverage disagrees with directory")
    if actual != local_map:
        raise ValueError(f"partition {rank}: index disagrees with actual tensor headers")
    return owned, local_map, payloads



def _merged_batch_widths(loaded) -> dict | None:
    """The whole run's joined-encode width histogram, or None when any part lacks one."""
    if not all("encode_batch_observed" in row[3] for row in loaded):
        return None
    widths = {}
    for row in loaded:
        histogram = row[3]["encode_batch_observed"]
        if type(histogram) is not dict:
            raise ValueError(
                f"partition {row[0]}: encode_batch_observed must be an object of width "
                f"counts, got {type(histogram).__name__}")
        limit = row[3].get("encode_batch")
        for width, calls in histogram.items():
            # A width is a canonical decimal from 1 up to the part's own encode_batch:
            # "08" would read as 8 and become a second entry for it, and no joined call
            # is wider than the knob that bounds it.
            if (type(width) is not str or not re.fullmatch(r"[1-9][0-9]*", width)
                    or type(limit) is not int or int(width) > limit):
                raise ValueError(
                    f"partition {row[0]}: encode_batch_observed[{width!r}] is not a width "
                    f"from 1 up to the part's encode_batch ({limit!r})")
            # Exactly an integer: a boolean would sum as 0 or 1 and a fraction would
            # stay one, claiming a count the part's own input does not establish.
            if type(calls) is not int or calls < 0:
                raise ValueError(
                    f"partition {row[0]}: encode_batch_observed[{width!r}] must be a "
                    f"non-negative integer count, got {calls!r}")
            widths[width] = widths.get(width, 0) + calls
    return dict(sorted(widths.items(), key=lambda item: int(item[0])))


def merge_serving_parts(paths, out: Path, source: Path, *, move=False,
                        source_digest_cache=None) -> dict:
    """Prove identities, ownership and written tensor coverage before publishing.

    ``source_digest_cache`` (a :class:`tessera.source_digest_cache.SourceDigestCache`)
    lets the whole-source pass reuse stat-bound shard digests instead of
    re-reading every shard; the merged manifest then records ``source_proof``
    with the receipt, whose ``mode`` is ``stat-bound`` when any digest was
    reused. Without it the pass hashes every shard and no ``source_proof`` is
    written. Output shards are always hashed.
    """
    from .moe_execution import ResearchSelectedMoeConfig, ResearchSelectedMoeInput

    out, source = Path(out), Path(source)
    if out.exists():
        raise ValueError(f"merge output already exists: {out}")
    loaded = []
    for path in map(Path, paths):
        manifest = read_serving_manifest(path / "tessera_serving_manifest.json")
        part = manifest.get("export_partition", {})
        if part.get("schema") != SCHEMA:
            raise ValueError(f"{path}: unsupported serving partition schema")
        config = json.loads((path / "tessera_part_config.json").read_text())
        index = json.loads((path / "model.safetensors.index.json").read_text())
        loaded.append((part["index"], path, part, manifest, config, index))
    if not loaded:
        raise ValueError("no serving partitions supplied")
    loaded.sort(key=lambda row: row[0])
    count = loaded[0][2]["count"]
    if (type(count) is not int or count < 1 or
            [r[0] for r in loaded] != list(range(count)) or
            any(r[2]["count"] != count for r in loaded)):
        raise ValueError("partition coverage must contain each INDEX in 0..COUNT-1 exactly once")
    # Each part stamps only the shards it read, so ``source`` differs between
    # honest parts; it is proved against --source below, and everything else
    # in the identity must be equal.
    shared = lambda ident: {k: v for k, v in ident.items() if k != "source"}
    identity = loaded[0][2]["identity"]
    if any(shared(row[2]["identity"]) != shared(identity) for row in loaded[1:]):
        raise ValueError("serving partition identity mismatch (plan, encoder or runtime)")
    research_execution = None
    if "research_selected_moe" in identity["options"]:
        research_execution = ResearchSelectedMoeInput.from_record(
            identity["options"]["research_selected_moe"])
    execution_config = ({"research_selected_moe": research_execution.config.as_checkpoint()}
                        if research_execution is not None else {})
    execution_record = ({"research_selected_moe": research_execution.record()}
                        if research_execution is not None else {})
    # The one pass over the whole source; every part's stamp is held to it.
    whole = source_part_identity(source, digest_cache=source_digest_cache)
    for rank, path, part, *_ in loaded:
        prove_source_part(part["identity"].get("source"), whole, f"partition {rank}")
    expected_source = set(whole["tensors"])
    source_config = json.loads((source / "config.json").read_text())
    source_config.pop("quantization_config", None)
    base_format = loaded[0][4]["quantization_config"]["format"]
    # What the merged artifact says about slicing is what every part said, and
    # a part that says nothing is not overridden into saying something: the
    # keys travel only when the first part carries them, every part must agree,
    # and a merge of parts written before the declaration existed produces an
    # artifact that declares nothing -- which the loader gate then refuses
    # above one rank rather than reading as permission (tessera#328).
    base_slicing = {k: loaded[0][4]["quantization_config"][k]
                    for k in ("schema_minor", "tp_agnostic")
                    if k in loaded[0][4]["quantization_config"]}
    modules, groups, weight_map, copies = {}, {}, {}, []
    ignore, covered = set(), set()
    passthrough_bytes = 0
    # Every part's output shards hash ahead on a bounded pool (tessera#499);
    # each digest is still taken, and each refusal raised, where the serial
    # loop hashed that file, so a refusal names the file it named before.
    with _HashAhead(_output_payloads(loaded)) as hashed:
        for rank, path, part, manifest, config, index in loaded:
            owned, local_map, payloads = _prove_part_contents(
                rank, count, path, part, manifest, index, whole, hashed.sha256)
            if covered & owned:
                raise ValueError(f"partition {rank}: source tensor coverage disagrees with ownership")
            covered.update(owned)
            qconfig = config["quantization_config"]
            # Validate every carrier before equality: JSON floats and booleans can
            # compare equal to the integer-only declaration in Python.
            if "research_selected_moe" in qconfig:
                ResearchSelectedMoeConfig.from_checkpoint(qconfig["research_selected_moe"])
            if "research_selected_moe" in manifest:
                ResearchSelectedMoeInput.from_record(manifest["research_selected_moe"])
            if "research_selected_moe" in part["identity"]["options"]:
                ResearchSelectedMoeInput.from_record(part["identity"]["options"]["research_selected_moe"])
            if ({k: qconfig[k] for k in ("research_selected_moe",)
                 if k in qconfig} != execution_config
                    or {k: manifest[k] for k in ("research_selected_moe",) if k in manifest}
                    != execution_record):
                raise ValueError(f"partition {rank}: research_selected_moe disagrees with sealed export identity")
            if {k: v for k, v in config.items() if k != "quantization_config"} != source_config:
                raise ValueError(f"partition {rank}: model config disagrees with source identity")
            if qconfig["quant_method"] != "tessera" or qconfig["format"] != base_format:
                raise ValueError("partition quantization config disagrees")
            if {k: qconfig[k] for k in ("schema_minor", "tp_agnostic")
                    if k in qconfig} != base_slicing:
                raise ValueError(
                    f"partition {rank}: the parts disagree about what their bytes admit at load "
                    "(schema_minor / tp_agnostic) -- they were written by different exporters, "
                    "and a merged artifact may not declare more than every part of it does")
            declarations = {target for group in qconfig["config_groups"].values() for target in group["targets"]}
            if declarations != set(manifest["modules"]):
                raise ValueError(f"partition {rank}: config targets disagree with manifest modules")
            if modules.keys() & manifest["modules"].keys() or groups.keys() & qconfig["config_groups"].keys():
                raise ValueError("module or config group is claimed by two partitions")
            modules.update(manifest["modules"])
            groups.update(qconfig["config_groups"])
            ignore.update(qconfig["ignore"])
            for payload, filename in payloads:
                copies.append((payload, f"part-{rank:05d}-{filename}"))
            if weight_map.keys() & local_map.keys():
                raise ValueError("tensor appears in two partitions")
            weight_map.update({n: f"part-{rank:05d}-{s}" for n, s in local_map.items()})
            passthrough_bytes += manifest["totals"]["passthrough_bytes"]
    if covered != expected_source:
        raise ValueError("source tensor coverage is incomplete")
    if ignore & modules.keys():
        raise ValueError("a merged target is also declared ignored")
    validate_explicit_plan(identity["options"].get("plan"), modules, groups,
                           source_tensors=expected_source)
    config = copy.deepcopy(source_config)
    if research_execution is not None:
        research_execution.config.require_targets(
            {target: group["scheme"] for group in groups.values() for target in group["targets"]},
            "resident")
    config["quantization_config"] = {"quant_method": "tessera", "format": base_format,
                                      **base_slicing,
                                      **execution_config,
                                      "config_groups": groups, "ignore": sorted(ignore)}
    manifest = copy.deepcopy(loaded[0][3])
    manifest.pop("export_partition")
    observed = _merged_batch_widths(loaded)
    if observed is not None:
        manifest["encode_batch_observed"] = observed
    else:
        # A partial observation is not a histogram of the whole merged run.
        manifest.pop("encode_batch_observed", None)
    manifest["modules"] = modules
    manifest["merged_from"] = [{"index": r[0], "path": str(r[1]),
                                 "output_sha256": r[2]["output_sha256"]} for r in loaded]
    # The published artifact records the whole source this merge proved, not
    # part 0's subset.
    manifest["export_identity"] = {**identity, "source": whole}
    if source_digest_cache is not None:
        manifest["source_proof"] = source_digest_cache.receipt()
    moe = manifest["routed_moe"]
    for field in ("modules", "quantized_stacks"):
        moe[field] = sorted({name for row in loaded for name in row[3]["routed_moe"][field]})
    for field in ("packed_source_tensors", "unpacked_source_tensors", "quantized_source_tensors", "quantized_logical_units"):
        moe[field] = sum(row[3]["routed_moe"][field] for row in loaded)
    moe["disposition"] = ("quantized" if moe["quantized_stacks"] and not moe["modules"]
                          else "mixed" if moe["quantized_stacks"] else "passed_through_bf16")
    size = sum(payload.stat().st_size for payload, _ in copies)
    manifest["totals"] = summarize_modules(modules, passthrough_bytes, size)
    # No output directory or loadable config exists until every proof above passes.
    out.mkdir(parents=True)
    transfer = shutil.move if move else shutil.copy2
    for payload, name in copies:
        destination = out / name
        transfer(str(payload), str(destination))
        # A completed artifact must be readable by the serving identity,
        # including root-squashed containers on NFS. Copy-mode source parts
        # keep their private modes; only the destination gains read bits.
        make_artifact_readable(destination)
    for pattern in ("*.json", "*.txt", "*.jinja", "*.model"):
        for aux in source.glob(pattern):
            if aux.name not in ("config.json", "model.safetensors.index.json", "tessera_serving_manifest.json", "tessera_part_config.json"):
                shutil.copy2(aux, out / aux.name)
    (out / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": size}, "weight_map": weight_map}, indent=2))
    write_serving_manifest(out / "tessera_serving_manifest.json", manifest)
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return manifest


#: The block a shard-split part stamps for the checkpoint it was cut from
#: (``tessera_config.json`` ``source``, written by
#: ``tessera.export.export_checkpoint_streaming``; tessera#300), and the
#: ``source`` a serving part stamps in ``export_partition.identity``
#: (tessera#495): ``source_identity``'s binding restricted to the shards the
#: part read, so both merges (``experiments/merge_tessera_parts.py`` and
#: ``merge_serving_parts``) prove their parts against one pass over
#: ``--source`` rather than every part re-hashing the whole checkpoint.
SOURCE_PART_SCHEMA = "tessera.source-part.v1"


def source_inventory(source: Path) -> dict:
    """``tensor -> shard`` for a checkpoint, the headers' word on it.

    ``source_identity``'s inventory step on its own: with an index, every
    shard it names is opened and the headers must reproduce the index exactly
    -- an index that names a tensor no shard holds, or omits one a shard
    holds, is refused here rather than discovered by the first reader to ask
    for the tensor; without an index the ``*.safetensors`` files are the
    inventory.  One home, so the exporter that stamps a part and the merge
    that proves it read the same roster.
    """
    source = Path(source)
    index_path = source / "model.safetensors.index.json"
    tensors: dict = {}
    if index_path.exists():
        declared = json.loads(index_path.read_text())["weight_map"]
        files = sorted({_leaf(name) for name in declared.values()})
    else:
        declared = None
        files = sorted(path.name for path in source.glob("*.safetensors"))
    for name in files:
        for tensor in sorted(tensor_names(source / name)):
            if tensor in tensors:
                raise ValueError(f"source tensor appears twice: {tensor}")
            tensors[tensor] = name
    if declared is not None and tensors != declared:
        extra = sorted(set(declared) - set(tensors))
        unlisted = sorted(set(tensors) - set(declared))
        moved = sorted(t for t in set(declared) & set(tensors) if declared[t] != tensors[t])
        raise ValueError(
            "source index does not exactly cover the source tensor files: "
            f"named but held by no shard {extra[:3]}, held but unnamed "
            f"{unlisted[:3]}, in another shard than named {moved[:3]}")
    if not tensors:
        raise ValueError("source has no safetensors tensors")
    return tensors


def source_part_identity(source: Path, shards=None, *, workers=None, digest_cache=None) -> dict:
    """``source_identity`` for a part that read only ``shards``.

    The same binding under the same field names -- ``config_sha256``,
    ``auxiliary_sha256``, ``tensors`` (the whole inventory, so two parts of
    one checkpoint stamp the same roster) and ``files`` -- with ``files``
    hashed for the shards the part read, every shard when ``shards`` is
    ``None``.  A part's stamp therefore costs one pass over its own input,
    and the merge's one pass over the whole source proves every part's entry
    against the checkpoint it publishes for.  Two deliberate differences from
    ``source_identity``: the block names its ``schema``, and a source with no
    ``config.json`` records ``config_sha256: null`` instead of refusing -- the
    streaming exporter takes bare shard directories -- which is proved against
    the merge's ``--source`` like every other field, since a recorded absence
    and a present file do not compare equal.  A ``shards`` entry that names no
    shard of the source is refused: a filter over absent shards is a mistyped
    range, not an empty part.

    The chosen shards are hashed concurrently by :func:`sha256_files`
    (``workers`` as there); the document and the order of its refusals are
    the serial pass's. ``digest_cache`` (a
    :class:`tessera.source_digest_cache.SourceDigestCache`) serves the shard
    digests only -- never config or auxiliary files -- and the document is the
    same as a fresh pass's; the caller records the cache's receipt.
    """
    source = Path(source)
    tensors = source_inventory(source)
    files = sorted(set(tensors.values()))
    if shards is None:
        chosen = files
    else:
        chosen = sorted({_leaf(name) for name in shards})
        unknown = sorted(set(chosen) - set(files))
        if unknown:
            raise KeyError(f"shard_filter names absent shards: {unknown[:5]}")
        if not chosen:
            raise ValueError("shard_filter selected no shards")
    auxiliary = sorted({p for pattern in ("*.json", "*.txt", "*.jinja", "*.model")
                        for p in source.glob(pattern)})
    config_path = source / "config.json"
    # Config and auxiliary first, as the serial pass evaluated them, so their
    # refusals still precede any shard's.
    config_sha256 = sha256_file(config_path) if config_path.exists() else None
    auxiliary_sha256 = {p.name: sha256_file(p) for p in auxiliary}
    digests = sha256_files([source / name for name in chosen], workers,
                           digest_cache.sha256 if digest_cache is not None else None)
    return {"schema": SOURCE_PART_SCHEMA,
            "config_sha256": config_sha256,
            "auxiliary_sha256": auxiliary_sha256,
            "files": dict(zip(chosen, digests)),
            "tensors": tensors}


def prove_source_part(stamp, whole: dict, who: str) -> None:
    """Hold one part's source stamp to one pass over the whole source.

    ``whole`` is a whole-checkpoint identity: :func:`source_part_identity`
    with no filter, or the schema-less :func:`source_identity` a cached-unit
    manifest carries; only the fields both share are read. The stamp must be
    a ``tessera.source-part.v1`` block, its ``config_sha256``,
    ``auxiliary_sha256`` and ``tensors`` must equal the source's, and every
    shard it read must be a shard of the source with the same sha256. A
    shard of the source the stamp does not name is not this part's to prove.
    Raises ``ValueError`` naming ``who`` and the field or shard at fault.
    """
    if not isinstance(stamp, dict) or stamp.get("schema") != SOURCE_PART_SCHEMA:
        found = stamp.get("schema") if isinstance(stamp, dict) else type(stamp).__name__
        raise ValueError(f"{who}: source identity is not a {SOURCE_PART_SCHEMA} stamp "
                         f"({found!r}); it was written before tessera#495, re-export it")
    for field in ("config_sha256", "auxiliary_sha256"):
        if stamp.get(field) != whole.get(field):
            raise ValueError(f"{who}: source identity changed since partition export: "
                             f"stamped {field} {stamp.get(field)!r}, source {whole.get(field)!r}")
    if stamp.get("tensors") != whole.get("tensors"):
        mine, theirs = stamp.get("tensors") or {}, whole.get("tensors") or {}
        moved = sorted(t for t in set(mine) & set(theirs) if mine[t] != theirs[t])
        raise ValueError(f"{who}: source identity changed since partition export: tensor "
                         f"inventory differs: only stamped {sorted(set(mine) - set(theirs))[:3]}, "
                         f"only in source {sorted(set(theirs) - set(mine))[:3]}, "
                         f"in another shard {moved[:3]}")
    files = stamp.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"{who}: source identity stamps no shards")
    for shard, digest in sorted(files.items()):
        if shard not in whole["files"]:
            raise ValueError(f"{who}: source identity names shard {shard}, "
                             f"which the source does not hold")
        if digest != whole["files"][shard]:
            raise ValueError(f"{who}: source identity changed since partition export: "
                             f"{shard} read sha256 {digest}, the source's is {whole['files'][shard]}")


#: The block an exporter stamps for the shards it WROTE (``tessera_config.json``
#: ``output``; tessera#337).  ``source`` says which checkpoint's bytes went in;
#: it cannot say which bytes came out, and that is the whole of #337: a retry
#: into a completed output directory overwrote shards one at a time, so a run
#: that stopped part way left new shards beside the previous run's index,
#: config and source seal -- valid replacement blobs for the same named unit at
#: the same rung and shape, which every name-, header- and manifest-shaped
#: check in the merge accepts.  The serving-part path already binds its output
#: (``export_partition.output_sha256``, verified in :func:`merge_serving_parts`
#: before a byte is copied); this is the same binding for the legacy parts,
#: under a schema so a future spelling is refused rather than misread.
OUTPUT_PART_SCHEMA = "tessera.output-part.v1"


def output_part_stamp(files: dict) -> dict:
    """The ``output`` block over digests a writer already took.

    The one home of the block's shape.  The exporter hashes each shard as it
    finishes it -- while the bytes it just wrote are still warm, and so that
    the stamp is of what THIS run wrote rather than of whatever is on disk
    when the config is written -- and hands the accumulated mapping here.
    """
    return {"schema": OUTPUT_PART_SCHEMA, "files": dict(sorted(files.items()))}


def output_part_identity(out: Path, files) -> dict:
    """The same block, hashed now: what a verifier computes and compares.

    ``files`` are local shard filenames under ``out``, digested with the same
    :func:`sha256_file` the serving path and the source stamp use, so there is
    one digest of a checkpoint file in this tree rather than three.  A merge
    that wants to name the shard that differs hashes shard by shard against
    the stamped mapping instead of comparing two whole blocks.
    """
    out = Path(out)
    return output_part_stamp({name: sha256_file(out / _leaf(name)) for name in files})
