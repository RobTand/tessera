"""Runtime observations for the task endpoint witness (tessera#1056).

This module is the joining half of the witness boundary. The serving
workers observe themselves through the serving-owned runtime module
and write one JSON file each; this module reads those files, reads the
served alias from the listener's live ``/v1/models`` reply, and proves
the bytes against the served directory -- all with the standard library
only. It takes no caller-supplied fact about loaded state: a caller
hands over worker observation files, a URL and a directory, never a
digest, rank list or lifetime. :mod:`tessera.endpoint_witness` owns the
join and its refusals; this module owns the reads. It never imports the
serving package, torch or vLLM, so the D50 task adapter consumes the
receipt with no serving import and no rank lifecycle work.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Mapping

#: File names that can carry tokenizer bytes in a served directory.
TOKENIZER_NAMES = ("tokenizer.json", "tokenizer_config.json", "vocab.json",
                   "merges.txt", "special_tokens_map.json")


def _stamp(clock: Callable[[], float] | None) -> float:
    observed = clock() if clock is not None else time.time()
    if not isinstance(observed, (int, float)) or not observed > 0:
        raise ValueError("observation clock returned no positive time")
    return observed


def _text(value: str, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} is not a non-empty string")
    return value


def _served_roster(root: Path, what: str) -> dict[str, str]:
    """The sha256 of every file under the served directory, by relative name."""
    from tessera import endpoint_witness as witness_module
    files = witness_module.served_roster(root)
    digests = {}
    for name, path in files.items():
        if "\\" in name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"{what} file name {name!r} is not a relative name")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
        except OSError as exc:
            raise ValueError(f"cannot read {what} file {name!r} ({exc})") from exc
        digests[name] = digest.hexdigest()
    return digests


def roster_sizes(root: Path, name: str) -> int:
    """The byte size of one served roster file, read at observation time."""
    try:
        return (root / name).stat().st_size
    except OSError as exc:
        raise ValueError(f"cannot read served file {name!r} ({exc})") from exc


def observe_listener(base_url: str, *, lifetime_id: str,
                     clock: Callable[[], float] | None = None,
                     timeout_s: float = 3.0) -> dict:
    """Read the served alias from the listener's ``/v1/models`` reply.

    Refuses a non-HTTP address, an unreachable listener, a non-JSON reply,
    a reply with no served model, and a reply with several served models.
    """
    _text(base_url, "listener base_url")
    _text(lifetime_id, "listener lifetime_id")
    if not base_url.startswith(("http://", "https://")):
        raise ValueError(f"listener base_url {base_url!r} is not an http(s) address")
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models",
                                    timeout=timeout_s) as reply:
            payload = json.loads(reply.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 -- any transport or parse failure refuses alike
        raise ValueError(f"cannot read the served models reply: {exc}") from exc
    entries = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("the served models reply lists no served model")
    names = [entry.get("id") for entry in entries
             if isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry["id"]]
    if len(names) != 1 or len(entries) != 1:
        raise ValueError(f"the served models reply lists several served models: {names!r}")
    return {"endpoint": base_url, "served_alias": names[0],
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}


def collect_worker_observations(paths: list[str | Path], *, lifetime_id: str) -> list[dict]:
    """Read one worker observation file per serving rank and refuse gaps.

    Each file holds the three reads the serving-owned runtime module
    took inside one worker: ``identity`` (rank, world size), ``model``
    (model path, served names, tokenizer path, vocabulary length) and
    ``wires`` (loaded module digests). All three must share the probe's
    ``lifetime_id``; a file from another serve is two serves, not one
    witness. The ranks must cover ``range(world_size)`` exactly: a missing
    rank is an incomplete world, never an observation about a smaller one.
    """
    _text(lifetime_id, "worker lifetime_id")
    if not isinstance(paths, list) or not paths:
        raise ValueError("worker observation paths are not a non-empty list")
    observed = []
    for raw in paths:
        try:
            item = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read worker observation {raw} ({exc})") from exc
        if not isinstance(item, dict):
            raise ValueError(f"worker observation {raw} is not an object")
        for section in ("identity", "model", "wires"):
            part = item.get(section)
            if not isinstance(part, dict):
                raise ValueError(f"worker observation {raw} carries no {section}")
            if part.get("lifetime_id") != lifetime_id:
                raise ValueError(f"worker observation {raw} section {section} "
                                 f"is from lifetime {part.get('lifetime_id')!r}, "
                                 f"not {lifetime_id!r}")
        identity, model, wires = item["identity"], item["model"], item["wires"]
        if (type(identity.get("rank")) is not int or identity["rank"] < 0
                or type(identity.get("world_size")) is not int
                or identity["world_size"] <= 0
                or identity["rank"] >= identity["world_size"]):
            raise ValueError(f"worker observation {raw} names no rank of its world")
        if not isinstance(model.get("served_names"), list) or not model["served_names"]:
            raise ValueError(f"worker observation {raw} names no served name")
        if not isinstance(wires.get("wires"), dict) or not wires["wires"]:
            raise ValueError(f"worker observation {raw} holds no loaded wire")
        observed.append(item)
    worlds = {item["identity"]["world_size"] for item in observed}
    if len(worlds) != 1:
        raise ValueError(f"worker observations disagree on world size: {sorted(worlds)}")
    world = worlds.pop()
    ranks = sorted(item["identity"]["rank"] for item in observed)
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"worker observations repeat a rank: {ranks}")
    if ranks != list(range(world)):
        raise ValueError(f"worker observations cover ranks {ranks}, not the whole world of {world}")
    models = {item["model"]["model"] for item in observed}
    if len(models) != 1:
        raise ValueError(f"worker observations disagree on the loaded model: {sorted(models)}")
    return sorted(observed, key=lambda item: item["identity"]["rank"])


def check_listener_owned(listener: Mapping[str, Any], workers: list[dict]) -> None:
    """Refuse a listener whose alias no serving worker serves.

    The alias the ``/v1/models`` reply lists must equal a served name from
    every worker's own model config. A listener that answers with another
    alias is another serve -- or a replaced one -- not this one.
    """
    alias = listener.get("served_alias")
    for item in workers:
        names = item["model"]["served_names"]
        if alias not in names:
            raise ValueError(f"listener alias {alias!r} is not a served name "
                             f"of rank {item['identity']['rank']} ({names!r})")


def observe_launch(attempt_id: str, workers: list[dict], *, lifetime_id: str,
                   clock: Callable[[], float] | None = None) -> dict:
    """Record the launch attempt and the complete observed rank set.

    The ranks come from the workers' own identities, never from the
    invocation: only a rank the distributed group established can appear.
    """
    _text(attempt_id, "launch attempt_id")
    _text(lifetime_id, "launch lifetime_id")
    if not isinstance(workers, list) or not workers:
        raise ValueError("launch workers are not a non-empty list")
    ranks = sorted(item["identity"]["rank"] for item in workers)
    return {"attempt_id": attempt_id, "ranks": ranks,
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}

def observe_rank_bytes(workers: list[dict], artifact_dir: str | Path, *,
                       lifetime_id: str,
                       clock: Callable[[], float] | None = None) -> list[dict]:
    """One loaded-byte observation per serving rank, keyed by its wires.

    Each rank's ``files`` map joins the served roster with the loaded wire
    digests its worker read from resident module state; ``sizes`` carries
    the live file size for roster entries and the digest length (32 bytes
    of sha256) for each wire entry. All ranks cover the same union roster:
    a rank that loaded a module another rank did not refuses at the join,
    and a rank with no loaded wire never reaches here -- the worker
    observation already refused it -- so an unrelated directory cannot
    stand in for a rank that loaded nothing.
    """
    _text(lifetime_id, "rank lifetime_id")
    if not isinstance(workers, list) or not workers:
        raise ValueError("rank workers are not a non-empty list")
    root = Path(artifact_dir)
    if not root.is_dir():
        raise ValueError(f"rank served directory {root} is not a directory")
    roster = _served_roster(root, "rank")
    union: set[str] = set()
    for item in workers:
        union.update(f"loaded:{prefix}" for prefix in item["wires"]["wires"])
    if not union:
        raise ValueError("serving workers hold no loaded wire")
    observed = []
    for item in sorted(workers, key=lambda entry: entry["identity"]["rank"]):
        rank = item["identity"]["rank"]
        wires = {f"loaded:{prefix}": digest
                 for prefix, digest in item["wires"]["wires"].items()}
        missing = sorted(union - set(wires))
        if missing:
            raise ValueError(f"rank {rank} loaded no wire for {missing}")
        files = dict(roster)
        files.update(wires)
        sizes = {name: (32 if name.startswith("loaded:") else roster_sizes(root, name))
                 for name in files}
        observed.append({"rank": rank, "files": files, "sizes": sizes,
                         "bytes": sum(sizes.values()),
                         "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)})
    return observed


def observe_server_tokenizer(workers: list[dict], artifact_dir: str | Path, *,
                             lifetime_id: str,
                             clock: Callable[[], float] | None = None) -> dict:
    """The server tokenizer facts the serving workers observed.

    The vocabulary length comes from the engine's initialized tokenizer
    inside each worker; every rank must agree, or the world serves two
    vocabularies and refuses. Tokenizer byte files are every served file
    whose name the receipt owns, hashed at observation time.
    """
    _text(lifetime_id, "tokenizer lifetime_id")
    if not isinstance(workers, list) or not workers:
        raise ValueError("tokenizer workers are not a non-empty list")
    sizes_agree = {item["model"]["vocab_size"] for item in workers}
    if len(sizes_agree) != 1:
        raise ValueError(f"serving ranks disagree on vocabulary length: {sorted(sizes_agree)}")
    vocab_size = sizes_agree.pop()
    if type(vocab_size) is not int or vocab_size <= 0:
        raise ValueError("the serving workers state no vocabulary length")
    root = Path(artifact_dir)
    if not root.is_dir():
        raise ValueError(f"tokenizer served directory {root} is not a directory")
    digests, sizes = {}, {}
    for name in TOKENIZER_NAMES:
        path = root / name
        if not path.is_file() or path.is_symlink():
            continue
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            sizes[name] = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"cannot read served tokenizer file {name!r} ({exc})") from exc
        digests[name] = digest.hexdigest()
    if not digests:
        raise ValueError("the served artifact holds no tokenizer file")
    return {"vocab_size": vocab_size, "vocab_source": "server-tokenizer",
            "files": digests, "sizes": sizes,
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}


def publish_witness(publish_root: str | Path, *, witness: dict[str, Any],
                    served_dir: str | Path) -> Path:
    """Write one self-contained receipt through an exclusive create.

    Re-proves the bytes against ``served_dir`` before writing: a witness
    whose proof block is JSON agreement only never publishes. Refuses to
    overwrite an existing receipt. Writes the receipt and its sha256 sidecar
    beside it. Returns the receipt path.
    """
    from tessera import endpoint_witness as witness_module
    reason = witness_module.verify_witness(witness, served_dir=served_dir)
    if reason is not None:
        raise ValueError(f"refused witness: {reason}")
    _text(witness["fingerprint"], "witness fingerprint")
    root = Path(publish_root)
    root.mkdir(parents=True, exist_ok=True)
    receipt = root / f"endpoint_runtime_witness.v1.{witness['fingerprint'][:16]}.json"
    try:
        with receipt.open("x", encoding="utf-8") as handle:
            handle.write(witness_module.canonical(witness) + "\n")
    except FileExistsError as exc:
        raise ValueError(f"receipt {receipt} already exists") from exc
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    receipt.with_suffix(receipt.suffix + ".sha256").write_text(digest + "\n", encoding="utf-8")
    return receipt


def receipt_path(publish_root: str | Path, witness: dict[str, Any]) -> Path:
    """The public receipt path :func:`publish_witness` writes for a witness."""
    from tessera import endpoint_witness as witness_module
    _text(witness.get("fingerprint", ""), "witness fingerprint")
    return (Path(publish_root)
            / f"endpoint_runtime_witness.v1.{witness['fingerprint'][:16]}.json")


__all__ = ["TOKENIZER_NAMES", "check_listener_owned",
           "collect_worker_observations", "observe_launch", "observe_listener",
           "observe_rank_bytes", "observe_server_tokenizer", "publish_witness",
           "receipt_path", "roster_sizes"]
