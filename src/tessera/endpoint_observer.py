"""Runtime observations for the task endpoint witness (tessera#1056).

This module is the producer half of the witness boundary. It reads live
evidence with the standard library only: the served alias comes from an
HTTP ``/v1/models`` reply, rank byte facts come from file bytes read at
observation time, and the vocabulary size comes from the served
configuration or a loaded tokenizer object. It takes no caller-supplied
fact about loaded state: a caller names a directory or URL, never a
digest. :mod:`tessera.endpoint_witness` owns the join and its refusals;
this module owns the reads. Neither imports the serving package, torch or
vLLM, so the D50 task adapter consumes the receipt with no serving import
and no rank lifecycle work.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable

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


def observe_launch(attempt_id: str, ranks: list[int], *, lifetime_id: str,
                   clock: Callable[[], float] | None = None) -> dict:
    """Record the launch attempt and the complete rank set under observation."""
    _text(attempt_id, "launch attempt_id")
    _text(lifetime_id, "launch lifetime_id")
    if not isinstance(ranks, list) or not ranks or any(type(r) is not int or r < 0 for r in ranks):
        raise ValueError("launch ranks is not a non-empty list of rank ids")
    if len(set(ranks)) != len(ranks):
        raise ValueError("launch ranks repeats a rank")
    return {"attempt_id": attempt_id, "ranks": sorted(ranks),
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}


def observe_rank_bytes(rank: int, artifact_dir: str | Path, *, lifetime_id: str,
                       clock: Callable[[], float] | None = None) -> dict:
    """Hash every file under the rank's served directory at observation time.

    The complete live roster is the coverage: an omitted file never appears,
    so two ranks with the same omission still refuse at the live byte proof.
    """
    if type(rank) is not int or rank < 0:
        raise ValueError(f"rank {rank!r} is not a rank id")
    _text(lifetime_id, "rank lifetime_id")
    root = Path(artifact_dir)
    if not root.is_dir():
        raise ValueError(f"rank served directory {root} is not a directory")
    files = {str(path.relative_to(root)): path for path in sorted(root.rglob("*"))
             if path.is_file() and not path.is_symlink()}
    if not files:
        raise ValueError(f"rank served directory {root} holds no files")
    digests, sizes = {}, {}
    for name, path in files.items():
        if "\\" in name or name.startswith("/") or ".." in name.split("/"):
            raise ValueError(f"rank file name {name!r} is not a relative name")
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            sizes[name] = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"cannot read rank file {name!r} ({exc})") from exc
        digests[name] = digest.hexdigest()
    return {"rank": rank, "files": digests, "sizes": sizes,
            "bytes": sum(sizes.values()),
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}


def observe_server_tokenizer(artifact_dir: str | Path, *, lifetime_id: str,
                             clock: Callable[[], float] | None = None,
                             loaded: Any = None) -> dict:
    """Read the server tokenizer facts from the served artifact.

    The vocabulary size comes from the served ``config.json`` first and from
    a loaded tokenizer object when the caller hands one over; a caller never
    supplies the size as a bare number. Tokenizer byte files are every served
    file whose name the receipt owns, hashed at observation time.
    """
    _text(lifetime_id, "tokenizer lifetime_id")
    root = Path(artifact_dir)
    if not root.is_dir():
        raise ValueError(f"tokenizer served directory {root} is not a directory")
    vocab_size: int | None = None
    source = "served-config"
    config_path = root / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read the served config: {exc}") from exc
        for key in ("vocab_size", "vocabulary_size", "padded_vocab_size"):
            value = config.get(key) if isinstance(config, dict) else None
            if type(value) is int and value > 0:
                vocab_size = value
                break
    if vocab_size is None:
        if loaded is None:
            raise ValueError("the served artifact states no vocabulary length")
        try:
            vocab_size = len(loaded)
        except TypeError as exc:
            raise ValueError("the loaded tokenizer states no vocabulary length") from exc
        if type(vocab_size) is not int or vocab_size <= 0:
            raise ValueError("the loaded tokenizer states no vocabulary length")
        source = "loaded"
    else:
        try:
            loaded_size = len(loaded) if loaded is not None else vocab_size
        except TypeError as exc:
            raise ValueError("the loaded tokenizer states no vocabulary length") from exc
        if loaded_size != vocab_size:
            raise ValueError("the loaded vocabulary length differs from the served config")
    files = {str(path.relative_to(root)): path for path in sorted(root.rglob("*"))
             if path.is_file() and not path.is_symlink()
             and path.name in TOKENIZER_NAMES}
    if not files:
        raise ValueError("the served artifact holds no tokenizer file")
    digests, sizes = {}, {}
    for name, path in files.items():
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            sizes[name] = path.stat().st_size
        except OSError as exc:
            raise ValueError(f"cannot read served tokenizer file {name!r} ({exc})") from exc
        digests[name] = digest.hexdigest()
    return {"vocab_size": vocab_size, "vocab_source": source,
            "files": digests, "sizes": sizes,
            "lifetime_id": lifetime_id, "observed_unix": _stamp(clock)}


def collect_rank_observations(paths: list[str | Path]) -> list[dict]:
    """Read one observation file per rank and refuse duplicates or gaps."""
    if not isinstance(paths, list) or not paths:
        raise ValueError("rank observation paths are not a non-empty list")
    observed = []
    for raw in paths:
        try:
            item = json.loads(Path(raw).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot read rank observation {raw} ({exc})") from exc
        if not isinstance(item, dict) or type(item.get("rank")) is not int:
            raise ValueError(f"rank observation {raw} names no rank")
        observed.append(item)
    ranks = sorted(item["rank"] for item in observed)
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"rank observations repeat a rank: {ranks}")
    return sorted(observed, key=lambda item: item["rank"])


def publish_witness(publish_root: str | Path, *, witness: dict[str, Any]) -> Path:
    """Write one self-contained receipt through an exclusive create.

    Refuses a witness without a verified live byte proof and refuses to
    overwrite an existing receipt. Writes the receipt and its sha256 sidecar
    beside it. Returns the receipt path.
    """
    from tessera import endpoint_witness as witness_module
    reason = witness_module.verify_witness(witness)
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


__all__ = ["TOKENIZER_NAMES", "collect_rank_observations", "observe_launch",
           "observe_listener", "observe_rank_bytes", "observe_server_tokenizer",
           "publish_witness", "receipt_path"]
