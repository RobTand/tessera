"""The task endpoint runtime witness: endpoint, alias, attempt, ranks, bytes, tokenizer (tessera#1056).

The D50 task adapter needs one public join of the runtime facts a serve
OBSERVED: the listener endpoint and served alias, the launch attempt and the
complete rank set, the loaded artifact bytes each rank observed, and the
server tokenizer facts. Input manifests, aliases, leases, launch arguments
and publication custody receipts do not establish loaded runtime state, so
this producer binds only observed evidence.

THE RECEIPT. :func:`build_witness` joins four observed pieces into one
self-contained JSON document with schema ``tessera.endpoint_runtime_witness.v1``:

* ``listener`` -- the endpoint and served alias the service witness observed.
* ``launch`` -- the attempt identity and the complete observed rank set.
* ``artifacts`` -- one rank-local loaded-byte observation per rank.
* ``tokenizer`` -- the server tokenizer facts, not client metadata.

THE RULE. :func:`verify_witness` re-derives the join and refuses missing,
incomplete or inconsistent evidence by name. No Tessera serving import lives
here or in :mod:`tessera.endpoint_observer`: the producer observes file
bytes and HTTP replies with the standard library, never a live rank object
or listener handle.

BYTE COVERAGE. Each rank observation names every file it covered, the
sha256 of each file's loaded bytes, and the byte size of each file; the
rank ``bytes`` total must equal the sum of its file sizes. Coverage is
complete only when every rank covers the same file set, the digests agree
across ranks, and :func:`prove_loaded_bytes` rehashes one live served
directory and compares every digest, size and roster entry. Rank agreement
alone never proves completeness: identical omissions or identical wrong
digests across ranks still refuse at the live proof.

OBSERVATION LIFETIME. All observations carry an ``observed_unix`` stamp and
a ``lifetime_id``. The join refuses observations from different lifetimes: a
byte observation from one serve and a listener observation from another is
two serves, not one witness. The live byte proof carries its own stamp; a
proof from another lifetime never verifies.

THE BYTE PROOF. :func:`stamp_byte_proof` reads one served directory, proves
the witness against its bytes through :func:`prove_loaded_bytes`, and stamps
a ``byte_proof`` block with the proven roster, total bytes, digest of the
proof input, and stamp time. :func:`verify_witness` requires that block and
re-derives it, so a receipt without a live proof, a contradictory
``byte_coverage`` block, or a forged size never verifies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

SCHEMA = "tessera.endpoint_runtime_witness.v1"

#: The fields a listener observation must carry. ``endpoint`` is the listener
#: address the witness probed (host and port). ``served_alias`` is the model
#: name the ``/v1/models`` reply listed. ``lifetime_id`` binds this
#: observation to the same serve as the byte and tokenizer facts.
LISTENER_KEYS = frozenset({"endpoint", "served_alias", "lifetime_id", "observed_unix"})

#: The fields a launch observation must carry. ``attempt_id`` is the launch
#: attempt (the PB nonce or container label set). ``ranks`` is the complete
#: observed rank set, each an integer rank id.
LAUNCH_KEYS = frozenset({"attempt_id", "ranks", "lifetime_id", "observed_unix"})

#: The fields one rank's artifact observation must carry. ``rank`` names the
#: rank. ``files`` maps a file name to its loaded-byte ``sha256``. ``sizes``
#: maps the same file name to its loaded-byte size. ``bytes`` is the total
#: loaded bytes the rank observed, and must equal the sum of ``sizes``.
ARTIFACT_KEYS = frozenset({"rank", "files", "sizes", "bytes", "lifetime_id", "observed_unix"})

#: The fields a server tokenizer observation must carry. ``vocab_size`` is
#: the size read from the served configuration or loaded object.
#: ``vocab_source`` names which one supplied it. ``files`` maps each served
#: tokenizer file name to its ``sha256`` as read from the served artifact,
#: and ``sizes`` maps the same name to its byte size.
TOKENIZER_KEYS = frozenset({"vocab_size", "vocab_source", "files", "sizes",
                            "lifetime_id", "observed_unix"})

#: Vocabulary sources the tokenizer observation accepts.
VOCAB_SOURCES = frozenset({"served-config", "loaded"})

_HEX64 = frozenset("0123456789abcdef")


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in _HEX64 for c in value)


def _refuse(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _require_keys(observed: Mapping[str, Any], keys: frozenset, where: str) -> None:
    _refuse(isinstance(observed, Mapping), f"{where} is not an object")
    missing = sorted(keys - set(observed))
    _refuse(not missing, f"{where} misses {missing}")
    unknown = sorted(set(observed) - keys)
    _refuse(not unknown, f"{where} carries unknown field(s) {unknown}")


def _check_name(name: Any, where: str) -> None:
    _refuse(isinstance(name, str) and name and "\\" not in name
            and not name.startswith("/") and ".." not in name.split("/"),
            f"{where} file name {name!r} is not a relative name")


def _check_sizes(sizes: Any, files: Mapping[str, Any], where: str) -> int:
    _refuse(isinstance(sizes, Mapping) and sizes, f"{where} sizes is not a non-empty map")
    _refuse(sorted(sizes) == sorted(files), f"{where} sizes cover different files than digests")
    total = 0
    for name, size in sizes.items():
        _refuse(type(size) is int and size > 0, f"{where} file {name!r} size is not a positive count")
        total += size
    return total


def _check_lifetime(records: list, lifetime: str, where: str) -> None:
    for record in records:
        _refuse(record.get("lifetime_id") == lifetime,
                f"{where} mixes lifetimes: {record.get('lifetime_id')!r} is not {lifetime!r}")


def _check_listener(listener: Mapping[str, Any]) -> None:
    _require_keys(listener, LISTENER_KEYS, "listener")
    _refuse(isinstance(listener["endpoint"], str) and listener["endpoint"],
            "listener endpoint is not a non-empty string")
    _refuse(isinstance(listener["served_alias"], str) and listener["served_alias"],
            "listener served_alias is not a non-empty string")
    _refuse(isinstance(listener["lifetime_id"], str) and listener["lifetime_id"],
            "listener lifetime_id is not a non-empty string")
    _refuse(isinstance(listener["observed_unix"], (int, float)) and listener["observed_unix"] > 0,
            "listener observed_unix is not a positive time")


def _check_launch(launch: Mapping[str, Any]) -> None:
    _require_keys(launch, LAUNCH_KEYS, "launch")
    _refuse(isinstance(launch["attempt_id"], str) and launch["attempt_id"],
            "launch attempt_id is not a non-empty string")
    ranks = launch["ranks"]
    _refuse(isinstance(ranks, list) and ranks and all(type(r) is int and r >= 0 for r in ranks),
            "launch ranks is not a non-empty list of rank ids")
    _refuse(len(set(ranks)) == len(ranks), "launch ranks repeats a rank")
    _refuse(isinstance(launch["lifetime_id"], str) and launch["lifetime_id"],
            "launch lifetime_id is not a non-empty string")
    _refuse(isinstance(launch["observed_unix"], (int, float)) and launch["observed_unix"] > 0,
            "launch observed_unix is not a positive time")


def _check_artifact(observed: Mapping[str, Any], where: str) -> None:
    _require_keys(observed, ARTIFACT_KEYS, where)
    _refuse(type(observed["rank"]) is int and observed["rank"] >= 0,
            f"{where} rank is not a rank id")
    files = observed["files"]
    _refuse(isinstance(files, Mapping) and files,
            f"{where} files is not a non-empty map")
    for name, digest in files.items():
        _check_name(name, where)
        _refuse(_is_hex64(digest), f"{where} file {name!r} sha256 is not a digest")
    total = _check_sizes(observed["sizes"], files, where)
    _refuse(type(observed["bytes"]) is int and observed["bytes"] > 0,
            f"{where} bytes is not a positive count")
    _refuse(observed["bytes"] == total,
            f"{where} sizes do not add to its bytes ({total} observed, {observed['bytes']} stated)")
    _refuse(isinstance(observed["lifetime_id"], str) and observed["lifetime_id"],
            f"{where} lifetime_id is not a non-empty string")
    _refuse(isinstance(observed["observed_unix"], (int, float)) and observed["observed_unix"] > 0,
            f"{where} observed_unix is not a positive time")


def _check_tokenizer(tokenizer: Mapping[str, Any]) -> None:
    _require_keys(tokenizer, TOKENIZER_KEYS, "tokenizer")
    _refuse(type(tokenizer["vocab_size"]) is int and tokenizer["vocab_size"] > 0,
            "tokenizer vocab_size is not a positive count")
    _refuse(tokenizer["vocab_source"] in VOCAB_SOURCES,
            f"tokenizer vocab_source {tokenizer.get('vocab_source')!r} is not a served source")
    files = tokenizer["files"]
    _refuse(isinstance(files, Mapping) and files,
            "tokenizer files is not a non-empty map")
    for name, digest in files.items():
        _check_name(name, "tokenizer")
        _refuse(_is_hex64(digest), f"tokenizer file {name!r} sha256 is not a digest")
    _check_sizes(tokenizer["sizes"], files, "tokenizer")
    _refuse(isinstance(tokenizer["lifetime_id"], str) and tokenizer["lifetime_id"],
            "tokenizer lifetime_id is not a non-empty string")
    _refuse(isinstance(tokenizer["observed_unix"], (int, float)) and tokenizer["observed_unix"] > 0,
            "tokenizer observed_unix is not a positive time")


def canonical(value: Any) -> str:
    """One spelling per JSON value, so two equal witnesses compare equal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def witness_fingerprint(witness: Mapping[str, Any]) -> str:
    """The sha256 of the joined body: endpoint, alias, attempt, ranks, bytes, tokenizer."""
    body = {key: witness[key] for key in ("listener", "launch", "artifacts", "tokenizer")}
    return hashlib.sha256(canonical(body).encode()).hexdigest()


def _served_files(served_dir: Path) -> dict[str, Path]:
    root = Path(served_dir)
    _refuse(root.is_dir(), f"served directory {root} is not a directory")
    found = {str(path.relative_to(root)): path for path in sorted(root.rglob("*"))
             if path.is_file() and not path.is_symlink()}
    _refuse(found, f"served directory {root} holds no files")
    return found


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prove_loaded_bytes(witness: Mapping[str, Any], served_dir: str | Path) -> str | None:
    """Why the witness bytes differ from the served directory, or None when they match.

    Reads every file under ``served_dir`` and compares the live roster,
    digests and sizes against the joined body. Never raises: an unreadable
    directory is a refusal string, not an exception.
    """
    try:
        live = _served_files(Path(served_dir))
    except (ValueError, OSError) as exc:
        return f"byte proof cannot read the served directory ({type(exc).__name__}: {exc})"
    try:
        body_files = sorted(witness["artifacts"][0]["files"])
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        return f"malformed witness ({type(exc).__name__}: {exc})"
    live_names = sorted(live)
    if live_names != body_files:
        missing = sorted(set(body_files) - set(live_names))
        extra = sorted(set(live_names) - set(body_files))
        if missing:
            return f"byte proof misses served files: {missing}"
        return f"byte proof holds unexpected served files: {extra}"
    for name in live_names:
        try:
            actual_digest = _digest_file(live[name])
            actual_size = live[name].stat().st_size
        except OSError as exc:
            return f"byte proof cannot read served file {name!r} ({exc})"
        try:
            ranks = [(item["rank"], item["files"][name], item["sizes"][name])
                     for item in witness["artifacts"]]
            token_digest = witness["tokenizer"]["files"].get(name)
            token_size = witness["tokenizer"]["sizes"].get(name)
        except (KeyError, TypeError, AttributeError) as exc:
            return f"malformed witness ({type(exc).__name__}: {exc})"
        for rank, digest, size in ranks:
            if digest != actual_digest:
                return (f"byte proof digest differs for served file {name!r} "
                        f"on rank {rank}")
            if size != actual_size:
                return f"byte proof size differs for served file {name!r} on rank {rank}"
        if token_digest is not None and token_digest != actual_digest:
            return f"byte proof digest differs for served tokenizer file {name!r}"
        if token_size is not None and token_size != actual_size:
            return f"byte proof size differs for served tokenizer file {name!r}"
    return None


def stamp_byte_proof(witness: Mapping[str, Any], served_dir: str | Path, *,
                     clock: Callable[[], float] | None = None) -> dict:
    """Stamp a live byte proof onto a joined witness, or refuse by name.

    Proves the witness against the served directory first: a witness whose
    digests, sizes or roster differ from the live bytes never stamps.
    """
    reason = prove_loaded_bytes(witness, served_dir)
    _refuse(reason is None, reason or "byte proof failed")
    live = _served_files(Path(served_dir))
    files = sorted(live)
    sizes = {name: live[name].stat().st_size for name in files}
    total = sum(sizes.values())
    observed = (clock() if clock is not None else __import__("time").time())
    _refuse(isinstance(observed, (int, float)) and observed > 0,
            "byte proof observed_unix is not a positive time")
    stamped = json.loads(canonical(witness))
    stamped["byte_proof"] = {
        "served_files": files,
        "served_sizes": sizes,
        "served_bytes": total,
        "proof_input": hashlib.sha256(canonical(
            {"files": {name: witness["artifacts"][0]["files"][name] for name in files},
             "sizes": {name: witness["artifacts"][0]["sizes"][name] for name in files},
             "served_sizes": sizes}).encode()).hexdigest(),
        "lifetime_id": witness["listener"]["lifetime_id"],
        "observed_unix": observed,
    }
    return stamped


def _expected_proof(rebuilt: Mapping[str, Any], presented: Mapping[str, Any]) -> dict:
    live_names = sorted(rebuilt["artifacts"][0]["files"])
    sizes = {name: rebuilt["artifacts"][0]["sizes"][name] for name in live_names}
    return {
        "served_files": live_names,
        "served_sizes": sizes,
        "served_bytes": sum(sizes.values()),
        "proof_input": hashlib.sha256(canonical(
            {"files": {name: rebuilt["artifacts"][0]["files"][name] for name in live_names},
             "sizes": sizes,
             "served_sizes": sizes}).encode()).hexdigest(),
        "lifetime_id": rebuilt["listener"]["lifetime_id"],
        "observed_unix": presented["byte_proof"]["observed_unix"],
    }


def build_witness(*, listener: Mapping[str, Any], launch: Mapping[str, Any],
                  artifacts: list, tokenizer: Mapping[str, Any]) -> dict:
    """Join four observed pieces into one witness, or refuse by name.

    Takes observations the producer read from a live serve, never a live
    rank or listener. A witness built from fixtures is a shape check that
    still needs :func:`stamp_byte_proof` against a served directory; it is
    never runtime evidence and :func:`verify_witness` refuses it as
    unverified until that stamp exists.
    """
    _check_listener(listener)
    _check_launch(launch)
    _refuse(isinstance(artifacts, list) and artifacts, "artifacts is not a non-empty list")
    for index, observed in enumerate(artifacts):
        _check_artifact(observed, f"artifacts[{index}]")
    _check_tokenizer(tokenizer)
    lifetime = listener["lifetime_id"]
    _refuse(launch["lifetime_id"] == lifetime, "launch lifetime differs from listener lifetime")
    _refuse(tokenizer["lifetime_id"] == lifetime, "tokenizer lifetime differs from listener lifetime")
    _check_lifetime(artifacts, lifetime, "artifacts")
    ranks = sorted(launch["ranks"])
    observed_ranks = sorted(observed["rank"] for observed in artifacts)
    _refuse(observed_ranks == ranks,
            f"artifact ranks {observed_ranks} do not cover launch ranks {ranks}")
    names = [sorted(observed["files"]) for observed in artifacts]
    _refuse(all(name == names[0] for name in names),
            "artifact ranks cover different file sets")
    for name in names[0]:
        digests = {observed["files"][name] for observed in artifacts}
        _refuse(len(digests) == 1, f"artifact file {name!r} digest differs across ranks")
        sizes = {observed["sizes"][name] for observed in artifacts}
        _refuse(len(sizes) == 1, f"artifact file {name!r} size differs across ranks")
    ordered = sorted(artifacts, key=lambda o: o["rank"])
    token_files = sorted(tokenizer["files"])
    witness = {
        "schema": SCHEMA,
        "listener": dict(listener),
        "launch": {"attempt_id": launch["attempt_id"], "ranks": ranks,
                   "lifetime_id": launch["lifetime_id"], "observed_unix": launch["observed_unix"]},
        "artifacts": [{**dict(observed),
                       "files": dict(observed["files"]), "sizes": dict(observed["sizes"])}
                      for observed in ordered],
        "tokenizer": {**dict(tokenizer),
                      "files": dict(tokenizer["files"]), "sizes": dict(tokenizer["sizes"])},
        "byte_coverage": {"files": names[0],
                          "sizes": [dict(observed["sizes"]) for observed in ordered],
                          "bytes_per_rank": [observed["bytes"] for observed in ordered],
                          "ranks": ranks},
        "qualification_scope": ("Loaded runtime state only. "
                                "No numerical, performance, or serving qualification follows. "
                                "A witness without a live byte proof qualifies nothing."),
    }
    witness["fingerprint"] = witness_fingerprint(witness)
    return witness


def verify_witness(witness: Mapping[str, Any], *, ranks: list | None = None) -> str | None:
    """Why this witness does not bind its runtime join, or None when it does.

    Never raises: malformed input is a refusal string, not an exception. A
    caller that names an expected rank set gets that check too; the join
    itself already refuses anything the launch record does not cover. A
    witness without a verified live byte proof never passes: fixture joins
    refuse as unverified, never as bound evidence.
    """
    try:
        rebuilt = build_witness(listener=witness["listener"], launch=witness["launch"],
                                artifacts=witness["artifacts"], tokenizer=witness["tokenizer"])
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        return f"malformed witness ({type(exc).__name__}: {exc})"
    if not isinstance(witness, Mapping):
        return "malformed witness (ValueError: witness is not an object)"
    if witness.get("schema") != SCHEMA:
        return f"witness schema {witness.get('schema')!r} is not {SCHEMA!r}"
    if witness.get("fingerprint") != rebuilt["fingerprint"]:
        return "witness fingerprint differs from its joined body"
    if witness.get("byte_coverage") != rebuilt["byte_coverage"]:
        return "witness byte_coverage differs from its joined body"
    proof = witness.get("byte_proof")
    if not isinstance(proof, Mapping):
        return "witness byte_proof holds no live byte proof; fixture joins qualify nothing"
    expected = _expected_proof(rebuilt, witness)
    for key in ("served_files", "served_sizes", "served_bytes", "proof_input", "lifetime_id"):
        if proof.get(key) != expected[key]:
            return f"witness byte_proof {key} differs from its joined body"
    if not isinstance(proof.get("observed_unix"), (int, float)) or proof["observed_unix"] <= 0:
        return "witness byte_proof observed_unix is not a positive time"
    if proof.get("lifetime_id") != rebuilt["listener"]["lifetime_id"]:
        return "witness byte_proof lifetime differs from listener lifetime"
    if ranks is not None and sorted(ranks) != rebuilt["launch"]["ranks"]:
        return f"witness ranks {rebuilt['launch']['ranks']} do not match expected {sorted(ranks)}"
    return None


__all__ = ["ARTIFACT_KEYS", "LAUNCH_KEYS", "LISTENER_KEYS", "SCHEMA", "TOKENIZER_KEYS",
           "VOCAB_SOURCES", "build_witness", "canonical", "prove_loaded_bytes",
           "stamp_byte_proof", "verify_witness", "witness_fingerprint"]
