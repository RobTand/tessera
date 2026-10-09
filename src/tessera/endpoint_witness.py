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
here: the producer takes observed dicts, never a live rank or listener.

BYTE COVERAGE. Each rank observation names the files it covered and the
sha256 of each file's loaded bytes. Coverage is complete only when every
rank covers the same file set and the digests agree across ranks.

OBSERVATION LIFETIME. All observations carry a ``observed_unix`` stamp and a
``lifetime_id``. The join refuses observations from different lifetimes: a
byte observation from one serve and a listener observation from another is
two serves, not one witness.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

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
#: rank. ``files`` maps a file name to its loaded-byte ``sha256``. ``bytes``
#: is the total loaded bytes the rank observed.
ARTIFACT_KEYS = frozenset({"rank", "files", "bytes", "lifetime_id", "observed_unix"})

#: The fields a server tokenizer observation must carry. ``vocab_size`` is
#: the size the server reported. ``files`` maps a tokenizer file name to its
#: ``sha256`` as read from the served artifact. ``lifetime_id`` binds it to
#: the same serve.
TOKENIZER_KEYS = frozenset({"vocab_size", "files", "lifetime_id", "observed_unix"})

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
        _refuse(isinstance(name, str) and name and "\\" not in name,
                f"{where} file name {name!r} is not a relative name")
        _refuse(_is_hex64(digest), f"{where} file {name!r} sha256 is not a digest")
    _refuse(type(observed["bytes"]) is int and observed["bytes"] > 0,
            f"{where} bytes is not a positive count")
    _refuse(isinstance(observed["lifetime_id"], str) and observed["lifetime_id"],
            f"{where} lifetime_id is not a non-empty string")
    _refuse(isinstance(observed["observed_unix"], (int, float)) and observed["observed_unix"] > 0,
            f"{where} observed_unix is not a positive time")


def _check_tokenizer(tokenizer: Mapping[str, Any]) -> None:
    _require_keys(tokenizer, TOKENIZER_KEYS, "tokenizer")
    _refuse(type(tokenizer["vocab_size"]) is int and tokenizer["vocab_size"] > 0,
            "tokenizer vocab_size is not a positive count")
    files = tokenizer["files"]
    _refuse(isinstance(files, Mapping) and files,
            "tokenizer files is not a non-empty map")
    for name, digest in files.items():
        _refuse(isinstance(name, str) and name and "\\" not in name,
                f"tokenizer file name {name!r} is not a relative name")
        _refuse(_is_hex64(digest), f"tokenizer file {name!r} sha256 is not a digest")
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


def build_witness(*, listener: Mapping[str, Any], launch: Mapping[str, Any],
                  artifacts: list, tokenizer: Mapping[str, Any]) -> dict:
    """Join four observed pieces into one witness, or refuse by name.

    Pure: everything it needs has already been observed. That is what lets a
    CPU test with no GPU check the schema, the refusals and the join on
    fixture observations. A witness built from fixtures is a shape check,
    never runtime evidence.
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
    witness = {
        "schema": SCHEMA,
        "listener": dict(listener),
        "launch": {"attempt_id": launch["attempt_id"], "ranks": ranks,
                   "lifetime_id": launch["lifetime_id"], "observed_unix": launch["observed_unix"]},
        "artifacts": [dict(observed) for observed in sorted(artifacts, key=lambda o: o["rank"])],
        "tokenizer": dict(tokenizer),
        "byte_coverage": {"files": names[0],
                         "bytes_per_rank": [observed["bytes"] for observed in
                                            sorted(artifacts, key=lambda o: o["rank"])],
                         "ranks": ranks},
        "qualification_scope": ("Loaded runtime state only. "
                                "No numerical, performance, or serving qualification follows."),
    }
    witness["fingerprint"] = witness_fingerprint(witness)
    return witness


def verify_witness(witness: Mapping[str, Any], *, ranks: list | None = None) -> str | None:
    """Why this witness does not bind its runtime join, or None when it does.

    Never raises: malformed input is a refusal string, not an exception. A
    caller that names an expected rank set gets that check too; the join
    itself already refuses anything the launch record does not cover.
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
    if ranks is not None and sorted(ranks) != rebuilt["launch"]["ranks"]:
        return f"witness ranks {rebuilt['launch']['ranks']} do not match expected {sorted(ranks)}"
    return None


__all__ = ["ARTIFACT_KEYS", "LAUNCH_KEYS", "LISTENER_KEYS", "SCHEMA", "TOKENIZER_KEYS",
           "build_witness", "canonical", "verify_witness", "witness_fingerprint"]
