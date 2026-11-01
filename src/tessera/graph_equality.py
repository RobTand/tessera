"""Authoritative short-context graph/eager equality population and receipt reader.

Long-context probes are screens, not members of this protocol. Importing this
module performs no IO and reads no environment or command-line configuration.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

TOPK = 20
LIMIT = 2048

# name -> (list of (length, offset), max_tokens)
CASES = {
    "b1_len1": ([(1, 0)], 32),
    "b1_len2": ([(2, 0)], 32),
    "b1_len5": ([(5, 0)], 32),
    "b1_len8": ([(8, 0)], 32),
    "b1_len17": ([(17, 0)], 32),
    "b1_len100": ([(100, 0)], 32),
    "b1_len500": ([(500, 3)], 32),
    "b1_len1500": ([(1500, 5)], 32),
    "b1_len2000": ([(2000, 7)], 32),
    "b2": ([(37, 11), (64, 13)], 24),
    "b3": ([(37, 11), (64, 13), (121, 17)], 24),
    "b4": ([(37, 11), (64, 13), (121, 17), (250, 19)], 24),
    "b5": ([(37, 11), (64, 13), (121, 17), (250, 19), (333, 23)], 24),
    "b6": ([(37, 11), (64, 13), (121, 17), (250, 19), (333, 23), (480, 29)], 24),
    "b7": ([(37, 11), (64, 13), (121, 17), (250, 19), (333, 23), (480, 29), (777, 31)], 24),
    "b8": ([(37, 11), (64, 13), (121, 17), (250, 19), (333, 23), (480, 29), (777, 31), (1024, 37)], 24),
    "rep_len17_a": ([(17, 0)], 32),
    "rep_len17_b": ([(17, 0)], 32),
    "rep_len1_a": ([(1, 0)], 32),
    "rep_len1_b": ([(1, 0)], 32),
}

CHOICES_PER_CASE = {name: len(specs) for name, (specs, _) in CASES.items()}
CHOICES_PER_PASS = sum(CHOICES_PER_CASE.values())


def digest(choice):
    """Legacy-compatible SHA256 of generated IDs and sorted top-logprob maps."""
    h = hashlib.sha256()
    h.update(json.dumps(choice["token_ids"]).encode())
    h.update(json.dumps(choice["logprobs"]["top_logprobs"], sort_keys=True).encode())
    return h.hexdigest()


def _require(condition, context, message):
    if not condition:
        raise ValueError(f"{context}: {message}")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _constant(value):
    raise ValueError(f"non-finite JSON constant {value}")


def _read(path):
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_object,
                           parse_constant=_constant)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path.name}: {exc}") from exc
    _require(isinstance(value, dict), path.name, "expected a JSON object")
    return value


def _finite(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _tokens(value, length, context, field):
    _require(isinstance(value, list) and len(value) == length,
             context, f"{field} must contain {length} tokens")
    _require(all(type(t) is int and t >= 0 for t in value),
             context, f"{field} must contain nonnegative integer token IDs")


def load_arm(directory, arm):
    """Read one complete, paused short-context equality pass.

    ``directory`` contains ``<arm>.eq.summary.json`` and exactly the raw case
    files prescribed by :data:`CASES`. ``arm`` may name the first or second
    pass; independence between passes/roles is the caller's responsibility.
    Returns ``{case_name: choices}`` in protocol order, each choice list sorted
    by its validated contiguous index. Raises ``ValueError`` on missing,
    extra, malformed or inconsistent evidence (including IO/JSON failures).

    Geometry comes from CASES, never from the submitted summary. Prompt IDs
    are checked for shape/type, not reconstructed: exact batch identity is
    checked by the membership predicate against the eager population. Long
    screens have distinct basenames and cannot satisfy this reader.
    """
    _require(isinstance(arm, str) and bool(arm) and arm not in (".", "..")
             and "/" not in arm and "\\" not in arm,
             "arm", "expected a nonempty basename")
    directory = Path(directory)
    summary = _read(directory / f"{arm}.eq.summary.json")
    _require(summary.get("_admission") == "paused", arm,
             "batches were not admitted paused")
    _require(set(summary) == set(CASES) | {"_admission"}, arm,
             "summary case population differs from authoritative CASES "
             f"(missing={sorted(set(CASES) - set(summary))}, "
             f"extra={sorted(set(summary) - set(CASES) - {'_admission'})})")
    expected = {f"{arm}.eq.{name}.json" for name in CASES}
    try:
        present = {p.name for p in directory.iterdir()
                   if p.name.startswith(f"{arm}.eq.") and p.name.endswith(".json")
                   and p.name != f"{arm}.eq.summary.json"}
    except OSError as exc:
        raise ValueError(f"{arm}: cannot enumerate raw population: {exc}") from exc
    _require(present == expected, arm,
             "raw case population differs from authoritative CASES "
             f"(missing={sorted(expected - present)}, extra={sorted(present - expected)})")
    cases = {}
    for name, (specs, max_tokens) in CASES.items():
        context = f"{arm}/{name}"
        row = summary[name]
        _require(isinstance(row, dict), context, "summary case must be an object")
        _require(type(row.get("batch")) is int and row["batch"] == len(specs),
                 context, "summary batch differs from protocol")
        lengths = row.get("lengths")
        _require(isinstance(lengths, list) and all(type(n) is int for n in lengths)
                 and lengths == [n for n, _ in specs], context,
                 "summary lengths differ from protocol")
        _require(type(row.get("max_tokens")) is int and row["max_tokens"] == max_tokens,
                 context, "summary max_tokens differs from protocol")
        digests = row.get("digests")
        _require(isinstance(digests, list) and len(digests) == len(specs),
                 context, "summary digest population differs from protocol")
        raw = _read(directory / f"{arm}.eq.{name}.json")
        choices = raw.get("choices")
        _require(isinstance(choices, list) and len(choices) == len(specs),
                 context, "raw choice population differs from protocol")
        _require(all(isinstance(c, dict) and type(c.get("index")) is int
                     for c in choices), context, "choice indices must be integers")
        _require(sorted(c["index"] for c in choices) == list(range(len(specs))),
                 context, "choice indices must be unique and contiguous from zero")
        choices = sorted(choices, key=lambda c: c["index"])
        for index, (choice, (length, _)) in enumerate(zip(choices, specs)):
            where = f"{context}/choice {index}"
            _tokens(choice.get("prompt_token_ids"), length, where, "prompt_token_ids")
            _tokens(choice.get("token_ids"), max_tokens, where, "token_ids")
            logprobs = choice.get("logprobs")
            _require(isinstance(logprobs, dict), where, "logprobs must be an object")
            values = logprobs.get("token_logprobs")
            _require(isinstance(values, list) and len(values) == max_tokens
                     and all(_finite(v) for v in values), where,
                     "token_logprobs must be finite and match generation length")
            tops = logprobs.get("top_logprobs")
            _require(isinstance(tops, list) and len(tops) == max_tokens, where,
                     "top_logprobs must match generation length")
            # The runner requests top K, but not return_tokens_as_token_ids:
            # distinct IDs can decode to the same text key. Validate the wire
            # map, not a lossless token count it cannot represent. vLLM may
            # also include the sampled token in addition to the top K.
            _require(all(isinstance(top, dict) and 1 <= len(top) <= TOPK + 1
                         and all(_finite(v) for v in top.values()) for top in tops),
                     where, "top_logprobs must contain nonempty finite maps of at most top-K + 1 keys")
            _require(digests[index] == digest(choice), where,
                     "summary digest does not match raw choice")
        cases[name] = choices
    return cases

