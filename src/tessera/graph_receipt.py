"""The "graph equals eager" receipt: what a CUDA-graph serve is attested to compute (tessera#702).

A ship card's ``native_export.graph`` slot (PrismaQuant ``shipcard.py``) needs
more than "loads and generates with graphs on": the graph serve must compute
what the eager serve computes, because every cell, census and KL receipt was
measured eager.  This module is the one home of the receipt that says so: its
schema, the rule that makes it ``equal``, and :func:`verify`, the check a
consumer runs against the serve it is about to publish.  A consumer calls
:func:`verify`; it does not re-derive the rule.

THE MEASUREMENT.  One eager pool and one or more graph arms, each a serve of
the same model, image, Tessera tree and serve settings except the execution
mode, run the tessera#508 equality set
(``experiments/glm53_508_graph_qual/equal-508.py``): greedy completions with
top-20 logprobs, batch 1 at prompt lengths 1..2000 and batches 2..8 admitted
paused, 48 choices per pass, two passes per serve.  A graph choice is a
MEMBER when its token ids and every top-20 list are bit-identical to the same
choice of some eager run of the same batch.  An arm is ``equal`` when every
choice of every pass is a member AND the serve replayed what it captured:
each actual FULL descriptor (mode, tokens, num_reqs, max_seq_len) has a
successful replay return for its process-local owner on every required rank.
Dispatch counts and size/class marginals cannot supply this proof; successful
returns are host observations, not evidence of device completion.

THE SCOPE.  An arm attests exactly the serve it ran: image, model (by config
digest), Tessera source digest, ``compilation_config`` as given, speculative
token count, ``max_model_len``, ``max_num_seqs`` and tensor-parallel size.
:func:`verify` matches every one of them exactly; nothing is extrapolated
from a stub to an artifact, from TP 1 to TP 2, or from one ``max_model_len``
to another.

THE SCREEN.  The equality set stops at 2048 tokens by construction: above
GLM-5.3's ``index_topk`` the stock top-k writes its selection in arrival
order and eager does not reproduce itself.  The long-context class is
therefore recorded as a ``screens`` entry (membership against the eager pool's
outcomes), never as part of the verdict.

THE FABRIC (v2, CEO decision dec-1005-003356-6ba2).  A tensor-parallel serve
reduces across the two Sparks over sockets or over RoCE, and the two are two
all-reduces: an eager pool measured on one does not judge a graph arm, or a
card, on the other.  The fabric is therefore scope, read from what NCCL
reported on every rank: ``socket`` or ``roce`` at tensor parallel 2 or more,
``none`` for a one-rank serve, which reduces across nothing.  A v1 receipt
(no fabric) stays readable -- :func:`finish` still re-derives it -- but
:func:`verify` refuses it for a card, whose fabric it cannot attest.

Schema ``tessera.graph_equals_eager.v2`` (JSON object):

- ``schema``, ``verdict`` (``equal`` | ``not_equal``), ``issue``;
- ``runtime``: ``image`` (by digest), ``fabric`` (``socket`` | ``roce`` |
  ``none``), ``vllm``, ``interface`` (the ``glm53_graphs`` inspected
  interface name);
- ``tessera``: ``commit``, ``src_sha256``;
- ``model``: ``path``, ``config_sha256``, ``index_topk``;
- ``equality_set``: ``name``, ``script_sha256``, ``choices_per_pass``;
- ``eager_pool``: list of arm records;
- ``arms``: list of graph arm records, each ``{name, pb_action, host,
  compilation_config, speculative_tokens, max_model_len, max_num_seqs,
  tensor_parallel_size, resolved: {custom_ops, ir_op_priority},
  graph: {managers: {<manager class>: {captured_sizes, replayed_sizes}},
  classes: {captured, replays}, dispatches: {<original basename>: <dispatch record>}},
  where each dispatch retains pid/captured/counts diagnostics and owner_evidence:
  {<Class@instance-hex>: {capture_complete, capture_error, captured: [descriptor],
  replays: [descriptor with count], replay_failures?: [descriptor with error/count]}},
  passes: [{name, members, choices}], replayed_everything, equal}``;
- ``attests``: one entry per ``equal`` arm, the scope :func:`verify` matches;
- ``screens``, ``not_measured``.
"""
from __future__ import annotations

import json
import re
from typing import Any

from tessera.graph_equality import CHOICES_PER_PASS

SCHEMA = "tessera.graph_equals_eager.v2"
#: The schema before the fabric was scope: readable, never a card's attestation.
SCHEMA_V1 = "tessera.graph_equals_eager.v1"

#: The serve fields an attestation is scoped to; :func:`verify` matches each exactly.
SCOPE_FIELDS = ("image", "model_config_sha256", "tessera_src_sha256", "compilation_config",
                "speculative_tokens", "max_model_len", "max_num_seqs", "tensor_parallel_size",
                "fabric")

#: The fabrics a tensor-parallel serve reduces over; ``none`` is the one-rank serve's.
FABRICS = ("socket", "roce")
NO_FABRIC = "none"


def fabric_refusal(fabric: Any, tensor_parallel_size: Any) -> str | None:
    """Why ``fabric`` cannot be this serve's, or None: socket/roce above one rank, none at one."""
    if tensor_parallel_size == 1:
        return None if fabric == NO_FABRIC else (
            f"fabric {fabric!r} at tensor_parallel_size 1, which reduces across nothing ({NO_FABRIC!r})")
    if fabric not in FABRICS:
        return (f"fabric {fabric!r} at tensor_parallel_size {tensor_parallel_size}: "
                f"a tensor-parallel serve names one of {list(FABRICS)}")
    return None


def canonical(value: Any) -> str:
    """One spelling per JSON value, so two equal configs compare equal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def names_refusal(eager: list[str], graph: list[str]) -> str | None:
    """Roles and their two expanded pass identities must be independent."""
    if not eager or not graph:
        return "eager and graph populations must both be present"
    names = eager + graph
    if any(not isinstance(n, str) or not n for n in names):
        return "arm names must be nonempty strings"
    expanded = [p for n in names for p in (n, f"{n}-r2")]
    if len(set(expanded)) != len(expanded):
        return "duplicate or overlapping arm roles or expanded pass names"
    return None


def complete_passes(arm: dict) -> bool:
    """Exactly the two required passes, with integral complete population counters.

    Eager membership may vary: completeness is not eager repeat-exactness.
    """
    passes = arm.get("passes") or []
    name = arm.get("name")
    return (len(passes) == 2
            and {p.get("name") for p in passes} == {name, f"{name}-r2"}
            and all(type(p.get("choices")) is int
                    and p["choices"] == CHOICES_PER_PASS
                    and type(p.get("members")) is int
                    and 0 <= p["members"] <= CHOICES_PER_PASS for p in passes))


def population_refusal(receipt: dict) -> str | None:
    """The shared population gate, independent of declared verdicts and totals."""
    eager, graph = receipt.get("eager_pool") or [], receipt.get("arms") or []
    why = names_refusal([a["name"] for a in eager], [a["name"] for a in graph])
    if why:
        return why
    total = receipt.get("equality_set", {}).get("choices_per_pass")
    if type(total) is not int or total != CHOICES_PER_PASS:
        return "equality_set choices_per_pass contradicts the authoritative protocol"
    for arm in eager + graph:
        if not complete_passes(arm):
            return f"malformed population for arm {arm['name']}: incomplete or invalid required equality passes"
    return None


def arm_equal(arm: dict) -> bool:
    """Every authoritative choice in both required passes is a member, with replay."""
    return (complete_passes(arm)
            and all(p["members"] == CHOICES_PER_PASS for p in arm["passes"])
            and bool(arm.get("replayed_everything")))


def _integer(value: Any, *, positive: bool = False) -> None:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError("expected a positive integer" if positive else "expected a nonnegative integer")


def _descriptor(record: dict) -> tuple:
    if not isinstance(record, dict) or record.get("mode") != "FULL":
        raise ValueError("expected a FULL descriptor")
    for field in ("tokens", "num_reqs"):
        _integer(record.get(field), positive=True)
    if "max_seq_len" not in record:
        raise ValueError("missing max_seq_len")
    if record["max_seq_len"] is not None:
        _integer(record["max_seq_len"], positive=True)
    return tuple(record[f] for f in ("mode", "tokens", "num_reqs", "max_seq_len"))


def _size_key(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise ValueError("invalid size key")


def validate_dispatch_diagnostics(record: dict) -> None:
    """Validate even unused historical counters; these never supply replay credit."""
    for count in record.get("counts", {}).values():
        _integer(count)
    for owner, sizes in record["captured"].items():
        if not isinstance(owner, str) or not isinstance(sizes, list):
            raise ValueError("invalid capture diagnostic")
        for size in sizes:
            _integer(size, positive=True)
    classes = record.get("tessera_classes", {})
    for by_bound in classes.get("captured", {}).values():
        for bound, count in by_bound.items():
            _size_key(bound)
            _integer(count)
    for count in classes.get("replays", {}).values():
        _integer(count)


def replay_refusal(graph: dict, name: str | None = None,
                   tensor_parallel_size: int | None = None) -> str | None:
    """Owner-local descriptor-exact successful-return coverage, never marginal coverage.

    finish supplies the authoritative arm and TP scope. The optional arguments
    retain the historical graph-only diagnostic helper calling convention.
    """
    try:
        # Validate additive aggregate diagnostics too, including unused entries.
        for manager in graph.get("managers", {}).values():
            for size in manager["captured_sizes"]:
                _integer(size, positive=True)
            for size, count in manager["replayed_sizes"].items():
                if type(size) is int:
                    _integer(size, positive=True)
                else:
                    _size_key(size)
                _integer(count)
        classes = graph.get("classes", {})
        for bounds in classes.get("captured", {}).values():
            for bound, count in bounds.items():
                _size_key(bound)
                _integer(count)
        for count in classes.get("replays", {}).values():
            _integer(count)
        dispatches = graph["dispatches"]
        if not isinstance(dispatches, dict) or not dispatches:
            return "missing owner-local dispatch evidence"
        ranks = set()
        inferred_names = set()
        for basename, record in dispatches.items():
            match = re.fullmatch(r"(.+?)(?:\.rank([0-9]+))?\.dispatch\.([0-9]+)\.json", basename)
            if not match or (name is not None and match[1] != name):
                return "invalid dispatch arm/rank basename"
            inferred_names.add(match[1])
            rank = match[2]
            ranks.add(rank)
            if not isinstance(record, dict):
                return "malformed dispatch record"
            _integer(record.get("pid"), positive=True)
            captured = record["captured"]
            owners = record["owner_evidence"]
            if not isinstance(captured, dict) or not isinstance(owners, dict) or not owners:
                return "missing captured owner inventory"
            if set(captured) != set(owners):
                return "unknown or missing captured owner mapping"
            validate_dispatch_diagnostics(record)
            for owner, evidence in owners.items():
                if not isinstance(owner, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*@[0-9a-fA-F]+", owner):
                    return "invalid process-local owner key"
                if evidence.get("capture_complete") is not True or evidence.get("capture_error") is not None:
                    return "failed or incomplete owner capture"
                if "capture_error" not in evidence:
                    return "missing owner capture status"
                inventory, replays = evidence["captured"], evidence["replays"]
                if not isinstance(inventory, list) or not inventory or not isinstance(replays, list):
                    return "missing owner descriptor inventory or successful returns"
                required = {_descriptor(d) for d in inventory}
                if len(required) != len(inventory):
                    return "duplicate capture descriptor"
                if {d[1] for d in required} != set(captured[owner]):
                    return "captured size diagnostic contradicts owner inventory"
                successful = set()
                for replay in replays:
                    descriptor = _descriptor(replay)
                    _integer(replay.get("count"))
                    if replay["count"] > 0:
                        successful.add(descriptor)
                failures = evidence.get("replay_failures", [])
                if not isinstance(failures, list):
                    return "malformed replay failures"
                for failure in failures:
                    _descriptor(failure)
                    _integer(failure.get("count"))
                    if not isinstance(failure.get("error"), str) or not failure["error"]:
                        return "missing replay failure error"
                if not required <= successful:
                    return "captured owner descriptor has no successful replay return"
        if len(inferred_names) != 1:
            return "dispatch records mix arm populations"
        if tensor_parallel_size is None:
            tensor_parallel_size = 1 if ranks == {None} else len(ranks)
        _integer(tensor_parallel_size, positive=True)
        want = {None} if tensor_parallel_size == 1 else {str(r) for r in range(tensor_parallel_size)}
        if ranks != want:
            return "dispatch evidence does not cover exactly the required ranks"
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError) as exc:
        return f"malformed replay evidence ({type(exc).__name__}: {exc})"
    return None


def replayed_everything(graph: dict) -> bool:
    """Whether the recorded dispatch population covers every actual owner descriptor."""
    return replay_refusal(graph) is None


def attestation(receipt: dict, arm: dict) -> dict:
    """The scope one equal arm attests."""
    return {"execution_mode": "compiled", "image": receipt["runtime"]["image"],
            "model_config_sha256": receipt["model"]["config_sha256"],
            "tessera_src_sha256": receipt["tessera"]["src_sha256"],
            "compilation_config": arm["compilation_config"],
            "speculative_tokens": arm["speculative_tokens"],
            "max_model_len": arm["max_model_len"], "max_num_seqs": arm["max_num_seqs"],
            "tensor_parallel_size": arm["tensor_parallel_size"],
            "fabric": receipt["runtime"].get("fabric"), "arm": arm["name"]}


def finish(receipt: dict) -> dict:
    """Set each arm's ``equal``, the ``attests`` list and the ``verdict`` from the rule."""
    why = population_refusal(receipt)
    for arm in receipt["arms"]:
        arm["replayed_everything"] = replay_refusal(
            arm["graph"], arm["name"], arm["tensor_parallel_size"]) is None
        arm["equal"] = why is None and arm_equal(arm)
    receipt["attests"] = [attestation(receipt, a) for a in receipt["arms"] if a["equal"]]
    receipt["verdict"] = ("equal" if receipt["arms"] and all(a["equal"] for a in receipt["arms"])
                          else "not_equal")
    return receipt


def verify(receipt: dict, serve: dict) -> str | None:
    """Why this receipt does not attest ``serve`` as graph-equals-eager, or None when it does.

    ``serve`` names every :data:`SCOPE_FIELDS` value of the serve the consumer
    is about to publish.  The receipt's own rule is re-applied, so a receipt
    edited to say ``equal`` without equal arms is refused.
    """
    if isinstance(receipt, dict) and receipt.get("schema") == SCHEMA_V1:
        return (f"a {SCHEMA_V1} receipt names no fabric, so it cannot attest a card's serve; "
                f"produce a {SCHEMA} receipt")
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        schema = receipt.get("schema") if isinstance(receipt, dict) else type(receipt).__name__
        return f"schema {schema!r} is not {SCHEMA}"
    try:
        return _verify_serve(receipt, serve)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError, OverflowError) as exc:
        return f"malformed receipt or serve ({type(exc).__name__}: {exc})"


def _verify_serve(receipt: dict, serve: dict) -> str | None:
    missing = [f for f in SCOPE_FIELDS if f not in serve]
    if missing:
        return f"the serve does not name {missing}"
    why = fabric_refusal(serve["fabric"], serve["tensor_parallel_size"])
    if why:
        return f"the serve's {why}"
    try:
        return _verify(receipt, serve)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        # A receipt the rule cannot read attests nothing: a reason, never an exception
        # a caller might not treat as a refusal.
        return f"malformed receipt ({type(exc).__name__}: {exc})"


def _verify(receipt: dict, serve: dict) -> str | None:
    why = population_refusal(receipt)
    if why:
        return why
    rederived = finish(json.loads(json.dumps(receipt)))
    for entry in rederived["attests"]:
        why = fabric_refusal(entry["fabric"], entry["tensor_parallel_size"])
        if why:
            return f"the receipt's arm {entry['arm']}: {why}"
    if rederived["verdict"] != receipt.get("verdict") or rederived["verdict"] != "equal":
        bad = [a["name"] for a in rederived["arms"] if not a["equal"]]
        return f"verdict is {rederived['verdict']!r} by the rule (arms not equal: {bad})"
    want = {f: canonical(serve[f]) for f in SCOPE_FIELDS}
    for entry in rederived["attests"]:
        if all(canonical(entry[f]) == want[f] for f in SCOPE_FIELDS):
            return None
    nearest = min(rederived["attests"],
                  key=lambda e: sum(canonical(e[f]) != want[f] for f in SCOPE_FIELDS))
    differ = [f for f in SCOPE_FIELDS if canonical(nearest[f]) != want[f]]
    return (f"no attested arm has this serve's scope; nearest ({nearest['arm']}) differs in "
            f"{differ}")


__all__ = ["FABRICS", "NO_FABRIC", "SCHEMA", "SCHEMA_V1", "SCOPE_FIELDS", "arm_equal",
           "attestation", "canonical", "fabric_refusal", "finish", "replayed_everything", "verify"]
