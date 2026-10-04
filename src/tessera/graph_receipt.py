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
each captured decode size at least once, and, when Tessera captured per
``max_seq_len`` class (``glm53_graphs``), each class it captured.  A pass
count without replays reads graphs that never ran.

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

Schema ``tessera.graph_equals_eager.v1`` (JSON object):

- ``schema``, ``verdict`` (``equal`` | ``not_equal``), ``issue``;
- ``runtime``: ``image`` (by digest), ``vllm``, ``interface`` (the
  ``glm53_graphs`` inspected interface name) ;
- ``tessera``: ``commit``, ``src_sha256``;
- ``model``: ``path``, ``config_sha256``, ``index_topk``;
- ``equality_set``: ``name``, ``script_sha256``, ``choices_per_pass``;
- ``eager_pool``: list of arm records;
- ``arms``: list of graph arm records, each ``{name, pb_action, host,
  compilation_config, speculative_tokens, max_model_len, max_num_seqs,
  tensor_parallel_size, resolved: {custom_ops, ir_op_priority},
  graph: {managers: {<manager class>: {captured_sizes, replayed_sizes}},
  classes: {captured, replays}},
  passes: [{name, members, choices}], replayed_everything, equal}``;
- ``attests``: one entry per ``equal`` arm, the scope :func:`verify` matches;
- ``screens``, ``not_measured``.
"""
from __future__ import annotations

import json
from typing import Any

SCHEMA = "tessera.graph_equals_eager.v1"

#: The serve fields an attestation is scoped to; :func:`verify` matches each exactly.
SCOPE_FIELDS = ("image", "model_config_sha256", "tessera_src_sha256", "compilation_config",
                "speculative_tokens", "max_model_len", "max_num_seqs", "tensor_parallel_size")


def canonical(value: Any) -> str:
    """One spelling per JSON value, so two equal configs compare equal."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def arm_equal(arm: dict) -> bool:
    """The rule: every choice of every pass a member, and every captured graph replayed."""
    passes = arm.get("passes") or []
    return (len(passes) >= 2
            and all(p["choices"] > 0 and p["members"] == p["choices"] for p in passes)
            and bool(arm.get("replayed_everything")))


def replayed_everything(graph: dict) -> bool:
    """Each manager's captured decode sizes replayed at least once, and each captured class too."""
    managers = graph.get("managers") or {}
    if not managers:
        return False
    for record in managers.values():
        sizes = record.get("captured_sizes") or []
        replayed = {int(k): v for k, v in (record.get("replayed_sizes") or {}).items()}
        if not sizes or any(replayed.get(int(s), 0) < 1 for s in sizes):
            return False
    classes = graph.get("classes") or {}
    captured = classes.get("captured") or {}
    replays = classes.get("replays") or {}
    for manager, by_bound in captured.items():
        for bound in by_bound:
            if replays.get(f"{manager}|{bound}", 0) < 1:
                return False
    return True


def attestation(receipt: dict, arm: dict) -> dict:
    """The scope one equal arm attests."""
    return {"execution_mode": "compiled", "image": receipt["runtime"]["image"],
            "model_config_sha256": receipt["model"]["config_sha256"],
            "tessera_src_sha256": receipt["tessera"]["src_sha256"],
            "compilation_config": arm["compilation_config"],
            "speculative_tokens": arm["speculative_tokens"],
            "max_model_len": arm["max_model_len"], "max_num_seqs": arm["max_num_seqs"],
            "tensor_parallel_size": arm["tensor_parallel_size"], "arm": arm["name"]}


def finish(receipt: dict) -> dict:
    """Set each arm's ``equal``, the ``attests`` list and the ``verdict`` from the rule."""
    for arm in receipt["arms"]:
        arm["replayed_everything"] = replayed_everything(arm["graph"])
        arm["equal"] = arm_equal(arm)
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
    if receipt.get("schema") != SCHEMA:
        return f"schema {receipt.get('schema')!r} is not {SCHEMA}"
    missing = [f for f in SCOPE_FIELDS if f not in serve]
    if missing:
        return f"the serve does not name {missing}"
    rederived = finish(json.loads(json.dumps(receipt)))
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


__all__ = ["SCHEMA", "SCOPE_FIELDS", "arm_equal", "attestation", "canonical", "finish",
           "replayed_everything", "verify"]
