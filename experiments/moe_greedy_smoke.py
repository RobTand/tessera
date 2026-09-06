#!/usr/bin/env python3
"""Greedy smoke on a served model, with an explicit degeneration rule (#198).

The routed-MoE cells' only recorded smoke was ``The capital of France is``
-> ``' France is France is ...'`` on BOTH the Tessera student and the BF16
source (``docs/measurements/moe-evidence-debt-2026-09-04.md`` section 7).
That control removed the evidence AGAINST the route without adding evidence
FOR it: no prompt on record shows the route generating coherently.  This
instrument runs a set of prompts against ONE served arm and writes a receipt
a second run on the other arm can be joined with, so the record is the pair.

THE RULE, stated so nobody has to read a completion and decide.  Greedy
decoding degenerates by entering a cycle: once the argmax sequence repeats a
state it repeats it until ``max_tokens``, and never leaves.  So the
signature is a cycle AT THE END of the completion.  For the completion's
tokens ``t[0..L-1]`` and every period ``p`` with ``2p <= L``, take the
longest suffix in which ``t[i] == t[i+p]`` throughout; it is a cycle only if
it holds at least two full periods (``s >= 2p``) -- one repeat is a
coincidence, two is the period observed.  The completion is ``repetitive``
when it ends in a cycle, whatever the cycle's share of the completion,
``not_recorded`` when it is EMPTY -- nothing is not a completion for a verdict
to be true of, and a positive record made of two non-answers is the hole #327
names -- and ``recorded`` otherwise.  The positive control is the v17 observation itself:
``' France is'`` x 8 is ``p=2, s=L=16``, and the rule must call it
``repetitive`` on both arms or the rule is wrong.  Tokens are the artifact's
own tokenizer's canonical encoding of the returned text (both arms ship
byte-identical ``tokenizer.json``), so the rule is reproducible from the
receipt alone; the receipt keeps the token ids, so a re-scoring is a re-read
and not a re-run.

Why no coverage threshold.  The first run of this instrument scored a tail
cycle only when it was a strict majority of the completion, and called the
student's P3 completion -- ``' trans'`` thirty-one times, from token 34 to
the 64-token cap -- ``recorded``.  A cycle that starts late is still a cycle
greedy decoding will not leave; where ``max_tokens`` cut it is not evidence
about the model.  Dropping the threshold moves the rule in the refusing
direction only: nothing it called ``repetitive`` before is ``recorded`` now.

What the rule does NOT decide: coherence.  A completion can be non-periodic
and still wrong (the same first run has a raw continuation that drifts into
Spanish and one that turns into assistant-style rambling).  The receipt
carries every completion verbatim so a reader can judge that, and the status
word rests on completions that pass BOTH the rule and that reading.

A prompt is either a raw ``prompt`` string, sent to ``/v1/completions`` in
the campaign's own request shape (``model, prompt, max_tokens, temperature:
0``; ``experiments/tessera_plugin_served.sh``), or a ``messages`` list, sent
to ``/v1/chat/completions`` so the serve applies the checkpoint's own
``chat_template.jinja``.  Both checkpoints are the instruct model (a chat
template, ``generation_config.json`` sampling defaults): a raw continuation
is not the interface the model was trained on, and the chat form is there so
the record has the interface it was.  Either way the serve's default sampling
parameters apply exactly as they did in the campaign.  On this model
the serve overrides its defaults from the checkpoint's ``generation_config
.json`` (``repetition_penalty: 1.05, temperature: 0.2, top_k: 80``, per the
campaign and control serve logs); ``temperature: 0`` in the request makes
the decode greedy, and the penalty stays in force.  ``--pure-greedy`` issues
each prompt a second time with ``repetition_penalty: 1.0`` and ``top_k: -1``
spelled out, recorded beside the primary form and never in its place.

usage: moe_greedy_smoke.py run --url URL --tokenizer DIR --prompts FILE
                               --arm NAME --out receipt.json [--pure-greedy]
       moe_greedy_smoke.py compare A.json B.json --out pair.json [--markdown pair.md]
                               [--subject ARM --reference bf16_source]

WHAT THE PAIR DECIDES, AND WHAT IT DOES NOT.  This file owns the
PER-COMPLETION rule above.  It does NOT own the AGGREGATION from a set of
completions to the one ``evidence.smoke.status`` word a serving cell
publishes: that lives in ``tessera.serving.contract.derive_smoke_status``,
because the contract is the thing a consumer reads and the wheel does not ship
``experiments/``.  With ``--subject`` the pair carries the
``evidence.smoke.record`` block a cell transcribes, and the word that function
reads off it, so the receipt and the contract cannot disagree about the word
without a test noticing (#327).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

# This checkout's ``src`` ahead of whatever Tessera the host carries (#341).
# The aggregation this file calls (``tessera.serving.contract``) must come from
# the SAME tree as this file, for the same reason
# ``experiments/runtime_image.sh`` resolves its root from ``BASH_SOURCE``
# instead of a caller's ``$TS``: selecting a checkout has to select the code
# that checkout's instrument runs, or the receipt records a tree commit that
# did not compute the word in it.  A wrapper cannot supply this by discipline
# -- it launches a fresh interpreter, and the one call site that forgot was
# reached only AFTER both expensive serves.  Resolved from ``__file__``, so a
# worktree binds to itself; ambient installs stay reachable behind it, which is
# what keeps ``tokenizers`` and ``requests`` resolving as before.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

#: The two request FORMS and the two model INTERFACES this instrument exercises.
#: They are vocabularies a contract cell's ``evidence.smoke.record`` quotes, so
#: they are declared here -- in the code that owns them -- and pinned against
#: ``tessera.serving.contract`` by ``tests/test_cell_evidence.py`` rather than
#: restated there.  A FORM is a sampling shape (``campaign`` is the v17 smoke's
#: request, byte for byte in what it sets; ``pure_greedy`` spells out
#: ``repetition_penalty 1.0, top_k -1`` on top of it).  An INTERFACE is which
#: model interface the prompt goes through, which is a different axis entirely:
#: a raw ``prompt`` to ``/v1/completions`` is a continuation the instruct
#: checkpoint was not trained on, and a ``messages`` list to
#: ``/v1/chat/completions`` is the serve applying the checkpoint's own
#: ``chat_template.jinja``.  Conflating the two is how "the status word is read
#: from the campaign form, because it is what the campaign measured" came to be
#: satisfied only by prompts sent through an interface the campaign never used.
FORMS = ("campaign", "pure_greedy")
INTERFACES = ("raw_completion", "chat_template")

#: This file's own repository path, which a contract cell's
#: ``evidence.smoke.record.instrument`` names: the rule and the scoring live
#: here, and a cell quotes them rather than restating them.
INSTRUMENT = "experiments/moe_greedy_smoke.py"

RULE = ("repetitive iff the completion ends in a cycle: some period p with 2p <= L "
        "has a p-periodic suffix holding >= 2 full periods (s >= 2p), whatever its "
        "share of the completion; not_recorded iff the completion is empty (L = 0), "
        "which is no completion for a verdict to be true of; recorded otherwise; "
        "tokens are the artifact tokenizer's canonical encoding of the returned text")


def periodic_tail(tokens):
    """The longest cycle at the end of ``tokens``: ``(period, suffix_len)``.

    ``suffix_len`` is the length of the longest suffix in which every token
    equals the one ``period`` places before it, counted only when it spans at
    least two full periods; ``(0, 0)`` when no period qualifies.  Ties on
    length go to the shortest period, which names the cycle rather than a
    multiple of it.
    """
    n = len(tokens)
    best = (0, 0)
    for p in range(1, n // 2 + 1):
        s = p  # the last p tokens trivially agree with themselves
        i = n - 1 - p
        while i >= 0 and tokens[i] == tokens[i + p]:
            s += 1
            i -= 1
        if s >= 2 * p and s > best[1]:
            best = (p, s)
    return best


def classify(tokens):
    """Apply the rule; return the verdict and the numbers it was read from.

    An EMPTY completion is ``not_recorded`` and never ``recorded`` (#327).
    ``recorded`` is a verdict about a completion -- "this one does not end in a
    cycle" -- and nothing is not a completion, so the cycle rule has no subject
    to be true of.  It mattered because the aggregation that decides a cell's
    ``evidence.smoke.status`` asks whether some prompt read ``recorded`` on
    BOTH arms: while ``classify([])`` returned ``recorded``, two arms that each
    returned ``""`` for one prompt manufactured a positive record out of two
    non-answers.  ``not_recorded`` is this vocabulary's own word for "no
    completion came back" (``tessera.serving.contract.EVIDENCE_SMOKE_STATUSES``)
    and is what an empty token list means.
    """
    n = len(tokens)
    period, suffix = periodic_tail(tokens)
    if not n:
        verdict = "not_recorded"
    else:
        verdict = "repetitive" if suffix else "recorded"
    return {"status": verdict, "tokens": n, "period": period,
            "periodic_suffix": suffix,
            "coverage": (suffix / n) if n else 0.0,
            "whole_completion_periodic": bool(n) and suffix == n}


def interface_of(entry):
    """Which model interface a prompt entry exercises.

    One home for the mapping ``cmd_run`` acts on: a ``messages`` entry goes to
    ``/v1/chat/completions`` and the serve applies the checkpoint's own chat
    template; anything else is a raw ``prompt`` continuation.
    """
    return "chat_template" if "messages" in entry else "raw_completion"


def _load_tokenizer(path):
    from tokenizers import Tokenizer

    tok_file = Path(path) / "tokenizer.json"
    return Tokenizer.from_file(str(tok_file)), hashlib.sha256(tok_file.read_bytes()).hexdigest()


def _post(url, body, timeout=600):
    import requests

    r = requests.post(url, json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def cmd_run(args):
    tokenizer, tok_sha = _load_tokenizer(args.tokenizer)
    prompts = json.loads(Path(args.prompts).read_text())
    extras = {"campaign": {}, "pure_greedy": {"repetition_penalty": 1.0, "top_k": -1}}
    forms = [(f, extras[f]) for f in FORMS if f == "campaign" or args.pure_greedy]
    results = []
    for entry in prompts["prompts"]:
        for form, extra in forms:
            body = {"model": args.model, "max_tokens": entry["max_tokens"], "temperature": 0, **extra}
            if "messages" in entry:
                # A chat prompt goes through the serve's own chat endpoint, which
                # applies the checkpoint's chat_template.jinja server-side: the
                # template is the model's, not this script's transcription of it.
                body["messages"] = entry["messages"]
                reply = _post(args.url.replace("/v1/completions", "/v1/chat/completions"), body)
                choice = reply["choices"][0]
                text = choice["message"]["content"]
            else:
                body["prompt"] = entry["prompt"]
                reply = _post(args.url, body)
                choice = reply["choices"][0]
                text = choice["text"]
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            verdict = classify(ids)
            cycle = tokenizer.decode(ids[len(ids) - verdict["period"]:]) if verdict["period"] else None
            record = {"id": entry["id"], "form": form, "interface": interface_of(entry),
                      "request": body,
                      "completion": text, "finish_reason": choice.get("finish_reason"),
                      "usage": reply.get("usage"), "token_ids": ids,
                      "cycle": cycle, **verdict}
            results.append(record)
            print(f"[{args.arm}] {entry['id']:>4} {form:<11} {verdict['status']:<10} "
                  f"L={verdict['tokens']:<3} p={verdict['period']:<2} s={verdict['periodic_suffix']:<3} "
                  f"{text!r}", flush=True)
    receipt = {"schema": "tessera.moe-greedy-smoke/1", "arm": args.arm,
               "recorded_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "url": args.url, "model": args.model,
               "tokenizer": {"path": str(args.tokenizer), "tokenizer_json_sha256": tok_sha},
               "prompts_file": str(args.prompts),
               "prompts_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
               "rule": RULE, "results": results}
    Path(args.out).write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"receipt -> {args.out}")


def _by_key(receipt):
    # Scored again from the stored token ids under the rule THIS file states,
    # so a pair joined after the rule changed reads under the rule it names;
    # the per-arm receipts keep the rule they were run under.
    return {(r["id"], r["form"]): {**r, **classify(r["token_ids"])} for r in receipt["results"]}


def _reference_arms():
    """The reference names the contract can read, from the contract itself.

    A roster restated here would pass on the day the contract's own list is
    wrong, so the choices come from the module that owns them; without tessera
    importable the flag simply offers the one reference that has been run.
    """
    try:
        from tessera.serving.contract import EVIDENCE_CONTROL_REFERENCES

        return EVIDENCE_CONTROL_REFERENCES
    except ImportError:  # pragma: no cover - a box without the package installed
        return ("bf16_source",)


def _interface_of_result(result):
    """The interface a scored result was taken on.

    Read from the result when the run recorded it, and otherwise from the
    request the run kept -- a ``messages`` body IS the chat endpoint -- so a
    pair joined from receipts written before the field existed still says which
    interface each row is, rather than leaving the axis unreadable.
    """
    return result.get("interface") or interface_of(result.get("request", {}))


def aggregation():
    """The contract's own status/attribution derivations, or a refusal (#341).

    ONE home for what the join needs, so the preflight below cannot drift from
    the import the join actually makes: a preflight that restates a list of
    symbols is a second rule, and the day they disagree the preflight passes a
    run that then fails after two serves.  Both callers take the same names
    from this one import.

    The refusal names the fix rather than the traceback, because the reader is
    a wrapper's log: the aggregation is `src/` code in the same checkout as
    this file and must be reachable from it.
    """
    try:
        from tessera.serving import contract

        expected = (Path(__file__).resolve().parents[1]
                    / "src/tessera/serving/contract.py").resolve()
        resolved = Path(contract.__file__).resolve()
        if resolved != expected:
            raise ImportError(
                f"contract resolved to {resolved}, expected this checkout's {expected}")
        from tessera.serving.contract import derive_smoke_attribution, derive_smoke_status
    except ImportError as exc:
        raise SystemExit(
            f"REFUSED: this instrument cannot import the contract's aggregation ({exc}). "
            f"tessera.serving.contract must come from the checkout holding {INSTRUMENT} "
            f"({Path(__file__).resolve().parents[1]}); it carries derive_smoke_status only "
            "since contract v22 (#327). A host install of an older Tessera does not supply "
            "it, and neither does the plugin inside the serve container.") from exc
    return derive_smoke_status, derive_smoke_attribution


def cmd_preflight(args):
    """Prove the join at the end of a run can be made, before the run starts.

    The wrapper serves both arms and only then joins them, so an unimportable
    aggregation used to cost two serves before it was noticed (#341).  This
    subcommand makes exactly the import the join makes and says where it
    resolved from, so a wrapper can refuse before it takes the box's serve
    lock -- the rule `experiments/runtime_image.sh` already states for the
    image pin.
    """
    aggregation()  # first, so an absent package refuses by name and not by traceback
    import tessera.serving.contract as module

    print(json.dumps({"ok": True, "instrument": INSTRUMENT,
                      "checkout": str(Path(__file__).resolve().parents[1]),
                      "contract_module": module.__file__,
                      "lane_eligibility_schema": module.LANE_ELIGIBILITY_SCHEMA}, indent=2))


def _contract_record(pair, args):
    """The ``evidence.smoke.record`` block a contract cell carries, or ``None``.

    Written only when the caller says WHICH arm the cell is about (``--subject``)
    and what the other arm is (``--reference``, a name from
    ``contract.EVIDENCE_CONTROL_REFERENCES``); the instrument does not guess
    which of two arms is the reference.  The block is the thing a cell
    transcribes verbatim, and ``derived`` beside it is what
    ``contract.derive_smoke_status`` / ``derive_smoke_attribution`` -- the one
    home of the aggregation -- read off it, so the receipt and the contract
    cannot disagree about the word without a test noticing.
    """
    if not args.subject:
        return None
    derive_smoke_status, derive_smoke_attribution = aggregation()
    arms = pair["arms"]
    if args.subject not in arms:
        sys.exit(f"REFUSED: --subject {args.subject!r} is not one of the arms {arms}")
    other = [arm for arm in arms if arm != args.subject]
    if len(other) != 1:
        sys.exit(f"REFUSED: the two receipts name one arm {arms}; a pair is two arms")
    record = {"instrument": INSTRUMENT, "rule": RULE, "reference": args.reference,
              "rows": [{"prompt": r["id"], "form": r["form"], "interface": r["interface"],
                        "status": r[args.subject]["status"],
                        "reference_status": (r[other[0]]["status"] if args.reference else None)}
                       for r in pair["rows"]]}
    smoke = {"record": record}
    return {"record": record,
            "derived": {"status": derive_smoke_status(smoke),
                        "attribution": derive_smoke_attribution(smoke)}}


def cmd_compare(args):
    a = json.loads(Path(args.a).read_text())
    b = json.loads(Path(args.b).read_text())
    if a["tokenizer"]["tokenizer_json_sha256"] != b["tokenizer"]["tokenizer_json_sha256"]:
        sys.exit("REFUSED: the two arms were tokenised with different tokenizer.json bytes")
    if a["prompts_sha256"] != b["prompts_sha256"]:
        sys.exit("REFUSED: the two arms answered different prompt files")
    ra, rb = _by_key(a), _by_key(b)
    if set(ra) != set(rb):
        sys.exit("REFUSED: the two arms answered different (prompt, form) sets")
    rows = []
    for key in sorted(ra, key=lambda k: (k[1] != "campaign", k[0])):
        x, y = ra[key], rb[key]
        rows.append({"id": key[0], "form": key[1], "interface": _interface_of_result(x),
                     "max_tokens": x["request"]["max_tokens"],
                     a["arm"]: {k: x[k] for k in ("status", "tokens", "period", "periodic_suffix",
                                                  "coverage", "finish_reason", "completion")},
                     b["arm"]: {k: y[k] for k in ("status", "tokens", "period", "periodic_suffix",
                                                  "coverage", "finish_reason", "completion")},
                     "identical_completion": x["completion"] == y["completion"],
                     "both_recorded": x["status"] == y["status"] == "recorded"})
    pair = {"schema": "tessera.moe-greedy-smoke-pair/2", "arms": [a["arm"], b["arm"]],
            "rule": RULE, "rule_at_run": sorted({a["rule"], b["rule"]}),
            "prompts_sha256": a["prompts_sha256"],
            "tokenizer_json_sha256": a["tokenizer"]["tokenizer_json_sha256"],
            "rows": rows,
            # Every (prompt, form) pair both arms answered without a cycle, in
            # EVERY form and on every interface.  It used to be filtered to the
            # campaign form "because that is what the campaign measured", which
            # was true of the sampling shape and false of the interface: the
            # prompts that satisfied it are chat turns the campaign never sent
            # (#327).  It is a REPORTED list either way; the word a cell
            # publishes is contract.derive_smoke_status over contract_record.
            "positive_record": [r["id"] for r in rows if r["both_recorded"]]}
    pair["contract_record"] = _contract_record(pair, args)
    Path(args.out).write_text(json.dumps(pair, indent=2) + "\n")
    if args.markdown:
        lines = [f"| prompt | form | max_tokens | {a['arm']} | {b['arm']} | identical |", "|---|---|---:|---|---|---|"]
        for r in rows:
            cell = lambda v: (f"`{v['status']}` L={v['tokens']} p={v['period']} s={v['periodic_suffix']} "
                              f"({v['finish_reason']})")
            lines.append(f"| {r['id']} | {r['form']} | {r['max_tokens']} | {cell(r[a['arm']])} | "
                         f"{cell(r[b['arm']])} | {'yes' if r['identical_completion'] else 'no'} |")
        Path(args.markdown).write_text("\n".join(lines) + "\n")
    derived = (pair["contract_record"] or {}).get("derived")
    print(json.dumps({"positive_record": pair["positive_record"], "derived": derived,
                      "rows": [(r["id"], r["form"], r["interface"], r[a["arm"]]["status"],
                                r[b["arm"]]["status"], r["identical_completion"])
                               for r in rows]}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--url", required=True)
    run.add_argument("--model", default="kl-target")
    run.add_argument("--tokenizer", required=True, help="directory holding tokenizer.json")
    run.add_argument("--prompts", required=True)
    run.add_argument("--arm", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--pure-greedy", action="store_true")
    run.set_defaults(func=cmd_run)
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("a")
    cmp_.add_argument("b")
    cmp_.add_argument("--out", required=True)
    cmp_.add_argument("--markdown")
    cmp_.add_argument("--subject", help="the arm a contract cell is about; with it the pair "
                                        "carries the evidence.smoke.record block and the word "
                                        "contract.derive_smoke_status reads off it")
    cmp_.add_argument("--reference", choices=list(_reference_arms()), default=None,
                      help="what the OTHER arm is, in the contract's own vocabulary")
    cmp_.set_defaults(func=cmd_compare)
    pre = sub.add_parser("preflight", help="refuse now if the join at the end of a run could "
                                           "not import the contract's aggregation")
    pre.set_defaults(func=cmd_preflight)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
