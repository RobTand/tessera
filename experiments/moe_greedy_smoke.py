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
when it ends in a cycle, whatever the cycle's share of the completion, and
``recorded`` otherwise.  The positive control is the v17 observation itself:
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
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

RULE = ("repetitive iff the completion ends in a cycle: some period p with 2p <= L "
        "has a p-periodic suffix holding >= 2 full periods (s >= 2p), whatever its "
        "share of the completion; tokens are the artifact tokenizer's canonical "
        "encoding of the returned text")


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
    """Apply the rule; return the verdict and the numbers it was read from."""
    n = len(tokens)
    period, suffix = periodic_tail(tokens)
    verdict = "repetitive" if suffix else "recorded"
    return {"status": verdict, "tokens": n, "period": period,
            "periodic_suffix": suffix,
            "coverage": (suffix / n) if n else 0.0,
            "whole_completion_periodic": bool(n) and suffix == n}


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
    forms = [("campaign", {})]
    if args.pure_greedy:
        forms.append(("pure_greedy", {"repetition_penalty": 1.0, "top_k": -1}))
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
            record = {"id": entry["id"], "form": form, "request": body,
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
        rows.append({"id": key[0], "form": key[1],
                     "max_tokens": x["request"]["max_tokens"],
                     a["arm"]: {k: x[k] for k in ("status", "tokens", "period", "periodic_suffix",
                                                  "coverage", "finish_reason", "completion")},
                     b["arm"]: {k: y[k] for k in ("status", "tokens", "period", "periodic_suffix",
                                                  "coverage", "finish_reason", "completion")},
                     "identical_completion": x["completion"] == y["completion"],
                     "both_recorded": x["status"] == y["status"] == "recorded"})
    pair = {"schema": "tessera.moe-greedy-smoke-pair/1", "arms": [a["arm"], b["arm"]],
            "rule": RULE, "rule_at_run": sorted({a["rule"], b["rule"]}),
            "prompts_sha256": a["prompts_sha256"],
            "tokenizer_json_sha256": a["tokenizer"]["tokenizer_json_sha256"],
            "rows": rows,
            "positive_record": [r["id"] for r in rows if r["form"] == "campaign" and r["both_recorded"]]}
    Path(args.out).write_text(json.dumps(pair, indent=2) + "\n")
    if args.markdown:
        lines = [f"| prompt | form | max_tokens | {a['arm']} | {b['arm']} | identical |", "|---|---|---:|---|---|---|"]
        for r in rows:
            cell = lambda v: (f"`{v['status']}` L={v['tokens']} p={v['period']} s={v['periodic_suffix']} "
                              f"({v['finish_reason']})")
            lines.append(f"| {r['id']} | {r['form']} | {r['max_tokens']} | {cell(r[a['arm']])} | "
                         f"{cell(r[b['arm']])} | {'yes' if r['identical_completion'] else 'no'} |")
        Path(args.markdown).write_text("\n".join(lines) + "\n")
    print(json.dumps({"positive_record": pair["positive_record"],
                      "rows": [(r["id"], r["form"], r[a["arm"]]["status"], r[b["arm"]]["status"],
                                r["identical_completion"]) for r in rows]}, indent=1))


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
    cmp_.set_defaults(func=cmd_compare)
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
