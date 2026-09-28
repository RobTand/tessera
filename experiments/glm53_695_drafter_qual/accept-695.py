"""tessera#695: drafter acceptance on a live serve, from vLLM's spec-decode counters.

Sends every prompt of an acceptance prompts file (schema
``prismaquant.pact_u4.accept_prompts/1``: token-id prompts, one max_tokens,
greedy) to /v1/completions one at a time, and reads the engine's
``vllm:spec_decode_*`` counters from /metrics before and after. The rates are
computed as the offline acceptance screen computes them, so a serve's figure
compares with that screen's:

  acceptance_rate           accepted draft tokens / proposed draft tokens
  mean_accepted_length      1 + accepted / drafts: tokens each verification step emits
  per_position_rates        accepted at draft position i / drafts

It also records each prompt's generated token ids, so two arms' outputs can be
compared, and the wall time of every request.

  accept-695.py PORT OUT ARM PROMPTS_JSON
Writes OUT/<arm>.accept.json; exit 0 when the counters were read and moved.
"""
import hashlib
import json
import pathlib
import re
import sys
import time
import urllib.request
import os

MODEL = os.environ.get("T508_MODEL", "glm53-stub")
SCHEMA = "prismaquant.pact_u4.accept_prompts/1"
COUNTERS = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
PER_POS = "num_accepted_tokens_per_pos"

port, out, arm, prompts_path = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
URL = f"http://127.0.0.1:{port}"
raw = pathlib.Path(prompts_path).read_bytes()
doc = json.loads(raw)
if doc.get("schema") != SCHEMA:
    raise SystemExit(f"{prompts_path}: schema {doc.get('schema')!r} is not {SCHEMA}")


def post(path, payload):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def spec_counters():
    """Sum every vllm:spec_decode_* counter over its label sets; per-position as a list."""
    with urllib.request.urlopen(URL + "/metrics", timeout=30) as r:
        text = r.read().decode()
    totals, names = {}, set()
    for line in text.splitlines():
        m = re.match(r'^(vllm:spec_decode_[a-z_]+?)(?:_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if not m:
            continue
        name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
        names.add(name)
        base = name[len("vllm:spec_decode_"):]
        if base == PER_POS:
            pos = re.search(r'position="(\d+)"', labels)
            if pos:
                vector = totals.setdefault(PER_POS, [])
                i = int(pos.group(1))
                vector.extend([0.0] * (i + 1 - len(vector)))
                vector[i] += value
        elif base in COUNTERS:
            totals[base] = totals.get(base, 0.0) + value
    return totals, sorted(names)


def settled():
    """Stats reach the API server after the step; read until two reads agree."""
    prev = spec_counters()
    for _ in range(20):
        time.sleep(0.25)
        cur = spec_counters()
        if cur == prev:
            return cur
        prev = cur
    return prev


before, names = settled()
t0 = time.time()
rows = []
for prompt in doc["prompts"]:
    r0 = time.perf_counter()
    res = post("/v1/completions", dict(model=MODEL, prompt=prompt["token_ids"],
                                       max_tokens=doc["max_tokens"], temperature=0,
                                       return_token_ids=True))
    choice = res["choices"][0]
    rows.append(dict(id=prompt["id"], wall_s=time.perf_counter() - r0,
                     token_ids=choice["token_ids"], finish_reason=choice.get("finish_reason")))
after, names_after = settled()
record = dict(arm=arm, prompts_file=dict(path=prompts_path, sha256=hashlib.sha256(raw).hexdigest()),
              max_tokens=doc["max_tokens"], prompts=len(rows), wall_s=time.time() - t0,
              present_metric_names=sorted(set(names) | set(names_after)),
              counters_before=before, counters_after=after, requests=rows,
              outputs_sha256=hashlib.sha256(json.dumps([r["token_ids"] for r in rows]).encode()).hexdigest())
missing = [k for k in COUNTERS if k not in before or k not in after]
if missing:
    record.update(status="not_measured", missing_counters=missing)
else:
    d = {k: after[k] - before[k] for k in COUNTERS}
    record["deltas"] = d
    if PER_POS in after:
        width = len(after[PER_POS])
        b = (before.get(PER_POS) or []) + [0.0] * width
        d[PER_POS] = [after[PER_POS][i] - b[i] for i in range(width)]
    if d["num_drafts"] <= 0 or d["num_draft_tokens"] <= 0:
        record.update(status="no_drafts")
    else:
        record.update(status="measured", acceptance_rate=d["num_accepted_tokens"] / d["num_draft_tokens"],
                      mean_accepted_length=1 + d["num_accepted_tokens"] / d["num_drafts"],
                      per_position_rates=[x / d["num_drafts"] for x in d.get(PER_POS, [])])
(out / f"{arm}.accept.json").write_text(json.dumps(record, indent=1))
print(arm, record["status"], {k: record.get(k) for k in
                              ("acceptance_rate", "mean_accepted_length", "per_position_rates")},
      "outputs", record["outputs_sha256"][:12], flush=True)
sys.exit(0 if record["status"] == "measured" else 1)
