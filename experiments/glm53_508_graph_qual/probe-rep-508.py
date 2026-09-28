"""tessera#508 diagnostic: the same batch-1 request, repeated inside one serve.

Two questions the equality suite cannot separate: does eager return the same
result for the same request (a) later in the same serve, after other requests
have used the engine's state, and (b) in a fresh serve. This probe sends each
exact-length token-id prompt (default lengths 1, 17 and 2000) REPEATS times back
to back, then the first length again REPEATS times after the longest prompt, all
greedy with top-20 logprobs and 16 generated tokens. Every request is recorded;
the summary names, per request, whether it equals the first request of its
length, the first differing generated position and the largest top-20 logprob
difference over tokens both results list. Records from two serves compare with
equal-compare-style tools on the per-request files.

With a DIGEST=1 serve, the requests are sequential, so the digest file splits
into one prefill line (real = prompt length) plus 15 decode lines per request,
in the order printed here (see digest-rep-compare-508.py).

  probe-rep-508.py PORT OUT ARM [LENGTHS] [REPEATS]
"""
import json
import math
import os
import pathlib
import sys
import urllib.request

MODEL = os.environ.get("T508_MODEL", "glm53-stub")
TOPK = 20
port, out, arm = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
lengths = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,17,2000").split(",")]
repeats = int(sys.argv[5]) if len(sys.argv) > 5 else 3
out.mkdir(parents=True, exist_ok=True)


def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


para = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "seventeen curious ravens observe the scene from a weathered oak branch. ")
ids = post("/tokenize", dict(model=MODEL, prompt=para * 200, add_special_tokens=False))["tokens"]
assert len(ids) >= max(lengths), (len(ids), max(lengths))


def run(n):
    res = post("/v1/completions", dict(model=MODEL, prompt=ids[:n], max_tokens=16, temperature=0,
                                       logprobs=TOPK, return_token_ids=True))
    ch = res["choices"][0]
    assert res["usage"]["prompt_tokens"] == n, (n, res["usage"])
    assert all(math.isfinite(v) for v in ch["logprobs"]["token_logprobs"])
    return res


def compare(a, b):
    ca, cb = a["choices"][0], b["choices"][0]
    first = next((i for i, (x, y) in enumerate(zip(ca["token_ids"], cb["token_ids"])) if x != y), None)
    dmax = 0.0
    for ta, tb in zip(ca["logprobs"]["top_logprobs"], cb["logprobs"]["top_logprobs"]):
        for tok in set(ta) & set(tb):
            dmax = max(dmax, abs(ta[tok] - tb[tok]))
    return dict(equal=(ca["token_ids"] == cb["token_ids"]
                       and ca["logprobs"]["top_logprobs"] == cb["logprobs"]["top_logprobs"]),
                first_token_diff=first, max_shared_top20_delta=dmax)


plan = [(n, "p0") for n in lengths] + [(lengths[0], "p1")]
summary = {"order": [], "results": {}}
firsts = {}
for n, tag in plan:
    for i in range(repeats):
        name = f"len{n}.{tag}.r{i}"
        res = run(n)
        (out / f"{arm}.rep.{name}.json").write_text(json.dumps(res, indent=1))
        cmp = compare(firsts[n], res) if n in firsts else dict(equal=True, first_token_diff=None,
                                                               max_shared_top20_delta=0.0)
        firsts.setdefault(n, res)
        summary["order"].append(dict(name=name, prompt_tokens=n))
        summary["results"][name] = cmp
        print(f"{name}: {'equal to first' if cmp['equal'] else 'DIFFERS from first'} "
              f"first_token_diff={cmp['first_token_diff']} max_delta={cmp['max_shared_top20_delta']:.6g}",
              flush=True)
(out / f"{arm}.rep.summary.json").write_text(json.dumps(summary, indent=1))
