"""tessera#508 equality suite: greedy top-20 logprobs, single and batched decode.

The graph question is whether a captured decode step returns what the eager
step returns, bit for bit. Decode graphs are captured at a few batch sizes and
a smaller batch is padded up to the next one, so the suite decodes at batch 1
and at batches that need padding (3, 5, 6, 7) as well as batches that do not
(2, 4, 8). Every case keeps prompt + generated tokens at or under 2048, the
index_topk threshold above which the stock top-k kernel writes its selection
in arrival order and eager is not repeat-exact (tessera#508, eager5/eager6).

A batch is ONE completions request carrying a list of token-id prompts. The
server turns it into one engine request per prompt, and without help those
requests can reach the scheduler in different steps, which changes the batch
composition from run to run. When the serve exposes the development endpoints
(VLLM_SERVER_DEV_MODE=1), every batched case is admitted under
``POST /pause?mode=keep``: the requests queue while the scheduler is paused and
``POST /resume`` releases them into the same step. Every case records the
change in ``vllm:iteration_tokens_total`` (engine steps and tokens), so two arms
can be checked for the same step structure before their logprobs are compared.

The ``rep_*`` cases at the end repeat two batch-1 cases back to back; they
measure whether eager returns the same result for the same request later in
the same serve.

Records: $OUT/<arm>.eq.<case>.json, and $OUT/<arm>.eq.summary.json with a
digest of every choice's token ids and top-20 logprobs.

  equal-508.py PORT OUT ARM [CASES]    CASES: comma list of case names (default all)

T702_LONG=1 swaps in the long-context screen (tessera#702) instead: cases whose
longest row passes index_topk, so a graph serve replays its long-context class.
Above that threshold eager is not repeat-exact (above), so these cases are a
screen against a pool of eager outcomes, never part of the equality set.
T702_MAX_MODEL_LEN is the serve's max_model_len, their length limit.
"""
import hashlib
import json
import math
import pathlib
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import os

# The served model name; the full-model graph smoke reuses this probe on its serve.
MODEL = os.environ.get("T508_MODEL", "glm53-stub")

port, out, arm = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
out.mkdir(parents=True, exist_ok=True)
# T508_HOST: the API server's address when it runs on another box (a TP 2 serve's rank 0).
URL = f"http://{os.environ.get('T508_HOST', '127.0.0.1')}:{port}"
TOPK = 20
LIMIT = 2048


def post(path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(URL + path, data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metrics():
    """Engine-step counters: iteration_tokens_total count/sum and its buckets."""
    with urllib.request.urlopen(URL + "/metrics", timeout=30) as r:
        text = r.read().decode()
    found = {}
    for line in text.splitlines():
        m = re.match(r'^vllm:(iteration_tokens_total_(?:count|sum|bucket)|num_preemptions_total|'
                     r'prompt_tokens_total|generation_tokens_total)(\{[^}]*\})?\s+([0-9.eE+-]+)$', line)
        if m:
            name, labels, value = m.group(1), m.group(2) or "", float(m.group(3))
            le = re.search(r'le="([^"]+)"', labels)
            found[name + (f"[{le.group(1)}]" if le else "")] = value
    return found


def settled_metrics():
    """Stats reach the API server after the step; read until two reads agree."""
    prev = metrics()
    for _ in range(20):
        time.sleep(0.25)
        cur = metrics()
        if cur == prev:
            return cur
        prev = cur
    return prev


def delta(a, b):
    return {k: b[k] - a.get(k, 0.0) for k in b if b[k] - a.get(k, 0.0)}


para = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "seventeen curious ravens observe the scene from a weathered oak branch. ")
LONG = os.environ.get("T702_LONG") == "1"
ids = post("/tokenize", dict(model=MODEL, prompt=para * (400 if LONG else 200),
                             add_special_tokens=False))["tokens"]


def window(n, offset):
    """n token ids starting at offset: distinct prompts of chosen lengths."""
    return ids[offset:offset + n]


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

if LONG:
    LIMIT = int(os.environ["T702_MAX_MODEL_LEN"])
    CASES = {
        "long_b1_len2100": ([(2100, 0)], 32),
        "long_b1_len4000": ([(4000, 3)], 32),
        "long_b1_len8000": ([(8000, 5)], 32),
        # one row past index_topk makes the whole step's max_seq_len long
        "long_b2_mixed": ([(300, 11), (5000, 13)], 24),
        "long_b4_mixed": ([(37, 11), (900, 13), (2500, 17), (6000, 19)], 24),
        "long_rep_len4000_a": ([(4000, 3)], 32),
        "long_rep_len4000_b": ([(4000, 3)], 32),
    }


def digest(choice):
    h = hashlib.sha256()
    h.update(json.dumps(choice["token_ids"]).encode())
    h.update(json.dumps(choice["logprobs"]["top_logprobs"], sort_keys=True).encode())
    return h.hexdigest()


# Development endpoints present? A paused-and-resumed idle engine is a no-op.
try:
    post("/pause?mode=keep", timeout=60)
    post("/resume", timeout=60)
    admission = "paused"
except urllib.error.HTTPError as err:
    admission = f"unpaused (dev endpoints absent: HTTP {err.code})"
except Exception as err:  # a hung or refused pause is recorded, not fatal
    admission = f"unpaused (pause probe failed: {type(err).__name__}: {err})"
print("batched admission:", admission, flush=True)

summary = {"_admission": admission}
for name, (specs, max_tokens) in CASES.items():
    if only and name not in only:
        continue
    prompts = [window(n, off) for n, off in specs]
    assert all(len(p) + max_tokens <= LIMIT for p in prompts), name
    payload = dict(model=MODEL, prompt=prompts, max_tokens=max_tokens, temperature=0,
                   logprobs=TOPK, return_token_ids=True, ignore_eos=True)
    before = settled_metrics()
    t0 = time.time()
    if len(prompts) > 1 and admission == "paused":
        box = {}

        def send():
            try:
                box["res"] = post("/v1/completions", payload)
            except Exception as exc:  # re-raised on the main thread
                box["err"] = exc

        post("/pause?mode=keep", timeout=60)
        sender = threading.Thread(target=send)
        sender.start()
        time.sleep(2.0)  # every prompt of the list reaches the paused scheduler
        post("/resume", timeout=60)
        sender.join()
        if "err" in box:
            raise box["err"]
        res = box["res"]
    else:
        res = post("/v1/completions", payload)
    elapsed = time.time() - t0
    steps = delta(before, settled_metrics())
    (out / f"{arm}.eq.{name}.json").write_text(json.dumps(res, indent=1))
    choices = sorted(res["choices"], key=lambda c: c["index"])
    assert len(choices) == len(prompts), (name, len(choices))
    for c in choices:
        assert len(c["token_ids"]) == max_tokens, (name, c["index"], len(c["token_ids"]))
        assert all(math.isfinite(v) for v in c["logprobs"]["token_logprobs"]), name
    summary[name] = dict(batch=len(prompts), lengths=[n for n, _ in specs], max_tokens=max_tokens,
                         digests=[digest(c) for c in choices], elapsed_s=elapsed, steps=steps)
    print(name, len(prompts), [d[:12] for d in summary[name]["digests"]],
          "steps", int(steps.get("iteration_tokens_total_count", 0)),
          "tokens", int(steps.get("iteration_tokens_total_sum", 0)), flush=True)
(out / f"{arm}.eq.summary.json").write_text(json.dumps(summary, indent=1))
