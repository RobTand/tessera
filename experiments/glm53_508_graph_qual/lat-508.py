"""tessera#508 latency probe: prefill TTFT and decode ITL from streamed completions.

Client-side timestamps of every streamed chunk (time.perf_counter on the host
that serves; the serve is on 127.0.0.1). A chunk carrying k tokens contributes
k intervals of (gap / k). Cases (token-id prompts, greedy, ignore_eos):

  n1_p128   one stream, 128-token prompt, 128 new tokens, 5 repeats  (decode ITL)
  n1_p1024  one stream, 1024-token prompt, 128 new tokens, 5 repeats (decode ITL)
  n8_p512   8 concurrent streams, distinct 512-token prompts, 128 new tokens, 2 passes
  ttft_p1900  1900-token prompt (one prefill chunk), 1 new token, 5 repeats
  ttft_p3649  3649-token prompt (two prefill chunks: 2048 + 1601), 1 new token, 5 repeats

Every case records its UTC window so a 1 Hz power series can be cut per case.
Writes $OUT/<arm>.lat.json.   lat-508.py PORT OUT ARM [CASES]
"""
import json
import pathlib
import statistics
import sys
import threading
import time
import urllib.request
import os

# The served model name; the full-model graph smoke reuses this probe on its serve.
MODEL = os.environ.get("T508_MODEL", "glm53-stub")

port, out, arm = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
only = set(sys.argv[4].split(",")) if len(sys.argv) > 4 else None
URL = f"http://127.0.0.1:{port}"


def post(path, payload):
    req = urllib.request.Request(URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


para = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "seventeen curious ravens observe the scene from a weathered oak branch. ")
ids = post("/tokenize", dict(model=MODEL, prompt=para * 260, add_special_tokens=False))["tokens"]
assert len(ids) >= 3649 + 64, len(ids)


def stream(prompt, max_tokens):
    payload = dict(model=MODEL, prompt=prompt, max_tokens=max_tokens, temperature=0,
                   ignore_eos=True, stream=True, return_token_ids=True,
                   stream_options=dict(include_usage=True))
    req = urllib.request.Request(URL + "/v1/completions", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps, counts, usage = [], [], None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.strip()
            if not line.startswith(b"data: "):
                continue
            data = line[6:]
            if data == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for choice in obj.get("choices") or ():
                n = len(choice.get("token_ids") or ()) or (1 if choice.get("text") else 0)
                if n:
                    stamps.append(time.perf_counter())
                    counts.append(n)
    itl = []
    for i in range(1, len(stamps)):
        gap = stamps[i] - stamps[i - 1]
        itl.extend([gap / counts[i]] * counts[i])
    return dict(ttft_s=stamps[0] - t0, itl_s=itl, tokens=sum(counts), elapsed_s=stamps[-1] - t0,
                usage=usage)


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def summarize(runs):
    ttft = [r["ttft_s"] for r in runs]
    itl = [x for r in runs for x in r["itl_s"]]
    return dict(n_runs=len(runs), ttft_median_ms=1e3 * statistics.median(ttft),
                ttft_p90_ms=1e3 * pct(ttft, 0.9),
                itl_median_ms=1e3 * statistics.median(itl) if itl else None,
                itl_p90_ms=1e3 * pct(itl, 0.9) if itl else None,
                itl_mean_ms=1e3 * statistics.fmean(itl) if itl else None,
                tokens=sum(r["tokens"] for r in runs))


def concurrent(prompts, max_tokens):
    results = [None] * len(prompts)

    def one(i):
        results[i] = stream(prompts[i], max_tokens)

    threads = [threading.Thread(target=one, args=(i,)) for i in range(len(prompts))]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, time.perf_counter() - t0


report = {"arm": arm, "cases": {}}
plan = [
    ("n1_p128", lambda: [stream(ids[0:128], 128) for _ in range(5)], None),
    ("n1_p1024", lambda: [stream(ids[3:1027], 128) for _ in range(5)], None),
    ("n8_p512", None, lambda: [concurrent([ids[11 + 37 * i:11 + 37 * i + 512] for i in range(8)], 128)
                               for _ in range(2)]),
    ("ttft_p1900", lambda: [stream(ids[5:1905], 1) for _ in range(5)], None),
    ("ttft_p3649", lambda: [stream(ids[7:3656], 1) for _ in range(5)], None),
]
stream(ids[0:32], 8)  # warm-up: first-request lazy init stays out of every case
for name, seq, conc in plan:
    if only and name not in only:
        continue
    w0 = time.time()
    if seq is not None:
        runs = seq()
        case = summarize(runs)
    else:
        passes = conc()
        runs = [r for rs, _ in passes for r in rs]
        case = summarize(runs)
        case["passes"] = [dict(elapsed_s=dt, tokens=sum(r["tokens"] for r in rs),
                               tok_per_s=sum(r["tokens"] for r in rs) / dt) for rs, dt in passes]
    case["window_utc"] = [w0, time.time()]
    report["cases"][name] = case
    print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                            for k, v in case.items() if k != "window_utc"}), flush=True)
(out / f"{arm}.lat.json").write_text(json.dumps(report, indent=1))
