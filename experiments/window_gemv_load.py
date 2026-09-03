#!/usr/bin/env python3
"""Drive a served Tessera arm and read the latency the ENGINE measured.

Two loads, because the window GEMV serves one of the two regimes and the
materialised path serves the other:

* **decode**: a short prompt and many output tokens at concurrency 1, so
  almost every forward is ``M = 1`` -- the shape ``fp8_gemv.streamed_apply``
  routes to the kernel.
* **prefill**: a long prompt and one output token, so the cost is the
  many-row forward the GEMV refuses by name and the materialised path serves.

The numbers reported are ``vllm:time_to_first_token_seconds`` and
``vllm:request_time_per_output_token_seconds`` -- vLLM's own histograms, differenced
across each load, so they describe the requests this script drove and nothing
else.  Client-side wall clock is recorded beside them as a cross-check, never
as the headline: a client number folds in HTTP and this box's scheduler.

The torch profile is taken over a SECOND, shorter decode load, because a
profiled forward is not a timed forward -- profiling a run and quoting its
latency would be quoting the profiler.  The trace answers a different
question: which kernels launched.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request


def load_now() -> dict:
    """The host's run-queue load, right now.

    Recorded at both ends of every timed window because a latency number taken
    on a contended box is noise, and a contended number that SAYS SO is still
    useful while one that does not is worse than nothing.  This box runs many
    agents' jobs at once and the GPU lock does not serialise the CPU-bound ones,
    so "I held the lock" is not evidence the box was quiet.  ``nproc`` travels
    with the reading so load is interpretable as a ratio rather than a bare
    number.
    """
    one, five, fifteen = os.getloadavg()
    return {"load1": one, "load5": five, "load15": fifteen,
            "ncpu": os.cpu_count(),
            "load1_per_cpu": round(one / (os.cpu_count() or 1), 3)}

#: vLLM 0.28 publishes per-output-token latency as
#: ``vllm:request_time_per_output_token_seconds``; the name this script first
#: read, ``vllm:time_per_output_token_seconds``, does not exist in this release
#: and returned None for every window.  Recorded here as a named constant so the
#: stem is stated once.  ``vllm:inter_token_latency_seconds`` is read beside it:
#: same mean, but counted per token rather than per request, so the two together
#: say whether a difference is in the per-request or the per-token accounting.
TPOT_STEM = "vllm:request_time_per_output_token_seconds"

_HIST = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})? (?P<value>\S+)$')


def _get(url: str, timeout: float = 600.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return fh.read().decode()


def _post(url: str, payload: dict | None, timeout: float = 600.0) -> dict | None:
    data = b"" if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        body = fh.read().decode()
    return json.loads(body) if body.strip() else None


def scrape(url: str) -> dict:
    """``metric name -> value`` for the ``_sum``/``_count`` series we read."""
    out: dict[str, float] = {}
    for line in _get(url + "/metrics").splitlines():
        if line.startswith("#"):
            continue
        m = _HIST.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        if name.endswith("_sum") or name.endswith("_count"):
            try:
                out[name] = out.get(name, 0.0) + float(m.group("value"))
            except ValueError:
                pass
    return out


def moved(before: dict, after: dict) -> dict:
    """Every vLLM series that CHANGED over the window, with its delta.

    Insurance, and cheap: if a histogram is renamed between vLLM releases the
    two stems read below silently return ``None`` and the receipt would say
    nothing about a serve that cost a lock slot to take.  Recording what
    actually moved makes the number recoverable from the receipt instead of
    from a second serve.
    """
    out = {}
    for k, v in after.items():
        if not k.startswith("vllm:"):
            continue
        d = v - before.get(k, 0.0)
        if d:
            out[k] = d
    return out


def delta(before: dict, after: dict, stem: str):
    """Mean of one histogram over the window, or None when it did not move."""
    s = after.get(stem + "_sum", 0.0) - before.get(stem + "_sum", 0.0)
    n = after.get(stem + "_count", 0.0) - before.get(stem + "_count", 0.0)
    if n <= 0:
        return None
    return {"mean_s": s / n, "sum_s": s, "count": n}


def completion(prompt_tokens: int, max_tokens: int, req: int) -> dict:
    """One request's body.  Token ids, so "512 tokens" is 512 tokens and not a
    tokenizer's opinion of a string.

    EVERY REQUEST GETS ITS OWN PROMPT, and that is load-bearing rather than
    tidy.  The serve runs with vLLM's default ``enable_prefix_caching=True``,
    so a repeated prompt is a prefix-cache HIT: the many-row forward never
    runs and a "prefill" window would time a cache lookup instead of the
    materialised path this A/B exists to compare.  Perturbing the ids per
    request is the fix that keeps the serve's configuration identical to the
    KL serve's -- turning prefix caching off would make the latency arm a
    different serve from the one whose KL is reported.
    """
    ids = [(i * 7919 + 104729 * (req + 1)) % 100000 + 1000 for i in range(prompt_tokens)]
    return {"model": "kl-target", "prompt": ids, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": False}


#: Monotonic across the whole run, so no two requests anywhere in it -- warmup,
#: timed windows, profiled load -- share a prompt.
_REQ = 0


def drive(url: str, n: int, prompt_tokens: int, max_tokens: int) -> dict:
    global _REQ
    t0 = time.time()
    for _ in range(n):
        _REQ += 1
        _post(url + "/v1/completions", completion(prompt_tokens, max_tokens, _REQ))
    wall = time.time() - t0
    return {"requests": n, "prompt_tokens": prompt_tokens, "max_tokens": max_tokens,
            "wall_s": wall, "wall_s_per_request": wall / n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--regime", required=True)
    ap.add_argument("--decode-requests", type=int, default=12)
    ap.add_argument("--decode-out-tokens", type=int, default=128)
    ap.add_argument("--prefill-requests", type=int, default=12)
    ap.add_argument("--prefill-prompt-tokens", type=int, default=512)
    ap.add_argument("--warmup-requests", type=int, default=4)
    args = ap.parse_args()

    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    # Warm up: the first forwards of a compiled serve pay capture, and the
    # first of any serve pays allocator growth.  Neither belongs in the number.
    drive(args.url, args.warmup_requests, 32, 32)
    drive(args.url, 2, args.prefill_prompt_tokens, 1)

    windows = {}
    marks = {"decode_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    loads = {"decode_start": load_now()}
    b = scrape(args.url)
    windows["decode"] = drive(args.url, args.decode_requests, 32, args.decode_out_tokens)
    a = scrape(args.url)
    windows["decode"]["ttft"] = delta(b, a, "vllm:time_to_first_token_seconds")
    windows["decode"]["tpot"] = delta(b, a, TPOT_STEM)
    windows["decode"]["itl"] = delta(b, a, "vllm:inter_token_latency_seconds")
    windows["decode"]["series_moved"] = moved(b, a)
    marks["decode_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    loads["decode_end"] = load_now()

    marks["prefill_start"] = marks["decode_end"]
    loads["prefill_start"] = load_now()
    b = scrape(args.url)
    windows["prefill"] = drive(args.url, args.prefill_requests,
                               args.prefill_prompt_tokens, 1)
    a = scrape(args.url)
    windows["prefill"]["ttft"] = delta(b, a, "vllm:time_to_first_token_seconds")
    windows["prefill"]["tpot"] = delta(b, a, TPOT_STEM)
    windows["prefill"]["itl"] = delta(b, a, "vllm:inter_token_latency_seconds")
    windows["prefill"]["series_moved"] = moved(b, a)
    marks["prefill_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    loads["prefill_end"] = load_now()

    # The profile, over its own short load.  A profiled forward is not a timed
    # forward, so nothing above is taken from this window.
    profiled = None
    try:
        _post(args.url + "/start_profile", None)
        profiled = {"decode": drive(args.url, 2, 32, 24),
                    "prefill": drive(args.url, 2, args.prefill_prompt_tokens, 1)}
        _post(args.url + "/stop_profile", None, timeout=900.0)
        # The trace is flushed asynchronously; give the writer a moment.
        time.sleep(20)
    except Exception as exc:  # noqa: BLE001 -- a missing profile is reported, not fatal
        profiled = {"error": f"{type(exc).__name__}: {exc}"}

    receipt = {
        "schema": "tessera.window_gemv.served_latency/1",
        "arm": args.arm, "serve_mode": args.mode, "forward": args.regime,
        "started_utc": started,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "marks_utc": marks,
        "host_load": loads,
        # A verdict the receipt carries itself, so a reader does not have to
        # decide from four raw numbers whether the box was quiet.  The threshold
        # is one runnable process per core: above that the run queue is
        # oversubscribed and a host-driven latency number is contended by
        # definition.  It is a LABEL, never a filter -- the numbers are reported
        # either way.
        "contended": max(v["load1_per_cpu"] for v in loads.values()) > 1.0,
        "max_load1_per_cpu": max(v["load1_per_cpu"] for v in loads.values()),
        "windows": windows,
        "profiled_load": profiled,
    }
    with open(args.out, "w") as fh:
        json.dump(receipt, fh, indent=1, sort_keys=True)
    for name, w in windows.items():
        ttft = w.get("ttft") or {}
        tpot = w.get("tpot") or {}
        itl = w.get("itl") or {}
        print(f"{name}: wall {w['wall_s_per_request']*1000:.2f} ms/req  "
              f"TTFT {1000*ttft.get('mean_s', float('nan')):.2f} ms  "
              f"TPOT {1000*tpot.get('mean_s', float('nan')):.3f} ms "
              f"(n={tpot.get('count')})  "
              f"ITL {1000*itl.get('mean_s', float('nan')):.3f} ms "
              f"(n={itl.get('count')})")
    lo = max(v["load1_per_cpu"] for v in loads.values())
    print(f"host load1/cpu peaked at {lo:.2f} over the timed windows "
          f"({'CONTENDED -- report these as contended' if lo > 1.0 else 'box was quiet'})")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
