#!/usr/bin/env python3
"""Host-side streaming HTTP client for the GLM-5.3-Flash speed head-to-head.

Published from the independently reviewed deployment instrument (issue #872):
the client retains every streamed choice and the [DONE] marker, and its
compare-generation mode requires the complete paired decoded-text/finish
population before any served-gain claim. Decoded UTF-8 + finish equality is
the only claim; token-ID/logit equality and teacher-forced KL stay separate
gates.

One instrument for both halves (ours and the EXL3 reference): Python stdlib
only, run on a host against a served OpenAI-compatible /v1/completions
endpoint. Every request sends a frozen token-id prompt (prompts.json from
build_prompts.py), temperature 0, max_tokens OUTPUT, ignore_eos, stream with
continuous usage, so every token's arrival time is observed.

Cells: every (input length L, concurrency c). A cell runs 1 warmup trial
(excluded) and then TRIALS trials; a trial releases c requests at once
(threads behind a barrier) with c distinct prompts. A cell whose KV demand
c * (L + OUTPUT) exceeds --kv-tokens (the server's reported KV capacity) is
SKIPPED with the reason, never run, because queued requests would inflate
TTFT silently. The cell set and skips are part of the instrument.

Per request: ttft_s (first token), token arrival times, e2e_s, prompt and
completion token counts from the server's usage. Derived per trial (defined
here so both halves compute the same thing):
  prefill_tok_s  = sum(L over the c requests) / (last first-token time - trial start)
  decode_tok_s   = sum(completion_tokens - 1) / (last end - first first-token time)
  tpot_ms        = per request (e2e - ttft) / (completion_tokens - 1)
  itl_ms         = every gap between consecutive token arrivals (a chunk of k
                   tokens contributes k gaps of gap/k)

usage: served_generation_client.py --base-url http://HOST:8000 --prompts prompts.json --out result.json
         [--lens 512 2048 8192] [--conc 1 2 4] [--trials 5] [--output 128] [--kv-tokens N]
         [--label-mode eager|graph] [--label-fabric roce|socket] [--label-server TEXT] [--events EVENTS.jsonl]
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

SCHEMA = "prismaquant.pact_u4.speed_client/1"


def get_json(base, path, timeout=30):
    u = urlparse(base)
    c = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, json.loads(body) if body else None


def one_request(base, model, prompt, output, barrier, rec, timeout):
    """Stream one completion; record every token arrival (wall clock)."""
    u = urlparse(base)
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": output, "temperature": 0.0,
                       "ignore_eos": True, "stream": True,
                       "stream_options": {"include_usage": True, "continuous_usage_stats": True}})
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    arrivals, last_count, usage, err, status = [], 0, None, None, None
    choices, done = [], False
    try:
        barrier.wait()
        t0 = time.time()
        conn.request("POST", "/v1/completions", body=body, headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        status = resp.status
        if status != 200:
            err = f"HTTP {status}: {resp.read()[:500]!r}"
        else:
            while True:
                line = resp.readline()
                if not line:
                    break
                line = line.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    done = True
                    break
                t = time.time()
                d = json.loads(payload)
                choices.extend(d.get("choices") or [])
                u_ = d.get("usage")
                if u_:
                    usage = u_
                    n = int(u_.get("completion_tokens") or 0)
                    if n > last_count:
                        arrivals.append((t, n - last_count))
                        last_count = n
                elif d.get("choices"):
                    # server without continuous usage: one chunk = one token
                    arrivals.append((t, 1))
                    last_count += 1
        t1 = time.time()
    except Exception as exc:  # noqa: BLE001 -- recorded, the cell reports it
        t0 = t0 if "t0" in locals() else time.time()
        t1 = time.time()
        err = f"{type(exc).__name__}: {exc}"
    finally:
        conn.close()
    rec.update({"t_start": t0, "t_end": t1, "status": status, "error": err,
                "prompt_tokens_sent": len(prompt),
                "prompt_sha256": hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                "usage": usage, "arrivals": [[a, k] for a, k in arrivals],
                "completion_tokens": last_count,
                "generation": {"choices": choices, "done": done},
                "ttft_s": (arrivals[0][0] - t0) if arrivals else None, "e2e_s": t1 - t0})


def trial_metrics(reqs, t_release):
    ok = [r for r in reqs if not r["error"] and r["arrivals"]]
    out = {"requests": len(reqs), "ok": len(ok), "errors": [r["error"] for r in reqs if r["error"]]}
    if len(ok) != len(reqs):
        return out
    firsts = [r["arrivals"][0][0] for r in ok]
    ends = [r["arrivals"][-1][0] for r in ok]
    L = sum(r["prompt_tokens_sent"] for r in ok)
    out["ttft_ms"] = [1000 * r["ttft_s"] for r in ok]
    out["prefill_tok_s"] = L / (max(firsts) - t_release) if max(firsts) > t_release else None
    gen = sum(r["completion_tokens"] - 1 for r in ok)
    span = max(ends) - min(firsts)
    out["decode_tok_s"] = gen / span if span > 0 else None
    out["tpot_ms"] = [1000 * (r["arrivals"][-1][0] - r["arrivals"][0][0]) / (r["completion_tokens"] - 1)
                      for r in ok if r["completion_tokens"] > 1]
    itl = []
    for r in ok:
        for (a, _), (b, k) in zip(r["arrivals"], r["arrivals"][1:]):
            itl.extend([1000 * (b - a) / k] * k)
    out["itl_ms"] = itl
    out["completion_tokens"] = [r["completion_tokens"] for r in ok]
    out["server_prompt_tokens"] = [(r["usage"] or {}).get("prompt_tokens") for r in ok]
    return out


def spread(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    q = statistics.quantiles(xs, n=4) if len(xs) >= 2 else [xs[0]] * 3
    return {"n": len(xs), "median": statistics.median(xs), "min": min(xs), "max": max(xs),
            "q1": q[0], "q3": q[2], "p90": (statistics.quantiles(xs, n=10)[8] if len(xs) >= 2 else xs[0])}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="served model id (default: the first id from GET /v1/models)")
    ap.add_argument("--lens", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--conc", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--output", type=int, default=128)
    ap.add_argument("--kv-tokens", type=int, default=None, help="server KV capacity in tokens; cells above it are skipped")
    ap.add_argument("--timeout-s", type=float, default=900)
    ap.add_argument("--label-mode", default="unknown")
    ap.add_argument("--label-fabric", default="unknown")
    ap.add_argument("--label-server", default="")
    ap.add_argument("--events", help="append one JSON line per cell {phase:latency,event:cell,...} (U4 events.jsonl)")
    args = ap.parse_args()
    praw = Path(args.prompts).read_bytes()
    P = json.loads(praw)
    if args.trials + P["warmup"] > len(P["prompts"][str(args.lens[0])][str(args.conc[0])]):
        raise SystemExit("prompt set has fewer trials than requested")
    st, models = get_json(args.base_url, "/v1/models")
    model = args.model or models["data"][0]["id"]
    doc = {"schema": SCHEMA, "base_url": args.base_url, "model": model, "models_response": models,
           "prompts": args.prompts, "prompts_sha256": hashlib.sha256(praw).hexdigest(),
           "labels": {"mode": args.label_mode, "fabric": args.label_fabric, "server": args.label_server},
           "config": {"lens": args.lens, "conc": args.conc, "trials": args.trials, "warmup": P["warmup"],
                      "output": args.output, "kv_tokens": args.kv_tokens, "temperature": 0.0, "ignore_eos": True},
           "client_host": __import__("socket").gethostname(), "started_unix": time.time(), "cells": {}}
    out = Path(args.out)

    def flush():
        out.write_text(json.dumps(doc, indent=1) + "\n")

    for L in args.lens:
        for c in args.conc:
            label = f"host-L{L}-c{c}"
            need = c * (L + args.output)
            if args.kv_tokens is not None and need > args.kv_tokens:
                doc["cells"][label] = {"L": L, "c": c, "skipped": True,
                                       "reason": f"KV demand {need} tokens > server KV capacity {args.kv_tokens}"}
                print(f"[speed] {label}: SKIPPED ({doc['cells'][label]['reason']})", flush=True)
                flush()
                continue
            trials_p = P["prompts"][str(L)][str(c)]
            cell = {"L": L, "c": c, "skipped": False, "trials": [], "warmup": None}
            t_cell0 = time.time()
            for t in range(P["warmup"] + args.trials):
                recs = [dict() for _ in range(c)]
                barrier = threading.Barrier(c + 1)
                th = [threading.Thread(target=one_request, args=(args.base_url, model, trials_p[t][s], args.output,
                                                                 barrier, recs[s], args.timeout_s)) for s in range(c)]
                for x in th:
                    x.start()
                t_release = time.time()
                barrier.wait()
                for x in th:
                    x.join()
                m = trial_metrics(recs, t_release)
                entry = {"trial": t, "t_release": t_release, "t_end": max(r["t_end"] for r in recs),
                         "metrics": m, "requests": recs}
                if t < P["warmup"]:
                    cell["warmup"] = entry
                else:
                    cell["trials"].append(entry)
                print(f"[speed] {label} trial {t}{' (warmup)' if t < P['warmup'] else ''}: ok {m['ok']}/{m['requests']}"
                      f" ttft_ms {[round(x) for x in m.get('ttft_ms', [])]} decode_tok_s {m.get('decode_tok_s')}", flush=True)
            t_cell1 = time.time()
            meas = cell["trials"]
            cell["window_unix"] = [meas[0]["t_release"], meas[-1]["t_end"]] if meas else [t_cell0, t_cell1]
            cell["window_unix_incl_warmup"] = [t_cell0, t_cell1]
            allm = [tr["metrics"] for tr in meas]
            cell["complete"] = all(m["ok"] == m["requests"] for m in allm)
            cell["summary"] = {
                "ttft_ms": spread([x for m in allm for x in m.get("ttft_ms", [])]),
                "prefill_tok_s": spread([m.get("prefill_tok_s") for m in allm]),
                "decode_tok_s": spread([m.get("decode_tok_s") for m in allm]),
                "tpot_ms": spread([x for m in allm for x in m.get("tpot_ms", [])]),
                "itl_ms": spread([x for m in allm for x in m.get("itl_ms", [])]),
                "completion_tokens_all_equal_output": all(n == args.output for m in allm for n in m.get("completion_tokens", [])),
                "server_prompt_tokens_all_equal_L": all(n == L for m in allm for n in m.get("server_prompt_tokens", [])),
                "input_tokens_measured": sum(L * m["requests"] for m in allm),
                "output_tokens_measured": sum(sum(m.get("completion_tokens", [])) for m in allm),
            }
            doc["cells"][label] = cell
            flush()
            if args.events:
                with open(args.events, "a") as f:
                    f.write(json.dumps({"unix": time.time(), "phase": "latency", "event": "cell",
                                        "detail": {"label": label, "start": cell["window_unix"][0],
                                                   "end": cell["window_unix"][1], "rc": 0 if cell["complete"] else 1,
                                                   "client": "host"}}) + "\n")
    doc["ended_unix"] = time.time()
    flush()
    bad = [k for k, v in doc["cells"].items() if not v.get("skipped") and not v.get("complete")]
    print(json.dumps({"out": str(out), "cells": len(doc["cells"]), "incomplete": bad}), flush=True)
    return 1 if bad else 0


def generation_value(rec, prompt, output):
    """Exact decoded UTF-8 output and finish status, independent of SSE chunking.

    This is not token-ID or logit equality: the stock streaming API does not
    return token IDs for this request. Teacher-forced KL is a separate gate.
    """
    expected_hash = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
    usage = rec.get("usage") or {}
    gen = rec.get("generation") or {}
    if (rec.get("status") != 200 or rec.get("error") is not None
            or rec.get("prompt_sha256") != expected_hash
            or rec.get("prompt_tokens_sent") != len(prompt)
            or rec.get("completion_tokens") != output
            or usage.get("prompt_tokens") != len(prompt)
            or usage.get("completion_tokens") != output
            or gen.get("done") is not True):
        raise ValueError("incomplete or unbound generated response")
    chunks, finishes = [], []
    for choice in gen.get("choices", []):
        if choice.get("index") != 0 or not isinstance(choice.get("text"), str):
            raise ValueError("missing or unsupported completion choice")
        chunks.append(choice["text"])
        if choice.get("finish_reason") is not None:
            finishes.append(choice["finish_reason"])
    text = "".join(chunks)
    if not text or finishes != ["length"]:
        raise ValueError("missing generated text or non-length termination")
    return text, finishes[0]


def compare_generation(manifest_path, reference_dir, candidate_dir):
    """Require the entire sealed warmup/timed generation population, not a subset."""
    path = Path(manifest_path)
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    prompt_path = path.parent / manifest["prompt_file"]
    prompt_bytes = prompt_path.read_bytes()
    if hashlib.sha256(prompt_bytes).hexdigest() != manifest["prompt_sha256"]:
        raise ValueError("frozen prompt digest mismatch")
    prompts = json.loads(prompt_bytes)
    if (manifest["concurrency"] != [1] or manifest["warmup"] != 1
            or manifest["temperature"] != 0 or manifest["ignore_eos"] is not True):
        raise ValueError("unsupported deterministic generation protocol")
    compared, inputs = 0, []
    for length in manifest["lens"]:
        label = f"host-L{length}-c1"
        values, fabrics, models = [], [], []
        for directory in (reference_dir, candidate_dir):
            source = Path(directory) / f"host-L{length}.json"
            data = source.read_bytes()
            doc = json.loads(data)
            inputs.append({"path": str(source), "sha256": hashlib.sha256(data).hexdigest()})
            expected_config = {"lens": [length], "conc": [1], "trials": manifest["trials"],
                               "warmup": 1, "output": manifest["output_tokens"],
                               "temperature": 0, "ignore_eos": True}
            if (doc.get("prompts_sha256") != manifest["prompt_sha256"]
                    or any(doc.get("config", {}).get(k) != v for k, v in expected_config.items())
                    or set(doc.get("cells", {})) != {label}
                    or doc.get("model") != manifest["identities"]["tessera_artifact"]
                    or doc.get("labels", {}).get("mode") != "graphs-full-decode-only+mode-none"):
                raise ValueError("generation panel/model/setting identity mismatch")
            fabrics.append(doc["labels"].get("fabric"))
            models.append(doc["model"])
            cell = doc["cells"][label]
            if cell.get("skipped") is not False or cell.get("complete") is not True:
                raise ValueError("incomplete generation cell")
            entries = [cell.get("warmup")] + cell.get("trials", [])
            if len(entries) != manifest["trials"] + 1:
                raise ValueError("missing or extra generation trials")
            arm = []
            for trial, entry in enumerate(entries):
                if not isinstance(entry, dict) or entry.get("trial") != trial or len(entry.get("requests", [])) != 1:
                    raise ValueError("generation trial/slot population mismatch")
                prompt = prompts["prompts"][str(length)]["1"][trial][0]
                arm.append(generation_value(entry["requests"][0], prompt, manifest["output_tokens"]))
            values.append(arm)
        if fabrics[0] not in ("roce", "socket") or fabrics[0] != fabrics[1] or models[0] != models[1]:
            raise ValueError("matched transport/model mismatch")
        if values[0] != values[1]:
            raise ValueError(f"generated decoded text/finish mismatch at L{length}")
        compared += len(values[0])
    return {"schema": "prismaquant.pact_u4.generation_exactness/1", "passed": True,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "requests_per_arm": compared, "inputs": inputs,
            "equality": "decoded UTF-8 text and length finish, all warmup and timed requests",
            "not_claimed": "token-ID/logit equality or teacher-forced KL equivalence"}


def generation_main():
    ap = argparse.ArgumentParser(description="Compare actual frozen deterministic streamed generations")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(sys.argv[2:])
    try:
        result = compare_generation(args.manifest, args.reference, args.candidate)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        result = {"schema": "prismaquant.pact_u4.generation_exactness/1", "passed": False,
                  "error": str(exc)}
    Path(args.out).write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result))
    return 0 if result["passed"] else 1



if __name__ == "__main__":
    raise SystemExit(generation_main() if sys.argv[1:2] == ["compare-generation"] else main())
