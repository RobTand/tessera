"""Versioned L2048 streaming control; explicit request seeds, no speed claims.

The frozen October 5 client has no request seed input. This smaller instrument
keeps its prompt order, streaming options, sampling and full completion records,
without changing that source or manufacturing a new quality measurement.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import time
from urllib.parse import urlparse

from managed_window import atomic_json

SCHEMA = "tessera.seeded_control_client.v1"


def request(base, model, prompt, seed, timeout):
    payload = dict(model=model, prompt=prompt, max_tokens=128, temperature=0.0,
                   seed=seed, ignore_eos=True, stream=True,
                   stream_options=dict(include_usage=True, continuous_usage_stats=True))
    url = urlparse(base)
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    record = dict(request_seed=seed, request_payload=payload, status=None, error=None,
                  prompt_tokens_sent=len(prompt),
                  prompt_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                  usage=None, completion_tokens=0, generation=dict(choices=[], done=False),
                  started_unix=time.time())
    try:
        connection.request("POST", "/v1/completions", body=json.dumps(payload),
                           headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        record["status"] = response.status
        if response.status != 200:
            raise ValueError(f"HTTP {response.status}: {response.read(4096)!r}")
        while True:
            line = response.readline()
            if not line:
                break
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                record["generation"]["done"] = True
                break
            chunk = json.loads(data)
            record["generation"]["choices"].extend(chunk.get("choices") or [])
            if chunk.get("usage"):
                record["usage"] = chunk["usage"]
                record["completion_tokens"] = chunk["usage"].get("completion_tokens")
        if not record["generation"]["done"]:
            raise ValueError("stream ended before DONE")
    except (OSError, ValueError, http.client.HTTPException) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        connection.close()
        record["ended_unix"] = time.time()
    return record


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--base-url", required=True)
    result.add_argument("--model", required=True)
    result.add_argument("--prompts", type=Path, required=True)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--server-seed", type=int, required=True)
    result.add_argument("--request-seed-base", type=int, required=True)
    result.add_argument("--timeout-s", type=float, default=900)
    return result


def main(argv=None):
    from eager_determinism import PROTOCOL, prompt_population, output_population
    args = parser().parse_args(argv)
    raw = args.prompts.read_bytes()
    prompts = json.loads(raw)
    population = prompt_population(prompts)
    protocol = dict(PROTOCOL, server_seed=args.server_seed, request_seed_base=args.request_seed_base)
    doc = dict(schema=SCHEMA, model=args.model, base_url=args.base_url, protocol=protocol,
               prompts_sha256=hashlib.sha256(raw).hexdigest(), requests=[],
               seed_observation="Request seeds are sent explicitly; effective server seeds are not exposed by this API.")
    for trial, prompt in enumerate(population):
        row = request(args.base_url, args.model, prompt, args.request_seed_base + trial, args.timeout_s)
        row.update(trial=trial, slot=0, warmup=trial == 0, request_id=f"L2048-c1/trial{trial}/slot0")
        doc["requests"].append(row)
        atomic_json(args.out, doc)  # retain every complete or failed request before validation
        if row["error"]:
            return 1
    output_population(doc, prompts, doc["prompts_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
