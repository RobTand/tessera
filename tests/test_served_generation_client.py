"""The published generation client retains complete streamed generations.

Ported from the independently reviewed deployment instrument tests (issue
#872): a real HTTP/SSE round trip against a fake streaming server, the paired
decoded-text/finish population comparator with its negative controls, and the
orchestrator chain that requires generated-output evidence before SHIP.
"""
from pathlib import Path
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tools'
TTFT, ITL = 0.001, 0.001

_SPEC = importlib.util.spec_from_file_location('served_generation_client', HERE / 'served_generation_client.py')
client = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(client)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "fake-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n, L = req["max_tokens"], len(req["prompt"])
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        time.sleep(TTFT)
        for i in range(1, n + 1):
            if i > 1:
                time.sleep(ITL)
            d = {"choices": [{"text": "x", "index": 0, "finish_reason": "length" if i == n else None}],
                 "usage": {"prompt_tokens": L, "completion_tokens": i, "total_tokens": L + i}}
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            self.wfile.flush()
        d = {"choices": [], "usage": {"prompt_tokens": L, "completion_tokens": n, "total_tokens": L + n}}
        self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\ndata: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def test_streamed_generation_and_paired_refusals(tmp_path, monkeypatch):
    import copy
    import hashlib

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    prompt = [1, 2, 3]
    rec = {}
    try:
        client.one_request(f"http://127.0.0.1:{server.server_address[1]}", "fake-model", prompt,
                           2, threading.Barrier(1), rec, 5)
    finally:
        server.shutdown()
        server.server_close()
    assert client.generation_value(rec, prompt, 2) == ("xx", "length")
    panel = {"prompts": {"3": {"1": [[prompt], [prompt]]}}}
    raw = json.dumps(panel).encode()
    (tmp_path / "prompts.json").write_bytes(raw)
    prompt_hash = hashlib.sha256(raw).hexdigest()
    manifest = {"prompt_file": "prompts.json", "prompt_sha256": prompt_hash, "lens": [3],
                "concurrency": [1], "warmup": 1, "trials": 1, "output_tokens": 2,
                "temperature": 0, "ignore_eos": True, "identities": {"tessera_artifact": "fake-model"}}
    mp = tmp_path / "manifest.json"
    mp.write_text(json.dumps(manifest))
    document = {"model": "fake-model", "prompts_sha256": prompt_hash,
                "labels": {"mode": "graphs-full-decode-only+mode-none", "fabric": "roce"},
                "config": {"lens": [3], "conc": [1], "trials": 1, "warmup": 1, "output": 2,
                           "temperature": 0, "ignore_eos": True},
                "cells": {"host-L3-c1": {"skipped": False, "complete": True,
                          "warmup": {"trial": 0, "requests": [rec]},
                          "trials": [{"trial": 1, "requests": [rec]}]}}}
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(); b.mkdir()
    (a / "host-L3.json").write_text(json.dumps(document))
    (b / "host-L3.json").write_text(json.dumps(document))
    assert client.compare_generation(mp, a, b)["requests_per_arm"] == 2
    # Transport chunks may differ without changing decoded generation.
    changed = copy.deepcopy(document)
    changed["cells"]["host-L3-c1"]["trials"][0]["requests"][0]["generation"]["choices"] = [
        {"index": 0, "text": "xx", "finish_reason": "length"}]
    (b / "host-L3.json").write_text(json.dumps(changed))
    assert client.compare_generation(mp, a, b)["passed"] is True
    for mutation in ("text", "finish", "done", "missing", "usage", "prompt", "trial", "population", "fabric", "mode", "skipped"):
        changed = copy.deepcopy(document)
        cell = changed["cells"]["host-L3-c1"]
        request = cell["trials"][0]["requests"][0]
        if mutation == "text": request["generation"]["choices"][0]["text"] = "y"
        elif mutation == "finish": request["generation"]["choices"][-1]["finish_reason"] = "stop"
        elif mutation == "done": request["generation"]["done"] = False
        elif mutation == "missing": del request["generation"]
        elif mutation == "usage": request["usage"]["completion_tokens"] = 1
        elif mutation == "prompt": request["prompt_sha256"] = "unbound"
        elif mutation == "trial": cell["trials"][0]["trial"] = 0
        elif mutation == "population": cell["trials"] = []
        elif mutation == "fabric": changed["labels"]["fabric"] = "socket"
        elif mutation == "mode": changed["labels"]["mode"] = "eager"
        elif mutation == "skipped": cell["skipped"] = True
        (b / "host-L3.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            client.compare_generation(mp, a, b)


def test_existing_chain_requires_generated_output_gate():
    source = ROOT / 'tools' / 'serve_comparison_gates.sh'
    function = re.search(r"^lever_chain\(\) \{.*?^\}", source.read_text(), re.M | re.S).group()
    for returncode in (0, 1):
        script = """say() { :; }
run_val() { RV_STATUS=ran; RV_LV="levers as declared"; RV_KV=0; }
tr3_gate() { echo "BITWISE fixture"; }
generation_gate() { return "$GENERATION_RC"; }
BASE_ARM=A; GATE_ARM=B; FALLBACK_ARMS=""; SHIP=""; SHIP_RUN=""; CHAIN_ROOT=/unused; U=/unused
WID=fixture; N=/dev/null; GATE_TRAIL=""
""" + function + '\nlever_chain; printf "%s" "$SHIP"\n'
        result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, check=True,
                                env={**os.environ, "GENERATION_RC": str(returncode)})
        assert result.stdout == ("B" if returncode == 0 else "")
