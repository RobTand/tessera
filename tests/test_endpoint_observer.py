"""Exercise real HTTP reads and the public CLI without serving imports."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from tessera import endpoint_observer as eo
from tessera import endpoint_witness as ew
from _endpoint_fixture import receipt, resign

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def server(value, publication, *, status=200, envelope=True, replace=False, cached=False):
    original = copy.deepcopy(value)
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.reply(False)

        def do_POST(self):
            self.reply(True)

        def reply(self, publish):
            calls.append(publish)
            current = copy.deepcopy(original)
            if not cached:
                request_id = parse_qs(urlsplit(self.path).query)["request_id"][0]
                current["lifetime"]["request_id"] = request_id
                current["tokenizer"]["request_id"] = request_id
                for rank in current["artifacts"]:
                    rank["request_id"] = request_id
            if replace and len(calls) > 1:
                current["listener"]["owner"]["start_ticks"] += 1
                current["launch"]["attempt_id"] = ew.attempt_id(current["listener"]["owner"])
            resign(current)
            path = eo.publish_witness(publication, witness=current) if publish else None
            payload = {"receipt": current, "public_receipt_path": str(path) if path else None} if envelope else {"data": [{"id": "fixture"}]}
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):
            pass

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    host, port = http.server_address
    url = f"http://{host}:{port}"
    original["listener"]["endpoint"] = url
    resign(original)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield url, original, calls
    finally:
        http.shutdown()
        http.server_close()
        thread.join()


def command(tool, *args):
    return subprocess.run([sys.executable, str(ROOT / "tools" / tool), *map(str, args)],
                          capture_output=True, text=True, timeout=45)


def test_the_probe_publishes_and_the_standalone_verifier_rechecks_the_listener(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    with server(value, tmp_path / "public") as (url, observed, calls):
        result = command("probe_endpoint_witness.py", "--base-url", url, "--artifact-dir", artifact)
        assert result.returncode == 0, result.stderr
        published = Path(result.stdout.strip())
        assert json.loads(published.read_bytes())["listener"] == observed["listener"]
        checked = command("verify_endpoint_witness.py", published, "--served-dir", artifact, "--expect-ranks", "0,1")
        assert checked.returncode == 0, checked.stderr
        assert "runtime witness valid" in checked.stdout
        assert calls == [True, False, False]


def test_same_alias_replacement_listener_cannot_reuse_the_receipt(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    with server(value, tmp_path / "public", replace=True) as (url, _, _):
        result = command("probe_endpoint_witness.py", "--base-url", url, "--artifact-dir", artifact)
        assert result.returncode == 4
        assert "binding differs" in result.stderr


def test_alias_and_models_reply_alone_are_not_runtime_evidence(tmp_path):
    value = receipt(tmp_path / "artifact")
    with server(value, tmp_path / "public", envelope=False) as (url, _, _):
        with pytest.raises(ValueError, match="runtime response"):
            eo.fetch_witness(url)


def test_unreachable_or_incomplete_runtime_evidence_stays_explicit(tmp_path):
    value = receipt(tmp_path / "artifact")
    with server(value, tmp_path / "public", status=503) as (url, _, _):
        with pytest.raises(ValueError, match="runtime evidence is unavailable"):
            eo.fetch_witness(url)
    with pytest.raises(ValueError, match="runtime evidence is unavailable"):
        eo.fetch_witness(url, timeout_s=1)


def test_a_cached_runtime_reply_cannot_supply_fresh_listener_evidence(tmp_path):
    value = receipt(tmp_path / "artifact")
    with server(value, tmp_path / "public", cached=True) as (url, _, _):
        with pytest.raises(ValueError, match="fresh runtime request"):
            eo.fetch_witness(url)


@pytest.mark.parametrize("url", ["artifact", "file:///tmp/artifact", "http://host", "http://host:3/path", "http://a:b@host:3", "http://host:3/?x=1"])
def test_runtime_addresses_cannot_be_input_paths_or_indirect_urls(url):
    with pytest.raises(ValueError, match="direct HTTP"):
        eo.fetch_witness(url)


def test_publication_reproves_the_actual_observed_source_bytes(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    path = artifact / "model.safetensors"
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="artifact file bytes differ"):
        eo.publish_witness(tmp_path / "public", witness=value)


def test_publication_is_complete_and_never_overwrites_different_bytes(tmp_path):
    value = receipt(tmp_path / "artifact")
    published = eo.publish_witness(tmp_path / "public", witness=value)
    original = published.read_bytes()
    assert original == (ew.canonical(value) + "\n").encode()
    assert eo.publish_witness(tmp_path / "public", witness=value) == published
    published.write_bytes(b"foreign bytes")
    with pytest.raises(ValueError, match="immutable runtime receipt"):
        eo.publish_witness(tmp_path / "public", witness=value)
    assert published.read_bytes() == b"foreign bytes"


def test_publication_refuses_missing_or_incomplete_loaded_inputs(tmp_path):
    value = receipt(tmp_path / "artifact")
    value["artifacts"][0]["models"][0]["inputs"] = []
    resign(value)
    with pytest.raises(ValueError, match="loader inputs"):
        eo.publish_witness(tmp_path / "public", witness=value)
    assert not (tmp_path / "public").exists()


def test_standalone_verifier_never_imports_serving_or_torch(tmp_path):
    artifact = tmp_path / "artifact"
    value = receipt(artifact)
    with server(value, tmp_path / "public") as (url, _, _):
        published = eo.fetch_witness(url, publish=True)["public_receipt_path"]
        script = """import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'vllm') or fullname.startswith('tessera.serving'):
            raise RuntimeError('consumer imported a serving dependency: ' + fullname)
sys.meta_path.insert(0, Block())
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name='__main__')
"""
        result = subprocess.run([sys.executable, "-c", script, str(ROOT / "tools/verify_endpoint_witness.py"),
                                 published, "--served-dir", str(artifact)],
                                capture_output=True, text=True, timeout=45)
        assert result.returncode == 0, result.stderr
