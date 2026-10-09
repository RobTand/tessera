"""CPU checks for ``tessera.endpoint_observer``: runtime observation and receipt publication (tessera#1056).

The observer reads live evidence only. The served alias comes from an HTTP
``/v1/models`` reply. Byte facts come from file bytes read at observation
time. The vocabulary size comes from the served configuration or a loaded
object. No caller-supplied dictionary establishes a loaded fact. These tests
use a loopback stub server and small fixture files, so they prove the
observation rule without a GPU serve. Fixture observations qualify nothing.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tessera import endpoint_observer as eo
from tessera import endpoint_witness as ew

CONFIG_BYTES = b'{"vocab_size": 512}'
WEIGHT_BYTES = bytes(range(256)) * 4
TOKENIZER_BYTES = b'{"model": {"vocab": {"a": 0, "b": 1}}}'


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def served_dir(tmp_path):
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "config.json").write_bytes(CONFIG_BYTES)
    (root / "model.safetensors").write_bytes(WEIGHT_BYTES)
    (root / "tokenizer.json").write_bytes(TOKENIZER_BYTES)
    return root


def _serve(payload: bytes, status: int = 200):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if self.path != "/v1/models":
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _models_url(server) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}"


def _listener(url, **over):
    args = {"lifetime_id": "serve-1", "clock": lambda: 1000.0}
    args.update(over)
    return eo.observe_listener(url, **args)


def _rank(rank, served_dir, **over):
    args = {"lifetime_id": "serve-1", "clock": lambda: 1001.0}
    args.update(over)
    return eo.observe_rank_bytes(rank, served_dir, **args)


class _LoadedTokenizer:
    def __init__(self, size):
        self._size = size

    def __len__(self):
        return self._size


def test_the_served_alias_comes_from_the_models_reply():
    server = _serve(json.dumps({"data": [{"id": "glm53-artifact"}]}).encode())
    try:
        observed = _listener(_models_url(server))
    finally:
        server.shutdown()
    assert observed["served_alias"] == "glm53-artifact"
    assert observed["endpoint"] == _models_url(server)


def test_a_listener_with_no_served_model_is_refused():
    server = _serve(json.dumps({"data": []}).encode())
    try:
        with pytest.raises(ValueError, match="no served model"):
            _listener(_models_url(server))
    finally:
        server.shutdown()


def test_a_listener_with_several_models_is_refused():
    server = _serve(json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode())
    try:
        with pytest.raises(ValueError, match="several served models"):
            _listener(_models_url(server))
    finally:
        server.shutdown()


def test_an_unreachable_listener_is_refused():
    server = _serve(b"{}")
    url = _models_url(server)
    server.shutdown()
    server.server_close()
    with pytest.raises(ValueError, match="cannot read"):
        _listener(url)


def test_a_non_http_base_url_is_refused():
    with pytest.raises(ValueError, match="http"):
        _listener("artifact-dir")


def test_rank_byte_facts_come_from_the_file_bytes(served_dir):
    observed = _rank(0, served_dir)
    assert observed["files"] == {"config.json": _digest(CONFIG_BYTES),
                                 "model.safetensors": _digest(WEIGHT_BYTES),
                                 "tokenizer.json": _digest(TOKENIZER_BYTES)}
    assert observed["sizes"] == {"config.json": len(CONFIG_BYTES),
                                 "model.safetensors": len(WEIGHT_BYTES),
                                 "tokenizer.json": len(TOKENIZER_BYTES)}
    assert observed["bytes"] == len(CONFIG_BYTES) + len(WEIGHT_BYTES) + len(TOKENIZER_BYTES)


def test_rank_observation_covers_every_served_file(served_dir):
    (served_dir / "extra.bin").write_bytes(b"x")
    observed = _rank(1, served_dir)
    assert observed["files"]["extra.bin"] == _digest(b"x")


def test_rank_observation_reads_changed_bytes(served_dir):
    first = _rank(0, served_dir)
    (served_dir / "model.safetensors").write_bytes(WEIGHT_BYTES + b"x")
    second = _rank(0, served_dir)
    assert second["files"]["model.safetensors"] != first["files"]["model.safetensors"]
    assert second["files"]["model.safetensors"] == _digest(WEIGHT_BYTES + b"x")


def test_an_empty_served_directory_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no files"):
        _rank(0, tmp_path)


def test_a_shared_omission_is_proved_against_the_served_directory(served_dir, tmp_path):
    short = tmp_path / "short"
    short.mkdir()
    (short / "config.json").write_bytes(CONFIG_BYTES)
    (short / "tokenizer.json").write_bytes(TOKENIZER_BYTES)
    first = _rank(0, short)
    second = _rank(1, short)
    listener = {"endpoint": "http://10.100.96.2:8142", "served_alias": "glm53-artifact",
                "lifetime_id": "serve-1", "observed_unix": 1000.0}
    launch = {"attempt_id": "nonce-abc", "ranks": [0, 1],
              "lifetime_id": "serve-1", "observed_unix": 1001.0}
    tokenizer = {"vocab_size": 512, "vocab_source": "served-config",
                 "files": {"tokenizer.json": _digest(TOKENIZER_BYTES)},
                 "sizes": {"tokenizer.json": len(TOKENIZER_BYTES)},
                 "lifetime_id": "serve-1", "observed_unix": 1004.0}
    witness = ew.build_witness(listener=listener, launch=launch,
                               artifacts=[first, second], tokenizer=tokenizer)
    # Both ranks agree, and both omit model.safetensors: the live proof refuses.
    reason = ew.prove_loaded_bytes(witness, served_dir)
    assert reason is not None and "model.safetensors" in reason


def test_the_vocab_size_comes_from_the_served_config(served_dir):
    observed = eo.observe_server_tokenizer(served_dir, lifetime_id="serve-1",
                                           clock=lambda: 1004.0)
    assert observed["vocab_size"] == 512
    assert observed["vocab_source"] == "served-config"
    assert observed["files"] == {"tokenizer.json": _digest(TOKENIZER_BYTES)}


def test_the_vocab_size_comes_from_the_loaded_object(tmp_path):
    bare = tmp_path / "novocab"
    bare.mkdir()
    (bare / "config.json").write_bytes(b'{"model_type": "glm"}')
    (bare / "tokenizer.json").write_bytes(TOKENIZER_BYTES)
    observed = eo.observe_server_tokenizer(bare, lifetime_id="serve-1",
                                           clock=lambda: 1004.0,
                                           loaded=_LoadedTokenizer(512))
    assert observed["vocab_size"] == 512
    assert observed["vocab_source"] == "loaded"


def test_a_loaded_object_that_disagrees_with_config_is_refused(served_dir):
    with __import__("pytest").raises(ValueError, match="differs from the served config"):
        eo.observe_server_tokenizer(served_dir, lifetime_id="serve-1",
                                    loaded=_LoadedTokenizer(513))


def test_a_tokenizer_without_a_length_is_refused(served_dir):
    with pytest.raises(ValueError, match="vocabulary length"):
        eo.observe_server_tokenizer(served_dir, lifetime_id="serve-1",
                                    loaded=object())


def test_a_tokenizer_without_a_served_file_is_refused(tmp_path):
    empty = tmp_path / "bare"
    empty.mkdir()
    (empty / "config.json").write_bytes(CONFIG_BYTES)
    with pytest.raises(ValueError, match="no tokenizer file"):
        eo.observe_server_tokenizer(empty, lifetime_id="serve-1")


def test_duplicate_rank_observations_are_refused(served_dir, tmp_path):
    first = _rank(0, served_dir)
    path = tmp_path / "rank-0.json"
    path.write_text(json.dumps(first))
    with pytest.raises(ValueError, match="repeat a rank"):
        eo.collect_rank_observations([path, path])


def test_collected_ranks_arrive_sorted(served_dir, tmp_path):
    paths = []
    for rank in (1, 0):
        path = tmp_path / f"rank-{rank}.json"
        path.write_text(json.dumps(_rank(rank, served_dir)))
        paths.append(path)
    assert [item["rank"] for item in eo.collect_rank_observations(paths)] == [0, 1]


def test_publication_never_overwrites(served_dir, tmp_path):
    witness = _witness(served_dir)
    receipt = eo.publish_witness(tmp_path, witness=witness)
    assert receipt.is_file()
    sidecar = Path(str(receipt) + ".sha256")
    assert sidecar.read_text().strip() == hashlib.sha256(receipt.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="already exists"):
        eo.publish_witness(tmp_path, witness=witness)


def test_publication_refuses_an_unproved_witness(served_dir, tmp_path):
    witness = _witness(served_dir, stamp=False)
    with pytest.raises(ValueError, match="proof"):
        eo.publish_witness(tmp_path, witness=witness)


def test_the_full_pipeline_binds_and_proves(served_dir, tmp_path):
    server = _serve(json.dumps({"data": [{"id": "glm53-artifact"}]}).encode())
    try:
        listener = _listener(_models_url(server))
    finally:
        server.shutdown()
    launch = eo.observe_launch("nonce-abc", [1, 0], lifetime_id="serve-1",
                               clock=lambda: 1001.0)
    assert launch["ranks"] == [0, 1]
    artifacts = [_rank(0, served_dir), _rank(1, served_dir)]
    tokenizer = eo.observe_server_tokenizer(served_dir, lifetime_id="serve-1",
                                            clock=lambda: 1004.0)
    witness = ew.build_witness(listener=listener, launch=launch, artifacts=artifacts,
                               tokenizer=tokenizer)
    assert ew.verify_witness(witness) is not None
    stamped = ew.stamp_byte_proof(witness, served_dir, clock=lambda: 1005.0)
    receipt = eo.publish_witness(tmp_path, witness=stamped)
    loaded = json.loads(receipt.read_text())
    assert ew.verify_witness(loaded) is None
    assert ew.prove_loaded_bytes(loaded, served_dir) is None


def test_the_probe_publishes_one_bound_receipt(served_dir, tmp_path):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "probe_endpoint_witness", str(root / "tools" / "probe_endpoint_witness.py"))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    server = _serve(json.dumps({"data": [{"id": "glm53-artifact"}]}).encode())
    try:
        url = _models_url(server)
        out = tmp_path / "receipts"
        status = probe.main(["--base-url", url, "--attempt-id", "nonce-abc",
                             "--ranks", "0,1", "--artifact-dir", str(served_dir),
                             "--lifetime-id", "serve-1", "--publish-root", str(out)])
        receipts = sorted(out.glob("endpoint_runtime_witness.v1.*.json"))
        again = probe.main(["--base-url", url, "--attempt-id", "nonce-abc",
                            "--ranks", "0,1", "--artifact-dir", str(served_dir),
                            "--lifetime-id", "serve-1", "--publish-root", str(out)])
    finally:
        server.shutdown()
        server.server_close()
    assert status == 0, "the probe must publish one receipt"
    assert again == 0, "a second probe run stamps a new receipt"
    receipts = sorted(out.glob("endpoint_runtime_witness.v1.*.json"))
    assert len(receipts) == 2, "each probe run publishes its own stamped receipt"
    for receipt in receipts:
        loaded = json.loads(receipt.read_text())
        assert ew.verify_witness(loaded) is None
        assert ew.prove_loaded_bytes(loaded, served_dir) is None
    with __import__("pytest").raises(ValueError, match="already exists"):
        eo.publish_witness(out, witness=json.loads(receipts[0].read_text()))


def test_the_producer_needs_no_tessera_serving_import():
    import ast
    from pathlib import Path as _Path
    checked = [_Path(eo.__file__), _Path(ew.__file__)]
    probe = _Path(eo.__file__).resolve().parents[1] / "tools" / "probe_endpoint_witness.py"
    if probe.is_file():
        checked.append(probe)
    verifier = _Path(eo.__file__).resolve().parents[1] / "tools" / "verify_endpoint_witness.py"
    if verifier.is_file():
        checked.append(verifier)
    for path in checked:
        tree = ast.parse(path.read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(entry.name.split(".")[0] for entry in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        assert "torch" not in imported, path
        assert "vllm" not in imported, path
        assert "tessera.serving" not in path.read_text(), path


def _witness(served_dir, stamp=True):
    listener = {"endpoint": "http://10.100.96.2:8142", "served_alias": "glm53-artifact",
                "lifetime_id": "serve-1", "observed_unix": 1000.0}
    launch = {"attempt_id": "nonce-abc", "ranks": [0, 1],
              "lifetime_id": "serve-1", "observed_unix": 1001.0}
    artifacts = [_rank(0, served_dir, clock=lambda: 1002.0),
                 _rank(1, served_dir, clock=lambda: 1003.0)]
    tokenizer = eo.observe_server_tokenizer(served_dir, lifetime_id="serve-1",
                                            clock=lambda: 1004.0)
    witness = ew.build_witness(listener=listener, launch=launch, artifacts=artifacts,
                               tokenizer=tokenizer)
    if stamp:
        return ew.stamp_byte_proof(witness, served_dir, clock=lambda: 1005.0)
    return witness
