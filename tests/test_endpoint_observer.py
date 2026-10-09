"""CPU checks for ``tessera.endpoint_observer``: worker joins and receipt publication (tessera#1056).

The observer joins worker observations the serving ranks wrote with the
listener's live reply and the served bytes. Worker files carry rank
identity, served names, loaded wire digests and the server vocabulary
length. The listener alias must equal a served name, or the listener is
not this serve's. These tests use a loopback stub server, worker fixture
files and small served files, so they prove the join rule without a GPU
serve. Fixture observations qualify nothing.
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

def _worker(rank, world=2, **over):
    item = {
        "identity": {"rank": rank, "local_rank": rank, "world_size": world,
                     "rank_source": "torch.distributed",
                     "lifetime_id": "serve-1", "observed_unix": 1000.0},
        "model": {"model": "/art/glm53", "served_names": ["glm53-artifact"],
                  "tokenizer_path": "/art/glm53", "vocab_size": 512,
                  "vocab_source": "server-tokenizer",
                  "lifetime_id": "serve-1", "observed_unix": 1001.0},
        "wires": {"wires": {"model.layers.0.mlp": _digest(f"wire-{rank}".encode())},
                  "coords": {"model.layers.0.mlp": [rank, world]},
                  "modules": 1, "bytes": 32,
                  "lifetime_id": "serve-1", "observed_unix": 1002.0},
    }
    for section, fields in over.items():
        item[section].update(fields)
    return item


def _worker_files(tmp_path, *items):
    paths = []
    for index, item in enumerate(items):
        path = tmp_path / f"worker-{index}.json"
        path.write_text(json.dumps(item))
        paths.append(path)
    return paths


def _workers(tmp_path, world=2):
    return eo.collect_worker_observations(
        _worker_files(tmp_path, *(_worker(rank, world) for rank in range(world))),
        lifetime_id="serve-1")


def test_the_served_alias_comes_from_the_models_reply():
    server = _serve(json.dumps({"data": [{"id": "glm53-artifact"}]}).encode())
    try:
        observed = _listener(_models_url(server))
    finally:
        server.shutdown()
        server.server_close()
    assert observed["served_alias"] == "glm53-artifact"
    assert observed["endpoint"] == _models_url(server)


def test_a_listener_with_no_served_model_is_refused():
    server = _serve(json.dumps({"data": []}).encode())
    try:
        with pytest.raises(ValueError, match="no served model"):
            _listener(_models_url(server))
    finally:
        server.shutdown()
        server.server_close()


def test_a_listener_with_several_models_is_refused():
    server = _serve(json.dumps({"data": [{"id": "a"}, {"id": "b"}]}).encode())
    try:
        with pytest.raises(ValueError, match="several served models"):
            _listener(_models_url(server))
    finally:
        server.shutdown()
        server.server_close()


def test_an_unreachable_listener_is_refused():
    server = _serve(b"{}")
    url = _models_url(server)
    server.shutdown()
    server.server_close()
    with pytest.raises(ValueError, match="cannot read the served models reply"):
        _listener(url)


def test_a_non_http_base_url_is_refused():
    with pytest.raises(ValueError, match="http"):
        _listener("artifact-dir")


def test_worker_files_must_cover_the_whole_world(tmp_path):
    paths = _worker_files(tmp_path, _worker(0))
    with pytest.raises(ValueError, match="not the whole world"):
        eo.collect_worker_observations(paths, lifetime_id="serve-1")


def test_duplicate_worker_ranks_are_refused(tmp_path):
    paths = _worker_files(tmp_path, _worker(0), _worker(0, world=1),
                          _worker(1))
    with pytest.raises(ValueError, match="repeat a rank|disagree on world size"):
        eo.collect_worker_observations(paths, lifetime_id="serve-1")


def test_workers_from_another_lifetime_are_refused(tmp_path):
    other = _worker(1, identity={"lifetime_id": "serve-2"})
    paths = _worker_files(tmp_path, _worker(0), other)
    with pytest.raises(ValueError, match="from lifetime"):
        eo.collect_worker_observations(paths, lifetime_id="serve-1")


def test_workers_that_disagree_on_the_model_are_refused(tmp_path):
    other = _worker(1, model={"model": "/art/other"})
    paths = _worker_files(tmp_path, _worker(0), other)
    with pytest.raises(ValueError, match="disagree on the loaded model"):
        eo.collect_worker_observations(paths, lifetime_id="serve-1")


def test_a_worker_with_no_loaded_wire_is_refused(tmp_path):
    empty = _worker(0, wires={"wires": {}})
    paths = _worker_files(tmp_path, empty, _worker(1))
    with pytest.raises(ValueError, match="no loaded wire"):
        eo.collect_worker_observations(paths, lifetime_id="serve-1")


def test_collected_workers_arrive_sorted(tmp_path):
    paths = _worker_files(tmp_path, _worker(1), _worker(0))
    workers = eo.collect_worker_observations(paths, lifetime_id="serve-1")
    assert [item["identity"]["rank"] for item in workers] == [0, 1]


def test_a_listener_with_another_alias_is_not_this_serve(tmp_path):
    workers = _workers(tmp_path)
    listener = {"endpoint": "http://10.100.96.2:8142", "served_alias": "other",
                "lifetime_id": "serve-1", "observed_unix": 1000.0}
    with pytest.raises(ValueError, match="not a served name"):
        eo.check_listener_owned(listener, workers)


def test_a_listener_with_the_served_alias_is_owned(tmp_path):
    workers = _workers(tmp_path)
    listener = {"endpoint": "http://10.100.96.2:8142", "served_alias": "glm53-artifact",
                "lifetime_id": "serve-1", "observed_unix": 1000.0}
    assert eo.check_listener_owned(listener, workers) is None


def test_launch_ranks_come_from_the_workers(tmp_path):
    workers = _workers(tmp_path)
    launch = eo.observe_launch("nonce-abc", workers, lifetime_id="serve-1",
                               clock=lambda: 1001.0)
    assert launch["ranks"] == [0, 1]
    assert launch["attempt_id"] == "nonce-abc"


def test_rank_bytes_carry_the_loaded_wires(served_dir, tmp_path):
    workers = _workers(tmp_path)
    observed = eo.observe_rank_bytes(workers, served_dir, lifetime_id="serve-1",
                                     clock=lambda: 1002.0)
    assert [item["rank"] for item in observed] == [0, 1]
    for item in observed:
        assert item["files"]["config.json"] == _digest(CONFIG_BYTES)
        assert item["files"]["model.safetensors"] == _digest(WEIGHT_BYTES)
        assert item["sizes"]["config.json"] == len(CONFIG_BYTES)
        loaded = {name: digest for name, digest in item["files"].items()
                  if name.startswith("loaded:")}
        assert len(loaded) == 1
        assert item["sizes"][next(iter(loaded))] == 32
        assert item["bytes"] == sum(item["sizes"].values())


def test_rank_bytes_differ_per_rank_by_wire(served_dir, tmp_path):
    workers = _workers(tmp_path)
    observed = eo.observe_rank_bytes(workers, served_dir, lifetime_id="serve-1",
                                     clock=lambda: 1002.0)
    first = {name: digest for name, digest in observed[0]["files"].items()
             if name.startswith("loaded:")}
    second = {name: digest for name, digest in observed[1]["files"].items()
              if name.startswith("loaded:")}
    assert set(first) == set(second) == {"loaded:model.layers.0.mlp"}
    assert first != second


def test_rank_observation_reads_changed_bytes(served_dir, tmp_path):
    workers = _workers(tmp_path)
    first = eo.observe_rank_bytes(workers, served_dir, lifetime_id="serve-1",
                                  clock=lambda: 1002.0)
    (served_dir / "model.safetensors").write_bytes(WEIGHT_BYTES + b"x")
    second = eo.observe_rank_bytes(workers, served_dir, lifetime_id="serve-1",
                                   clock=lambda: 1003.0)
    assert second[0]["files"]["model.safetensors"] == _digest(WEIGHT_BYTES + b"x")
    assert first[0]["files"]["model.safetensors"] != second[0]["files"]["model.safetensors"]


def test_an_empty_served_directory_is_refused(tmp_path):
    workers = _workers(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        eo.observe_rank_bytes(workers, empty, lifetime_id="serve-1")


def test_the_vocab_size_comes_from_the_server_tokenizer(served_dir, tmp_path):
    workers = _workers(tmp_path)
    observed = eo.observe_server_tokenizer(workers, served_dir, lifetime_id="serve-1",
                                           clock=lambda: 1004.0)
    assert observed["vocab_size"] == 512
    assert observed["vocab_source"] == "server-tokenizer"
    assert observed["files"] == {"tokenizer.json": _digest(TOKENIZER_BYTES)}


def test_ranks_that_disagree_on_vocab_size_are_refused(served_dir, tmp_path):
    other = _worker(1, model={"vocab_size": 513})
    paths = _worker_files(tmp_path, _worker(0), other)
    workers = eo.collect_worker_observations(paths, lifetime_id="serve-1")
    with pytest.raises(ValueError, match="disagree on vocabulary length"):
        eo.observe_server_tokenizer(workers, served_dir, lifetime_id="serve-1")


def test_a_tokenizer_without_a_served_file_is_refused(tmp_path):
    empty = tmp_path / "bare"
    empty.mkdir()
    (empty / "config.json").write_bytes(CONFIG_BYTES)
    workers = _workers(tmp_path)
    with pytest.raises(ValueError, match="no tokenizer file"):
        eo.observe_server_tokenizer(workers, empty, lifetime_id="serve-1")


def test_publication_never_overwrites(served_dir, tmp_path):
    witness = _witness(served_dir, tmp_path)
    eo.publish_witness(tmp_path, witness=witness, served_dir=served_dir)
    with pytest.raises(ValueError, match="already exists"):
        eo.publish_witness(tmp_path, witness=witness, served_dir=served_dir)


def test_publication_refuses_an_unproved_witness(served_dir, tmp_path):
    witness = _witness(served_dir, tmp_path, stamp=False)
    with pytest.raises(ValueError, match="proof|structural agreement"):
        eo.publish_witness(tmp_path, witness=witness, served_dir=served_dir)


def test_publication_reproves_the_bytes(served_dir, tmp_path):
    witness = _witness(served_dir, tmp_path)
    (served_dir / "model.safetensors").write_bytes(WEIGHT_BYTES + b"x")
    with pytest.raises(ValueError, match="byte proof|differs"):
        eo.publish_witness(tmp_path, witness=witness, served_dir=served_dir)


def test_the_full_pipeline_binds_and_proves(served_dir, tmp_path):
    server = _serve(json.dumps({"data": [{"id": "glm53-artifact"}]}).encode())
    try:
        url = _models_url(server)
        workers = _workers(tmp_path)
        listener = _listener(url)
        eo.check_listener_owned(listener, workers)
        launch = eo.observe_launch("nonce-abc", workers, lifetime_id="serve-1",
                                   clock=lambda: 1001.0)
        artifacts = eo.observe_rank_bytes(workers, served_dir, lifetime_id="serve-1",
                                          clock=lambda: 1002.0)
        tokenizer = eo.observe_server_tokenizer(workers, served_dir,
                                                lifetime_id="serve-1",
                                                clock=lambda: 1004.0)
        witness = ew.build_witness(listener=listener, launch=launch,
                                   artifacts=artifacts, tokenizer=tokenizer)
        assert ew.verify_witness(witness, served_dir=served_dir) is not None
        stamped = ew.stamp_byte_proof(witness, served_dir, clock=lambda: 1005.0)
        receipt = eo.publish_witness(tmp_path, witness=stamped, served_dir=served_dir)
        loaded = json.loads(receipt.read_text())
        assert ew.verify_witness(loaded, served_dir=served_dir) is None
        assert ew.prove_loaded_bytes(loaded, served_dir) is None
    finally:
        server.shutdown()
        server.server_close()


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
        paths = _worker_files(tmp_path, _worker(0), _worker(1))
        status = probe.main(["--base-url", url, "--attempt-id", "nonce-abc",
                             "--worker-observations", ",".join(str(path) for path in paths),
                             "--artifact-dir", str(served_dir),
                             "--lifetime-id", "serve-1", "--publish-root", str(out)])
        receipts = sorted(out.glob("endpoint_runtime_witness.v1.*.json"))
    finally:
        server.shutdown()
        server.server_close()
    assert status == 0, "the probe must publish one receipt"
    assert len(receipts) == 1, "one probe run publishes one receipt"
    loaded = json.loads(receipts[0].read_text())
    assert ew.verify_witness(loaded, served_dir=served_dir) is None
    assert ew.prove_loaded_bytes(loaded, served_dir) is None
    with pytest.raises(ValueError, match="already exists"):
        eo.publish_witness(out, witness=loaded, served_dir=served_dir)


def test_the_probe_refuses_a_replaced_listener(served_dir, tmp_path):
    import importlib.util
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "probe_endpoint_witness", str(root / "tools" / "probe_endpoint_witness.py"))
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    server = _serve(json.dumps({"data": [{"id": "replacement"}]}).encode())
    try:
        url = _models_url(server)
        out = tmp_path / "receipts"
        paths = _worker_files(tmp_path, _worker(0), _worker(1))
        status = probe.main(["--base-url", url, "--attempt-id", "nonce-abc",
                             "--worker-observations", ",".join(str(path) for path in paths),
                             "--artifact-dir", str(served_dir),
                             "--lifetime-id", "serve-1", "--publish-root", str(out)])
    finally:
        server.shutdown()
        server.server_close()
    assert status == 4, "a listener with another alias is another serve"


def test_the_probe_refuses_an_unrelated_directory(served_dir, tmp_path):
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
        other = tmp_path / "other"
        other.mkdir()
        (other / "config.json").write_bytes(b"{}")
        paths = _worker_files(tmp_path, _worker(0), _worker(1))
        status = probe.main(["--base-url", url, "--attempt-id", "nonce-abc",
                             "--worker-observations", ",".join(str(path) for path in paths),
                             "--artifact-dir", str(other),
                             "--lifetime-id", "serve-1", "--publish-root", str(out)])
    finally:
        server.shutdown()
        server.server_close()
    assert status == 4, "a directory without the tokenizer file proves nothing"


def test_the_join_needs_no_tessera_serving_import():
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


def _witness(served_dir, tmp_path, stamp=True):
    workers = _workers(tmp_path)
    listener = {"endpoint": "http://10.100.96.2:8142", "served_alias": "glm53-artifact",
                "lifetime_id": "serve-1", "observed_unix": 1000.0}
    eo.check_listener_owned(listener, workers)
    launch = eo.observe_launch("nonce-abc", workers, lifetime_id="serve-1",
                               clock=lambda: 1001.0)
    artifacts = eo.observe_rank_bytes(workers, served_dir, lifetime_id="serve-1",
                                      clock=lambda: 1002.0)
    tokenizer = eo.observe_server_tokenizer(workers, served_dir, lifetime_id="serve-1",
                                            clock=lambda: 1004.0)
    witness = ew.build_witness(listener=listener, launch=launch, artifacts=artifacts,
                               tokenizer=tokenizer)
    if stamp:
        return ew.stamp_byte_proof(witness, served_dir, clock=lambda: 1005.0)
    return witness
