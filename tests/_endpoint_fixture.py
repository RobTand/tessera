"""Small byte fixtures for contract checks; these are not serving evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from tessera import endpoint_witness as ew


def artifact(root: Path):
    root.mkdir(exist_ok=True)
    payload = b"\x01\x02\x03\x04"
    tensors = {"weight": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]}}
    header = json.dumps(tensors).encode()
    raw = len(header).to_bytes(8, "little") + header + payload
    (root / "model.safetensors").write_bytes(raw)
    backend = {"version": "1.0", "model": {"type": "WordLevel", "vocab": {"<s>": 0, "a": 1}, "unk_token": "<s>"},
               "added_tokens": [], "truncation": None, "padding": None, "normalizer": None,
               "pre_tokenizer": None, "post_processor": None, "decoder": None}
    (root / "tokenizer.json").write_text(json.dumps(backend))
    (root / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<s>"}))
    source = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
              "data_start": 8 + len(header), "tensors": tensors}
    return payload, source, backend


def process(pid):
    return {"host": "fixture", "boot_id": "fixture-boot", "pid": pid, "start_ticks": 10,
            "started_unix": 100.0}


def receipt(root: Path, world=2):
    payload, source, backend = artifact(root)
    loaded_hash = hashlib.sha256(payload).hexdigest()
    model = {"model_path": str(root), "load_started_unix": 200.0, "load_finished_unix": 300.0,
             "files": {"model.safetensors": source},
             "inputs": [{"file": "model.safetensors", "tensor": "weight", "start": 0, "end": 4,
                         "sha256": loaded_hash, "source_sha256": loaded_hash, "target": "weight", "loaded_unix": 250.0}],
             "resident": {"parameter:weight": {"sha256": loaded_hash, "bytes": 4, "dtype": "torch.uint8", "shape": [4]}}}
    token_files = {}
    for name in ("tokenizer.json", "tokenizer_config.json"):
        raw = (root / name).read_bytes()
        token_files[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "content": json.loads(raw)}
    result = {"schema": ew.SCHEMA,
              "listener": {"endpoint": "http://127.0.0.1:8142", "served_alias": "fixture", "owner": process(10)},
              "launch": {"attempt_id": ew.attempt_id(process(10)), "ranks": list(range(world))},
              "lifetime": {"request_id": "fixture-request", "started_unix": 400.0, "finished_unix": 600.0},
              "artifacts": [{"rank": rank, "world_size": world, "owner": process(20 + rank),
                             "request_id": "fixture-request", "observed_unix": 500.0,
                             "models": [copy.deepcopy(model)]} for rank in range(world)],
              "tokenizer": {"path": str(root), "request_id": "fixture-request", "observed_unix": 500.0,
                            "files": token_files, "backend": backend, "vocab": {"<s>": 0, "a": 1},
                            "special_ids": {"bos": 0, "eos": None, "pad": None, "unk": None,
                                            "sep": None, "cls": None, "mask": None}},
              "byte_coverage": {"kind": ew.COVERAGE, "files": ["model.safetensors"], "tensor_payload_bytes": 4},
              "qualification_scope": ew.QUALIFICATION_SCOPE}
    return resign(result)


def resign(value):
    value["fingerprint"] = ew.fingerprint(value)
    return value


def expectations(value):
    """Consumer facts for the fixture, separate from runtime observations."""
    sources = value["artifacts"][0]["models"][0]["files"]
    return {
        "endpoint": value["listener"]["endpoint"],
        "served_alias": value["listener"]["served_alias"],
        "artifacts": {name: {k: fact[k] for k in ("sha256", "bytes")} for name, fact in sources.items()},
        "tokenizer": {
            "files": {name: {k: fact[k] for k in ("sha256", "bytes")}
                      for name, fact in value["tokenizer"]["files"].items()},
            **{k: copy.deepcopy(value["tokenizer"][k]) for k in ("backend", "vocab", "special_ids")},
        },
        "attempt_id": value["launch"]["attempt_id"],
        "ranks": value["launch"]["ranks"].copy(),
    }


def expectation_args(value, root):
    expected = expectations(value)
    artifact_file = root / "expected-artifacts.json"
    tokenizer_file = root / "expected-tokenizer.json"
    artifact_file.write_text(json.dumps(expected["artifacts"]))
    tokenizer_file.write_text(json.dumps(expected["tokenizer"]))
    return ["--expect-endpoint", expected["endpoint"], "--expect-alias", expected["served_alias"],
            "--expect-artifacts", str(artifact_file), "--expect-tokenizer", str(tokenizer_file),
            "--expect-attempt", expected["attempt_id"],
            "--expect-ranks", ",".join(map(str, expected["ranks"]))]
