"""Content-bound dense row phases. These files contain no environment values.

Each phase and case carries its own common identities and full rule unit key.
Tensor payloads use raw bytes, explicit dtype and shape, and a SHA-256. No pickle
is read. The phase manifest is the completion marker and is written atomically.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.numerics_rows.common import Invalid, sha256, write_json
from experiments.numerics_rows.launch_state import json_digest


def identities(receipt):
    """Select all common rule bindings, independent of phase-local measurements."""
    result = {section: {key: copy.deepcopy(receipt[section][key]) for key in keys} for section, keys in {
        "tool": ("commit", "tree_sha256"),
        "tessera": ("commit", "contract_sha256", "src_tree_sha256", "kernel_source_sha256", "build_provenance_sha256"),
        "runtime": ("image_digest", "vllm", "torch", "launch_state_names", "launch_state_sha256", "start_environment_spec_sha256", "dependency_math", "dependency_math_sha256"),
        "inputs": ("population_sha256", "selected_modules", "selected_cuts", "case_manifest_sha256", "build_sha256", "q1_table_sha256", "launch_levers_sha256"),
        "rule": ("sha256", "h1_sha256", "frozen_utc"),
    }.items()}
    for name in ("freeze_addendum_sha256",):
        if name in receipt["rule"]:
            result["rule"][name] = receipt["rule"][name]
    result["tessera"]["retained_source"] = copy.deepcopy(receipt["tessera"]["retained_source"])
    result["canary"] = receipt["canary"]
    return result


def unit_scope(unit, wire_sha256):
    from experiments.numerics_rows.row import unit_identity
    return {"unit_key": unit_identity(unit), "module": unit["module"], "cut": copy.deepcopy(unit["tp"]), "wire_sha256": wire_sha256}


class Artifact:
    def __init__(self, root, phase, common):
        self.root, self.phase, self.common = Path(root), phase, copy.deepcopy(common)
        self.root.mkdir(parents=True, exist_ok=True)
        self.document = {"schema": "tessera.dense_row_phase.v1", "phase": phase, "identities": self.common,
                         "completion": False, "scopes": [], "cases": [], "files": {}}

    def tensor(self, tensor):
        import torch
        tensor = tensor.detach().cpu().contiguous()
        data = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest = hashlib.sha256(data).hexdigest()
        name = "bytes/" + digest
        path = self.root / name
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
        if sha256(path) != digest:
            raise Invalid("existing phase payload differs from its content name")
        self.document["files"][name] = digest
        return {"file": name, "sha256": digest, "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}

    def add(self, scope, configuration, kind, tensors, **fields):
        key = json_digest([scope, kind, len(self.document["cases"])])
        case = {"id": key, "identities": copy.deepcopy(self.common), "scope": copy.deepcopy(scope),
                "configuration": copy.deepcopy(configuration), "kind": kind,
                "tensors": {name: self.tensor(value) for name, value in tensors.items()}, **fields}
        self.document["cases"].append(case)
        return case

    def finish(self, **fields):
        self.document.update(fields, completion=True)
        write_json(self.root / "phase.json", self.document)
        return sha256(self.root / "phase.json")


def payload(root, reference):
    import torch
    root = Path(root).resolve()
    path = (root / reference["file"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise Invalid("phase tensor path or digest differs")
    data = bytearray(path.read_bytes())
    if hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise Invalid("phase tensor path or digest differs")
    dtype = getattr(torch, reference["dtype"], None)
    if not isinstance(dtype, torch.dtype):
        raise Invalid("phase tensor dtype is unsupported")
    shape = reference["shape"]
    if not isinstance(shape, list) or any(type(n) is not int or n < 0 for n in shape):
        raise Invalid("phase tensor shape is malformed")
    tensor = torch.frombuffer(data, dtype=dtype).clone() if data else torch.empty(0, dtype=dtype)
    try:
        return tensor.reshape(shape)
    except RuntimeError as error:
        raise Invalid("phase tensor shape differs from its bytes") from error


def read(root, phase, common, scopes):
    root = Path(root)
    raw = (root / "phase.json").read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "tessera.dense_row_phase.v1" or document.get("phase") != phase or document.get("completion") is not True:
        raise Invalid("row phase is missing, incomplete or has the wrong kind")
    if document.get("identities") != common or document.get("scopes") != scopes:
        raise Invalid("row phase common identities, module or cut differ")
    ids = set()
    for case in document["cases"]:
        if case["id"] in ids:
            raise Invalid("row phase repeats a case")
        ids.add(case["id"])
        if case.get("identities") != common or case.get("scope") not in scopes:
            raise Invalid("row phase case has another unit or common identity")
        if case.get("configuration") != case["scope"]["unit_key"][4]:
            raise Invalid("row phase case configuration differs from its unit key")
        for reference in case["tensors"].values():
            if document["files"].get(reference["file"]) != reference["sha256"]:
                raise Invalid("row phase tensor is not in the file manifest")
    for name, digest in document["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or sha256(path) != digest:
            raise Invalid("row phase file digest differs")
    return document, hashlib.sha256(raw).hexdigest()


def check_capture(prepare, capture, prepare_digest):
    if capture.get("prepare_digest") != prepare_digest:
        raise Invalid("capture consumed a different prepare artifact")
    if capture["identities"] != prepare["identities"] or capture["scopes"] != prepare["scopes"]:
        raise Invalid("prepare and capture identities differ")
    expected = {case["id"]: case for case in prepare["cases"]}
    observed = {case["id"]: case for case in capture["cases"]}
    if len(observed) != len(capture["cases"]) or set(observed) != set(expected):
        raise Invalid("capture must contain each prepared case exactly once")
    for key, case in observed.items():
        wanted = expected[key]
        if any(case.get(name) != wanted.get(name) for name in ("scope", "configuration", "kind", "identities")):
            raise Invalid("captured case is attached to another unit")
    return [(case, observed[case["id"]]) for case in prepare["cases"]]


def tensor_subset_equal(canary, counted):
    """Match local case parameters. Check all referenced tensor bytes, dtypes and shapes."""
    roots = [Path(canary), Path(counted)]
    documents = [json.loads((root/'phase.json').read_text()) for root in roots]
    scopes = {json_digest(s) for s in documents[0]['scopes']}
    excluded = {'id', 'ordinal', 'identities', 'tensors', 'seconds', 'utc_start', 'utc_end',
                'elapsed_seconds', 'worker_elapsed_seconds', 'host', 'hostname', 'job_id', 'flux_job_id', 'instance'}
    compared = []
    for root, document in zip(roots, documents):
        if document.get('completion') is not True:
            raise Invalid('E/P prepare is unfinished')
        cases = {}
        for case in document['cases']:
            if json_digest(case['scope']) not in scopes:
                continue
            key = json_digest({k: v for k, v in case.items() if k not in excluded})
            tensors = {}
            for name, ref in case['tensors'].items():
                path = (root/ref['file']).resolve()
                if not path.is_relative_to(root.resolve()) or sha256(path) != ref['sha256']:
                    raise Invalid('E/P tensor bytes differ from their hash')
                tensors[name] = {k: ref[k] for k in ('sha256', 'dtype', 'shape')}
            cases.setdefault(key, []).append(json_digest(tensors))
        compared.append({k: sorted(v) for k, v in cases.items()})
    if not compared[0] or compared[0] != compared[1]:
        raise Invalid('E/P scope-local cases or tensor bytes differ')
    return {'cases': sum(map(len, compared[0].values())), 'tensor_map_sha256': json_digest(compared[0])}
