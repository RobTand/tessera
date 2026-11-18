"""Write ordered phase records and verify the bytes consumed by each reader."""
import copy
import hashlib
import json
from itertools import chain, zip_longest
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
    result["rule"].update({k: receipt["rule"][k] for k in ("freeze_addendum_sha256",) if k in receipt["rule"]})
    result["tessera"]["retained_source"] = copy.deepcopy(receipt["tessera"]["retained_source"])
    result["canary"] = receipt["canary"]
    return result


def unit_scope(unit, wire_sha256):
    from experiments.numerics_rows.row import unit_identity
    return {"unit_key": unit_identity(unit), "module": unit["module"], "cut": copy.deepcopy(unit["tp"]), "wire_sha256": wire_sha256}


def records(root, name, document=None):
    count, digest = 0, hashlib.sha256()
    with (Path(root) / (name + ".jsonl")).open("rb") as stream:
        for line in stream:
            digest.update(line)
            count += 1
            yield json.loads(line)
    if document is not None and document["streams"][name] != {"count": count, "sha256": digest.hexdigest()}:
        raise Invalid("row phase stream digest or count differs")


class Artifact:
    def __init__(self, root, phase, common):
        self.root, self.common = Path(root), copy.deepcopy(common)
        self.root.mkdir(parents=True, exist_ok=True)
        if (self.root / "phase.json").exists() or any(self.root.glob("*.jsonl")):
            raise Invalid("use a new phase attempt directory")
        self.case_ids = hashlib.sha256()
        self.document = {"schema": "tessera.dense_row_phase.v2", "phase": phase, "identities": self.common,
                         "completion": False, "scopes": [], "ordinal_start": 0, "case_set": "all", "streams": {}}

    def record(self, name, value):
        if name == "cases":
            self.case_ids.update((value["id"] + "\n").encode("ascii"))
        with (self.root / (name + ".jsonl")).open("a") as stream:
            stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
        self.document["streams"].setdefault(name, {"count": 0})["count"] += 1

    def file(self, name):
        path = self.root / name
        self.record("files", {"file": str(name), "bytes": path.stat().st_size, "sha256": sha256(path)})

    def tensor(self, tensor):
        import torch
        tensor = tensor.detach().cpu().contiguous()
        data = memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B")
        digest = hashlib.sha256(data).hexdigest()
        name = "bytes/" + digest
        path = self.root / name
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            path.write_bytes(data)
        if sha256(path) != digest:
            raise Invalid("existing phase payload differs from its content name")
        self.file(name)
        return {"file": name, "sha256": digest, "dtype": str(tensor.dtype).removeprefix("torch."), "shape": list(tensor.shape)}

    def add(self, scope, configuration, kind, tensors, **fields):
        ordinal = self.document["ordinal_start"] + self.document["streams"].get("cases", {}).get("count", 0)
        case = {"id": json_digest([scope, kind, ordinal]), "ordinal": ordinal, "identities": self.common,
                "scope": scope, "configuration": configuration, "kind": kind,
                "tensors": {name: self.tensor(value) for name, value in sorted(tensors.items())}, **fields}
        self.record("cases", case)
        return case

    def finish(self, **fields):
        if self.document.get("expected_cases", self.document["streams"].get("cases", {}).get("count", 0)) != self.document["streams"].get("cases", {}).get("count", 0):
            raise Invalid("prepare differs from the deterministic full row case count")
        for name, record in self.document["streams"].items():
            record["sha256"] = sha256(self.root / (name + ".jsonl"))
        self.document.update(fields, completion=True, case_ids_sha256=self.case_ids.hexdigest())
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


def cases(root, document):
    count, digest, previous = 0, hashlib.sha256(), -1
    positions = iter(chain.from_iterable(range(start, start + length) for start, length in document.get("ranges", [[document["ordinal_start"], document["streams"].get("cases", {}).get("count", 0)]])))
    for case in records(root, "cases", document):
        ordinal = next(positions, None)
        if type(case.get("ordinal")) is not int or case["ordinal"] != ordinal or case["ordinal"] <= previous or case.get("id") != json_digest([case["scope"], case["kind"], ordinal]):
            raise Invalid("row phase must contain each ordered case exactly once")
        previous = case["ordinal"]
        if case.get("identities") != document["identities"] or case.get("scope") not in document["scopes"]:
            raise Invalid("row phase case has another unit or common identity")
        if case.get("configuration") != case["scope"]["unit_key"][4]:
            raise Invalid("row phase case configuration differs from its unit key")
        count += 1
        digest.update((case["id"] + "\n").encode("ascii"))
        yield case
    if next(positions, None) is not None or count != document.get("expected_cases", count) or count != document["streams"].get("cases", {}).get("count", 0) or digest.hexdigest() != document["case_ids_sha256"]:
        raise Invalid("row phase must contain each prepared case exactly once")


def read(root, phase, common, scopes, *, check_case=lambda case: None):
    root = Path(root)
    raw = (root / "phase.json").read_bytes()
    document = json.loads(raw)
    if document.get("schema") != "tessera.dense_row_phase.v2" or document.get("phase") != phase or document.get("completion") is not True:
        raise Invalid("row phase is missing, incomplete or has the wrong kind")
    if document.get("identities") != common or document.get("scopes") != scopes:
        raise Invalid("row phase common identities, module or cut differ")
    if set(document["streams"]) - {"cases", "arrays", "files"}:
        raise Invalid("row phase has an unknown stream")
    def files():
        for reference in records(root, "files", document) if "files" in document["streams"] else ():
            path = (root / reference["file"]).resolve()
            if not path.is_relative_to(root.resolve()) or path.stat().st_size != reference["bytes"] or sha256(path) != reference["sha256"]:
                raise Invalid("row phase file digest differs")
            yield reference
    inventory = files()
    for case in cases(root, document):
        check_case(case)
        for ref in case.get("tensors", {}).values():
            item = next(inventory, None)
            if item is None or any(item[k] != ref[k] for k in ("file", "sha256")):
                raise Invalid("row phase tensor is not in the ordered file manifest")
    for _ in chain(inventory, records(root, "arrays", document) if "arrays" in document["streams"] else ()):
        pass
    return document, hashlib.sha256(raw).hexdigest()


def check_capture(prepare, capture, prepare_digest, prepare_root, capture_root):
    if capture.get("prepare_digest") != prepare_digest:
        raise Invalid("capture consumed a different prepare artifact")
    if any(capture.get(k) != prepare.get(k) for k in ("identities", "scopes", "ordinal_start", "case_set")):
        raise Invalid("prepare and capture identities differ")
    for wanted, observed in zip_longest(cases(prepare_root, prepare), cases(capture_root, capture)):
        if wanted is None or observed is None or any(observed.get(k) != wanted.get(k) for k in
                ("id", "ordinal", "scope", "configuration", "kind", "identities", "category")):
            raise Invalid("capture must contain each prepared case exactly once")
        yield wanted, observed


def tensor_subset_equal(canary, counted):
    """Compare scope-local case parameters and raw bytes in order."""
    roots = [Path(canary), Path(counted)]
    documents = [json.loads((root / "phase.json").read_text()) for root in roots]
    scopes = documents[0]["scopes"]
    excluded = {"id", "ordinal", "identities", "tensors", "seconds", "utc_start", "utc_end",
                "elapsed_seconds", "worker_elapsed_seconds", "host", "hostname", "job_id", "flux_job_id", "instance"}
    def selected(root, document):
        if document.get("completion") is not True:
            raise Invalid("E/P prepare is unfinished")
        for case in cases(root, document):
            if case["scope"] not in scopes:
                continue
            tensors = {}
            for name, ref in case["tensors"].items():
                path = (root / ref["file"]).resolve()
                if not path.is_relative_to(root.resolve()) or sha256(path) != ref["sha256"]:
                    raise Invalid("E/P tensor bytes differ from their hash")
                tensors[name] = {k: ref[k] for k in ("sha256", "dtype", "shape")}
            yield {k: v for k, v in case.items() if k not in excluded}, tensors
    count, digest = 0, hashlib.sha256()
    for first, second in zip_longest(*(selected(r, d) for r, d in zip(roots, documents))):
        if first is None or first != second:
            raise Invalid("E/P scope-local cases or tensor bytes differ")
        digest.update(json_digest(first).encode())
        count += 1
    if not count:
        raise Invalid("E/P scope-local cases are missing")
    return {"cases": count, "tensor_map_sha256": digest.hexdigest()}
