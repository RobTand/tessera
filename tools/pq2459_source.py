"""Load the accepted producer source from a captured Git archive."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tarfile
import tempfile




def add_source_arguments(parser):
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--source-archive-sha256", required=True)


def unpack_archive(archive, expected_sha256):
    archive = Path(archive).resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise ValueError("source archive bytes differ from the declared input")
    root = Path(tempfile.mkdtemp(prefix="pq2459-source-"))
    with tarfile.open(archive, "r:") as stream:
        commit = stream.pax_headers.get("comment")
        if not isinstance(commit, str) or len(commit) != 40:
            raise ValueError("source archive lacks its Git commit record")
        stream.extractall(root, filter="data")
    return {"root": str(root), "commit": commit, "archive": str(archive), "archive_sha256": digest}


def activate_source(archive, expected_sha256):
    if any(name == "tessera" or name.startswith("tessera.") for name in sys.modules):
        raise RuntimeError("select the immutable source before any Tessera import")
    record = unpack_archive(archive, expected_sha256)
    root = Path(record["root"])
    src = root / "src"
    if not (src / "tessera/serving/runtime_contract.json").is_file():
        raise ValueError("source archive lacks the producer contract")
    sys.path.insert(0, str(src))
    os.environ["PYTHONPATH"] = str(src)
    os.environ["TESSERA_GIT"] = record["commit"]
    import tessera
    from tessera.cached_unit import encoder_source_sha256
    from tessera.serving.source_identity import serving_source_sha256
    contract = src / "tessera/serving/runtime_contract.json"
    return {
        **record,
        "package_root": str(Path(tessera.__file__).resolve().parent),
        "src": str(src), "source_algorithm": "tessera.package_source.v1",
        "serving_source_sha256": serving_source_sha256(src),
        "producer_source_algorithm": "tessera.encoder_source.v1",
        "producer_source_sha256": encoder_source_sha256(),
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
    }
