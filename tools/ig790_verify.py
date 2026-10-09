"""Caller reads for the issue 790 measurement; not a production API."""
import hashlib
from pathlib import Path


def verify_projected_units(source, projection, stack):
    """Read the selected projection units and report their tensor digests."""
    from safetensors import safe_open

    source = Path(source)
    units = projection["stacks"][stack]["units"][:4]
    tensors = projection["source"]["tensors"]
    total = 0
    for rec in units:
        path = source / tensors[rec["source_tensor"]]
        with safe_open(path, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(rec["source_tensor"])
            assert list(tensor.shape) == [rec["rows"], rec["cols"]]
            raw = bytes(tensor.untyped_storage())
            total += len(raw)
            print("VERIFY", rec["tensor"], "bytes", len(raw),
                  "sha256", hashlib.sha256(raw).hexdigest()[:16])
    print("CALLER-VERIFY units=4 tensor_bytes=%d" % total)
