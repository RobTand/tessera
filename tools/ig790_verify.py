"""Caller reads for the issue 790 measurement; not a production API."""
from pathlib import Path


def verify_projected_units(source, projection, stack):
    """Authenticate every consumed shard through the existing caller owner."""
    from safetensors import safe_open
    from prismaquant.tessera_calibration_cache import CaptureSourceAuthentication

    source = Path(source)
    units = projection["stacks"][stack]["units"]
    tensors = projection["source"]["tensors"]
    total = 0
    with CaptureSourceAuthentication.recording(source, projection["source"]) as owner:
        for rec in units:
            path = source / tensors[rec["source_tensor"]]
            with owner.safe_open(safe_open, path, framework="pt", device="cpu") as handle:
                tensor = handle.get_tensor(rec["source_tensor"])
                assert list(tensor.shape) == [rec["rows"], rec["cols"]]
                total += tensor.numel() * tensor.element_size()
                del tensor
        receipt = owner.receipt()
        for row in receipt["verified_files"]:
            expected = projection["source"]["files"][row["name"]]
            actual = row["sha256"]
            if actual != expected:
                raise RuntimeError(f"calibration source differs from census producer: {row['name']}")
            print("CALLER-COMPARE", row["name"], "expected", expected,
                  "actual", actual, "equal", actual == expected)
        print("CALLER-VERIFY units=%d tensor_bytes=%d" % (len(units), total))
        return receipt
