#!/usr/bin/env python3
"""Run the source exporter with durable unit and shard progress."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    import tessera.export_serving as exporter
    helper = os.environ.get("PRISMABUILD_ACTION_PROGRESS_HELPER")
    if helper is None:
        raise RuntimeError("construction requires the admitted progress contract")
    commit = runpy.run_path(helper)["commit"]
    output = Path(sys.argv[2])
    checkpoint = output.with_name(output.name + "-unit-checkpoints")
    checkpoint.mkdir(parents=True, exist_ok=False)
    count = 0
    original_encode = exporter.encode_linear_planes
    original_save = exporter.save_serving_shard

    def fence_directory(path):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def encoded(*args, **kwargs):
        nonlocal count
        result = original_encode(*args, **kwargs)
        name = kwargs["name"]
        blob = result[0].blob
        path = checkpoint / (hashlib.sha256(name.encode()).hexdigest() + ".tessera")
        with path.open("xb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        fence_directory(checkpoint)
        count += 1
        commit(count, "construct", unit="durable checkpoints")
        print(json.dumps({"source_unit": name, "wire_sha256": hashlib.sha256(blob).hexdigest(),
                          "wire_bytes": len(blob), "completed_checkpoints": count}), flush=True)
        return result

    def saved(payload, path):
        nonlocal count
        original_save(payload, path)
        with Path(path).open("rb") as handle:
            os.fsync(handle.fileno())
        fence_directory(Path(path).parent)
        count += 1
        commit(count, "construct", unit="durable checkpoints")
        print(json.dumps({"published_shard": str(path), "completed_checkpoints": count}), flush=True)

    exporter.encode_linear_planes = encoded
    exporter.save_serving_shard = saved
    try:
        sys.argv = ["tessera.export_serving", *sys.argv[1:]]
        exporter.main()
    finally:
        exporter.encode_linear_planes = original_encode
        exporter.save_serving_shard = original_save
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
