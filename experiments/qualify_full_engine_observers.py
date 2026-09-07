"""Tiny admitted CUDA controls for the full-engine observers, without pricing."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.full_engine_resources import FullEngineResourceRecorder, TensorOwner, BlasWorkspaceObserver


def run(args):
    fixture = hashlib.sha256(b"tiny CUDA argument and BLAS map observation control v1").hexdigest()
    identity = {"schema": "tessera.full_engine_resource_identity.v1",
                **{name: fixture for name in ("model_sha256", "configuration_sha256", "runtime_manifest_sha256",
                                             "assignment_sha256", "canonical_units_sha256", "workload_sha256")},
                "device_id": 0, "device_uuid": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()}
    recorder = FullEngineResourceRecorder(args.collector, identity, max_checkpoints=3)
    torch = recorder._torch
    first = torch.ones((32, 32), device="cuda")
    output = first @ first
    torch.cuda.synchronize()
    observer = BlasWorkspaceObserver(args.workspaces, args.workspaces_sha256)
    workspaces = list(observer.owners())
    assert workspaces, "Torch BLAS map was empty after actual matrix multiplication"
    owners = [TensorOwner("fixture:input", "shared", first, "tiny matrix control"),
              TensorOwner("fixture:output", "shared", output, "tiny matrix control"), *workspaces]
    recorder.snapshot("after_matmul", owners=owners)
    cudart = ctypes.CDLL("libcudart.so.13")
    host, device = ctypes.c_void_p(), ctypes.c_void_p()
    cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    cudart.cudaHostGetDevicePointer.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint]
    cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    assert cudart.cudaFree(None) == 0
    assert cudart.cudaHostAlloc(ctypes.byref(host), 128, 2) == 0
    assert cudart.cudaHostGetDevicePointer(ctypes.byref(device), host, 0) == 0
    recorder.snapshot("host_mapping_live", owners=owners)
    assert cudart.cudaFreeHost(host) == 0
    receipt = recorder.finish(args.output / "capture", owners=owners)
    raw = json.loads((args.output / "capture/capture.json").read_text())
    domains = receipt["cuda_argument_domains"]
    assert domains["status"] == "observed_argument_domains", domains
    assert domains["null_device_frees"]
    allocations = [row for row in domains["host_allocations"] if row["address"] == host.value and row["bytes"] == 128]
    assert len(allocations) == 1 and allocations[0]["freed_ns"] is not None
    assert any(row["host_address"] == host.value and row["device_address"] == device.value for row in domains["host_mappings"])
    active = {block["address"]: block["requested_size"] for segment in raw["torch_snapshot"]["segments"]
              for block in segment["blocks"] if block["state"] == "active_allocated"}
    assert all(active.get(owner.address) == owner.bytes for owner in workspaces), (active, workspaces)
    result = {"schema": "tessera.full_engine_observer_qualification.v1", "status": "passed",
              "scope": "tiny actual CUDA pointer/host lifetime and Torch BLAS-map controls; no engine pricing",
              "collector_sha256": hashlib.sha256(args.collector.read_bytes()).hexdigest(),
              "workspace_observer_sha256": args.workspaces_sha256,
              "host_address": host.value, "device_address": device.value,
              "host_lifetime": allocations[0], "blas_workspaces": [owner.__dict__ for owner in workspaces],
              "raw_capture_sha256": hashlib.sha256((args.output / "capture/capture.json").read_bytes()).hexdigest(),
              "full_model_fixed_resources_complete": False}
    path = args.output / "qualification.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"artifact": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "status": "passed"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--workspaces", type=Path, required=True)
    parser.add_argument("--workspaces-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run(args)
