"""Timing probe for tessera#1171: witness per-request hash cost at population scale.

Runs the real producer path (tessera.serving.endpoint_runtime.resident_bytes)
over the full bytes of the largest checkpoint file the CUDA population reads,
and times it against the 30 s fetch_witness timeout. Untracked throwaway: not
part of the suite, deleted after the measurement lands.
"""
from __future__ import annotations

import time
from pathlib import Path

SHARD = Path(
    "/mnt/shared/tessera-measurements/glm-canonical-census-20260908/"
    "first-artifact-exports/a4/merged-4c384e60/"
    "part-00045-model-00104-of-00120.safetensors"
)
TIMEOUT_S = 30.0


def measure():
    torch = __import__("torch")
    assert torch.cuda.is_available(), "probe needs the device it prices"
    from safetensors import safe_open

    from tessera.serving import endpoint_runtime as er

    assert SHARD.exists(), f"largest touched shard absent: {SHARD}"
    file_bytes = SHARD.stat().st_size
    model = torch.nn.Module()
    resident = 0
    with safe_open(str(SHARD), framework="pt") as handle:
        for name in handle.keys():
            tensor = handle.get_tensor(name).cuda()
            model.register_parameter(name.replace(".", "_"), torch.nn.Parameter(tensor, requires_grad=False))
            resident += tensor.numel() * tensor.element_size()
    assert resident > 0, "shard placed no resident bytes"
    started = time.time()
    facts = er.resident_bytes(model)
    elapsed = time.time() - started
    assert sum(fact["bytes"] for fact in facts.values()) == resident, "hash covered every resident byte"
    return resident, file_bytes, elapsed


def test_resident_hash_at_population_scale():
    resident, file_bytes, elapsed = measure()
    print(f"PROBE resident_bytes bytes={resident} file_bytes={file_bytes} elapsed_s={elapsed:.1f} "
          f"timeout_s={TIMEOUT_S} within={elapsed < TIMEOUT_S}")
    assert elapsed < TIMEOUT_S, "per-request resident re-hash exceeds the fetch timeout"


if __name__ == "__main__":
    resident, file_bytes, elapsed = measure()
    print(f"PROBE resident_bytes bytes={resident} file_bytes={file_bytes} elapsed_s={elapsed:.1f} "
          f"timeout_s={TIMEOUT_S} within={elapsed < TIMEOUT_S}")
    raise SystemExit(0 if elapsed < TIMEOUT_S else 4)
