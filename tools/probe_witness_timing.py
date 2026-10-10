"""Timing probe for tessera#1171: witness per-request hash cost at population scale.

Model: the routed A4 TP2 checkpoint merged-4c384e60 from the
glm-canonical-census-20260908 first-artifact-exports bundle. The probe loads
shard part-00045-model-00104-of-00120.safetensors (5,368,754,216 bytes on disk,
5,368,709,120 resident tensor bytes). That shard is the largest single
safetensors file the CUDA population reads: the a4, a8 and a16 merged bundles
each peak at this same 5,368,754,216-byte shard, and the largest packed-expert
source shard (Qwen3.8-Flash-Next model-00062-of-00131) is 3,510,240,000 bytes.
No population test resident-loads a full 166 GiB bundle; the per-request cost
scales with resident bytes, so this shard bounds it.

Endpoint path: the probe calls tessera.serving.endpoint_runtime.resident_bytes,
the exact function observe_worker calls on every witness request
(endpoint_runtime.py:286, resident_bytes(model) == record["resident"]). The
30 s bound is fetch_witness timeout_s=30 (endpoint_observer.py:23): the HTTP
GET/POST the listener must answer inside that window triggers the collective
RPC that runs this re-hash on each rank. The probe times that re-hash.
Untracked throwaway: not part of the suite, deleted after the measurement lands.
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
    print(f"PROBE model=merged-4c384e60 shard=part-00045-model-00104-of-00120 "
          f"resident_bytes={resident} file_bytes={file_bytes} elapsed_s={elapsed:.1f} "
          f"timeout_s={TIMEOUT_S} within={elapsed < TIMEOUT_S}")
    assert elapsed < TIMEOUT_S, "per-request resident re-hash exceeds the fetch timeout"


if __name__ == "__main__":
    resident, file_bytes, elapsed = measure()
    print(f"PROBE model=merged-4c384e60 shard=part-00045-model-00104-of-00120 "
          f"resident_bytes={resident} file_bytes={file_bytes} elapsed_s={elapsed:.1f} "
          f"timeout_s={TIMEOUT_S} within={elapsed < TIMEOUT_S}")
    raise SystemExit(0 if elapsed < TIMEOUT_S else 4)
