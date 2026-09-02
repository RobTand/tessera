"""Is the byte the exporter writes the byte PrismaQuant priced?

PrismaQuant re-encoded every rung the allocator chose (its ``verify_chosen.py``)
and kept the wire beside the score.  Those blobs were produced at Tessera
``3d419e7``; this export runs at HEAD (f3e7d0a + this branch).  Same weights,
same rung, same weights-only settings -- so either the encode is the same object
or the priced bytes are not the served bytes.

Compared three ways, cheapest first: the payload byte count, the packed body /
plane bits, and the decoded FP8 tile the route actually multiplies.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "/home/rob/tessera/.claude/worktrees/agent-a6d34c0d5bba700a6/src")
import torch
from safetensors import safe_open
from tessera.alphabet import E4M3_GRID
from tessera.export import encode_linear_planes
from tessera.unit_artifact import parse_unit_artifact
from tessera.decode import materialize_fp8

PQ = Path("/mnt/shared/tessera-runs/pq-continuous/qwen06b/verify_cache/wire")
MODEL = Path("/home/rob/models/Qwen3-0.6B")
PICKS = [("self_attn.q_proj", 1083), ("self_attn.k_proj", 1083), ("self_attn.v_proj", 1083),
         ("self_attn.o_proj", 934), ("mlp.gate_proj", 1107), ("mlp.up_proj", 1107),
         ("mlp.down_proj", 749)]


def sha(x):
    if isinstance(x, torch.Tensor):
        return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]
    return repr(x)[:16]


def fingerprint(parsed):
    unit = parsed.unit
    out = {}
    for field in ("body_bits", "scale_lut", "scale_global", "scale_row", "scale_channel",
                  "window_table", "table"):
        value = getattr(unit, field, None)
        if value is not None:
            out[field] = sha(value)
    return out


rows = []
with safe_open(str(MODEL / "model.safetensors"), framework="pt") as handle:
    for role, rung in PICKS:
        name = f"model.layers.0.{role}.weight"
        weight = handle.get_tensor(name).to("cuda", torch.float32).contiguous()
        exported, _unit, _forests = encode_linear_planes(
            weight, grid=E4M3_GRID, q256=rung, name=name, verify=True)
        theirs = (PQ / f"model__layers__0__{role.replace('.', '__')}"
                       f"__TESSERA_E4M3_K1_R{rung}.tessera").read_bytes()
        mine_parsed = parse_unit_artifact(bytes(exported.blob), device="cuda")
        their_parsed = parse_unit_artifact(theirs, device="cuda")
        fa, fb = fingerprint(mine_parsed), fingerprint(their_parsed)
        fields = sorted(set(fa) | set(fb))
        planes_same = all(fa.get(f) == fb.get(f) for f in fields)
        ma = materialize_fp8(mine_parsed.unit, mine_parsed.forests, mine_parsed.code)
        mb = materialize_fp8(their_parsed.unit, their_parsed.forests, their_parsed.code)
        tiles_same = all(sha(a) == sha(b) for a, b in zip(ma, mb)) if isinstance(ma, tuple) else sha(ma) == sha(mb)
        rows.append((role, rung, exported.exact_bytes, len(exported.blob), len(theirs),
                     planes_same, tiles_same, fields))
        print(f"{role:20s} R{rung:5d} payload {exported.exact_bytes:9d}B  blob mine {len(exported.blob):9d} "
              f"pq {len(theirs):9d}  planes {'==' if planes_same else 'DIFFER'}  "
              f"decoded tile {'==' if tiles_same else 'DIFFER'}   ({','.join(fields)})", flush=True)

print()
print("all planes identical:", all(r[5] for r in rows))
print("all decoded tiles identical:", all(r[6] for r in rows))
print("payload bytes total:", sum(r[2] for r in rows), "bits:", sum(r[2] for r in rows) * 8)
