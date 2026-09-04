"""Price the derived per-unit LDLQ block on dense Qwen, before any arm is built.

tessera#12 asks whether the LDLQ block lever closes the dense-4 gap on the
served metric, and tessera#60 measured that a *global* flip 32 -> 4 costs ~8x
the encode model-wide while buying 0.4% on GLM experts, where the axis is flat.
``compensate.choose_ldl_block`` exists to resolve exactly that -- spend the
encode where the Hessian says feedback is being skipped -- but nothing in the
export path calls it, so nobody knows what it would pick or cost.

This script answers both, on CPU/GPU arithmetic only, from the capture the
served arms were encoded against.  For every unit it reports
``block_penalty`` at each legal power of two, the block a budget would choose
at ``floor=1`` (the ``encode_unit`` path's real floor, tessera#95), and the
segment count that block implies -- which is what the encode time is
proportional to.

No bytes are written and no arm is built: this is the price tag that decides
whether an arm is worth building.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from tessera.compensate import block_penalty, choose_ldl_block, regularize_hessian

BLOCKS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def role_of(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hessian", type=Path,
                    default=Path("/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"))
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="the LDLQ regulariser the served arms use")
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[1.005, 1.01, 1.02, 1.05, 1.10])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    payload = torch.load(str(args.hessian), map_location="cpu", weights_only=False)
    H_all = payload["H"]
    prov = dict(payload.get("provenance") or {})
    print(f"{len(H_all)} units; provenance "
          f"text={prov.get('text_sha256', '?')[:12]} tokens={prov.get('fit_tokens')}")

    units = []
    t0 = time.time()
    for i, (name, H) in enumerate(sorted(H_all.items())):
        cols = int(H.shape[0])
        H_reg = regularize_hessian(
            H.to(device=args.device, dtype=torch.float32), sigma_reg=args.sigma)
        pen = {}
        for b in BLOCKS:
            if b <= cols and cols % b == 0:
                pen[b] = block_penalty(H_reg, b)
        chosen = {}
        for budget in args.budgets:
            try:
                chosen[budget] = choose_ldl_block(
                    H_reg, max_penalty=budget, floor=1)
            except Exception as exc:  # a budget below the floor's own cost
                chosen[budget] = f"refused: {exc}"
        units.append({"name": name, "cols": cols, "role": role_of(name),
                      "penalty": pen, "chosen": chosen})
        del H_reg
        if (i + 1) % 28 == 0:
            print(f"  [{i + 1}/{len(H_all)}] {name} {time.time() - t0:.0f}s",
                  flush=True)

    # What each policy costs, in the currency the encode is proportional to.
    total_cols = sum(u["cols"] for u in units)
    policies: "dict[str, dict]" = {}
    for b in BLOCKS:
        if all(u["cols"] % b == 0 for u in units):
            segs = sum(u["cols"] // b for u in units)
            policies[f"uniform_b{b}"] = {"segments": segs, "kind": "uniform",
                                         "block": b}
    for budget in args.budgets:
        picks = {u["name"]: u["chosen"][budget] for u in units}
        if any(isinstance(v, str) for v in picks.values()):
            policies[f"budget_{budget}"] = {"refused": True}
            continue
        segs = sum(u["cols"] // picks[u["name"]] for u in units)
        by_role: "dict[str, dict[int, int]]" = {}
        for u in units:
            by_role.setdefault(u["role"], {}).setdefault(picks[u["name"]], 0)
            by_role[u["role"]][picks[u["name"]]] += 1
        policies[f"budget_{budget}"] = {
            "segments": segs, "kind": "derived", "max_penalty": budget,
            "blocks_by_role": {r: {str(k): v for k, v in sorted(d.items())}
                               for r, d in sorted(by_role.items())},
            "distinct_blocks": sorted({picks[u["name"]] for u in units}),
        }

    base = policies["uniform_b32"]["segments"]
    for key, pol in policies.items():
        if "segments" in pol:
            pol["segments_vs_b32"] = pol["segments"] / base

    out = {
        "schema": "tessera.dense4_block_budget_price/1",
        "hessian": str(args.hessian),
        "hessian_provenance": prov,
        "sigma_reg": args.sigma,
        "floor": 1,
        "floor_note": "the encode_unit path's real floor (tessera#95); "
                      "choose_ldl_block has no default floor",
        "total_input_columns": total_cols,
        "units": units,
        "policies": policies,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))

    print(f"\n{'policy':22s} {'segments':>10s} {'x b32':>7s}  blocks")
    for key, pol in policies.items():
        if "segments" not in pol:
            print(f"{key:22s} {'refused':>10s}")
            continue
        blocks = (str(pol["block"]) if pol["kind"] == "uniform"
                  else ",".join(str(b) for b in pol["distinct_blocks"]))
        print(f"{key:22s} {pol['segments']:10d} {pol['segments_vs_b32']:7.3f}  {blocks}")

    print(f"\nper-role penalty at the served default b32, and the b8/b4 predictions")
    roles: "dict[str, list]" = {}
    for u in units:
        roles.setdefault(u["role"], []).append(u)
    print(f"{'role':12s} {'n':>3s} {'cols':>5s} " +
          " ".join(f"{'b' + str(b):>8s}" for b in (4, 8, 16, 32, 64)))
    for role, us in sorted(roles.items()):
        gm = {}
        for b in (4, 8, 16, 32, 64):
            vals = [u["penalty"][b] for u in us if b in u["penalty"]]
            gm[b] = math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else float("nan")
        print(f"{role:12s} {len(us):3d} {us[0]['cols']:5d} " +
              " ".join(f"{gm[b]:8.5f}" for b in (4, 8, 16, 32, 64)))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
