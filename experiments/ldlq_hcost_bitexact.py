"""Issue #13: does the fast TCQ path write the same bytes as the old one?

Runs one unit's encode and dumps every field of the ``EncodedUnit`` plus the
``encoder_profile_id``, so the same script can be run against the PRE-change
tree and the post-change one and the two dumps compared field by field.  This
is the demonstration, not an assertion: the comparison is ``torch.equal`` on
every plane and ``==`` on the ``sse`` float.

usage:
  # pre-change tree
  PYTHONPATH=<base>/src  ldlq_hcost_bitexact.py ... --dump base.pt
  # post-change tree
  PYTHONPATH=src         ldlq_hcost_bitexact.py ... --dump new.pt
  ldlq_hcost_bitexact.py --compare base.pt new.pt
"""
import argparse
import sys

import torch


def load_unit(model, hfile, unit, grid_name, q256, block, sigma, arms):
    from safetensors import safe_open
    from tessera.alphabet import SERIALISABLE_GRIDS
    from tessera.compensate import block_ldl, regularize_hessian
    from tessera.export import encode_linear_planes, wire_recipe

    grid = [g for g in SERIALISABLE_GRIDS.values() if g.name == grid_name][0]
    payload = torch.load(hfile, map_location="cpu", weights_only=False)
    H = payload["H"][unit].to("cuda", torch.float32)
    del payload
    with safe_open(f"{model}/model.safetensors", framework="pt") as f:
        W = f.get_tensor(unit + ".weight").to("cuda", torch.float32).contiguous()
    L = block_ldl(regularize_hessian(H, sigma_reg=sigma), block)
    hn = (H.diagonal() / H.diagonal().mean()).clone()

    out = {}
    kws = {"weights_only": {},
           "ldlq": dict(ldl=L, ldl_block=block, refit_metric=hn)}
    for arm in arms:
        exported, encoded, _ = encode_linear_planes(
            W, grid=grid, q256=q256, name=unit, verify=False, **kws[arm])
        rec = {}
        for which, obj in (("exported", exported), ("encoded", encoded)):
            for field in obj.__dataclass_fields__:
                v = getattr(obj, field)
                key = f"{which}.{field}"
                if isinstance(v, torch.Tensor):
                    rec[key] = v.detach().cpu().clone()
                elif isinstance(v, (int, float, str, bool, type(None), tuple)):
                    rec[key] = v
                else:
                    rec[key] = repr(v)
        out[arm] = rec
    out["_recipe"] = repr(wire_recipe(grid, q256))
    return out


def compare(a_path, b_path):
    a = torch.load(a_path, map_location="cpu", weights_only=False)
    b = torch.load(b_path, map_location="cpu", weights_only=False)
    bad = 0
    for arm in sorted(k for k in a if not k.startswith("_")):
        for field in sorted(a[arm]):
            x, y = a[arm][field], b[arm][field]
            if isinstance(x, torch.Tensor):
                same = (x.shape == y.shape and x.dtype == y.dtype
                        and torch.equal(x, y))
            else:
                same = x == y and type(x) is type(y)
            if not same:
                bad += 1
                print(f"DIFF  {arm}.{field}: {x!r:.80} != {y!r:.80}")
        print(f"{arm}: {len(a[arm])} fields, "
              f"{sum(1 for f in a[arm] if isinstance(a[arm][f], torch.Tensor))} tensors")
    for k in ("_recipe",):
        if a[k] != b[k]:
            bad += 1
            print(f"DIFF {k}: {a[k]} != {b[k]}")
    print("BIT-EXACT" if bad == 0 else f"{bad} FIELDS DIFFER")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--model"); ap.add_argument("--h"); ap.add_argument("--unit")
    ap.add_argument("--grid", default="E2M1x2"); ap.add_argument("--q256", type=int, default=896)
    ap.add_argument("--block", type=int, default=32); ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--arms", default="weights_only,ldlq")
    ap.add_argument("--dump")
    a = ap.parse_args()
    if a.compare:
        sys.exit(1 if compare(*a.compare) else 0)
    rec = load_unit(a.model, a.h, a.unit, a.grid, a.q256, a.block, a.sigma,
                    a.arms.split(","))
    torch.save(rec, a.dump)
    print(f"-> {a.dump}")


if __name__ == "__main__":
    main()
