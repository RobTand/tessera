"""Predict the LDLQ block-size penalty from the Hessian alone (tessera#60).

Derivation.  LDLQ over input columns with ``H = L D L^T`` (L unit lower) has
proxy loss ``sum_j D_jj ||eta_j||^2`` when every column sees every earlier
column's error, and ``sum_j H_jj ||eta_j||^2`` when none does -- and
``H_jj = D_jj + sum_{k<j} L[j,k]^2 D_kk``.  ``block_ldl(H, b)`` zeroes the
strictly-lower part of each diagonal block, so a column sees earlier columns
from earlier blocks only.  The loss therefore interpolates exactly:

    Loss(b)  ~  tr(D) + S(b),
    S(b) := sum over diagonal blocks of sum_{j>k inside the block} L[j,k]^2 D_kk

taking ||eta_j||^2 as column-independent.  Everything here is computable from
the Hessian the encoder already holds; nothing is fit.
"""
import argparse, json, math, sys
import torch

sys.path.insert(0, "src")
from tessera.compensate import regularize_hessian


def S_of_b(L: torch.Tensor, d: torch.Tensor, b: int) -> float:
    """sum over diagonal blocks of the strictly-lower mass, D-weighted."""
    n = L.shape[0]
    assert n % b == 0, (n, b)
    tot = 0.0
    for s in range(0, n, b):
        blk = L[s:s + b, s:s + b]
        tri = torch.tril(blk, diagonal=-1)
        tot += float(((tri ** 2) * d[s:s + b].unsqueeze(0)).sum())
    return tot


def factor(H: torch.Tensor):
    """Unit-lower L and the LDL diagonal D from a Cholesky."""
    C = torch.linalg.cholesky(H)
    d = torch.diagonal(C).clone()
    L = C / d.unsqueeze(0)          # column-scale -> unit diagonal
    return L, d ** 2


def report(tag, H, blocks, sigma):
    Hreg = regularize_hessian(H, sigma_reg=sigma)
    L, D = factor(Hreg)
    trD = float(D.sum())
    out = {}
    for b in blocks:
        s = S_of_b(L, D, b)
        out[b] = (trD + s) / trD
    print(f"{tag}: n={H.shape[0]} tr(D)={trD:.6g}")
    for b in blocks:
        print(f"   b={b:4d}  predicted loss / full-LDLQ loss = {out[b]:.6f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    dev = a.device
    res = {}

    # --- dense Qwen: layers 0-1 attention.  q/k/v share one H per layer,
    # asserted by the sweep that produced the measurement being predicted.
    HFULL = "/mnt/shared/tessera-runs/ldlq/h_full_qwen06b.pt"
    Hs = torch.load(HFULL, map_location="cpu")["H"]
    for L in (0, 1):
        name = f"model.layers.{L}.self_attn.q_proj"
        H = Hs[name].to(dev, torch.float32)
        res[f"qwen.L{L}.attn"] = report(f"qwen L{L} attn (q=k=v)", H,
                                        [4, 8, 32], sigma=1.0)
        del H
        torch.cuda.empty_cache()
    del Hs

    # --- GLM experts: H rebuilt exactly as the wire harness builds it,
    # from the same fit rows (all but the trailing 1024 eval rows).
    ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
    for layer in (5, 20):
        blob = torch.load(
            f"{ACT}/model__language_model__layers__{layer}__mlp__experts.pt",
            map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        x_fit = xa[:xa.shape[0] - 1024].contiguous().to(dev)
        H = (x_fit.double().T @ x_fit.double()).float() / x_fit.shape[0]
        res[f"glm.L{layer}.experts"] = report(f"glm L{layer} experts", H,
                                              [16, 32, 128, 256], sigma=1.0)
        del blob, xa, x_fit, H
        torch.cuda.empty_cache()

    json.dump(res, open(a.out, "w"), indent=1)
    print("\n--- predicted ratios against the measured pairs ---")
    for k in res:
        r = res[k]
        if k.startswith("qwen"):
            print(f"{k}: b8/b32 predicted {r[8]/r[32]:.4f}  b4/b32 {r[4]/r[32]:.4f}")
        else:
            print(f"{k}: b16/b32 predicted {r[16]/r[32]:.4f}  "
                  f"b256/b32 {r[256]/r[32]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
