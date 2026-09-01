"""A reference EXL3 quantisation of the six GLM-5.3 expert tensors.

Runs the upstream exllamav3 quantiser (``quantize_exl3``) at K = 4, 5, 6, 8 on
the same six ``[2048, 4096]`` expert weights and the same Hessian Tessera is
measured on, so the two can be compared format-vs-format at matched rate.

Everything about the EXL3 side is the library's own: its trellis codebook
(``mul1``), its blockwise Hadamard pre-rotation and random sign flips, its
``sigma_reg = 0.025`` diagonal damping, its output-channel scaling, its global
scale search, and its LDLQ.  The only things this script supplies are the
weight, the Hessian, K, and the seed.

The Hessian is built from the FIRST ``tokens - 1024`` rows of the activation
capture; the last 1024 rows are the held-out evaluation set every Tessera
measurement uses and must never enter H.

Weight layout: exllamav3 wants ``(in_features, out_features)`` (it reads
``nn.Linear.weight.data.T``), so the checkpoint's ``[out, in]`` tensor is
transposed on the way in and the reconstruction is transposed back.  With
``return_weight_q = True`` the library already undoes its own rotation and
scales, so the saved tensor is in the original weight basis.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open

SRC = "/mnt/shared/models/GLM-5.3-Flash-BF16"
ACT = "/mnt/shared/dq-runs/glm53-bf16-pread-capture-1469b9b-20260901/act"
# The 256-row probe the 0.05653 prior was measured on (H from its first 128 rows,
# scored on its other 128).  Only used by --replicate-prior.
ACT_PROBE = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/act"
EXL3_SRC = "/home/rob/dq-runs/exl3-ref/src"
OUT = "/home/rob/dq-runs/exl3-ref"
# The K=4 weight-leg error a previous EXL3 artifact gave on these six tensors
# (experiments/tessera_fp4_native_levers.py).  Reported against, never tuned to.
EXL3_K4_PRIOR = 0.05653


def library_commit() -> str:
    import subprocess
    return subprocess.run(["git", "-C", EXL3_SRC, "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def build_H(x_fit: torch.Tensor):
    """H as the library accumulates it: the unnormalised sum X^T X, plus the row
    count.  ``finalize_capture_H`` divides by the count itself, then damps and
    rotates in place.  Accumulated in fp64 and cast to the fp32 the library's
    own capture buffer uses."""
    n = x_fit.shape[0]
    H = (x_fit.double().T @ x_fit.double()).float()
    return H, n


def make_H_data(H: torch.Tensor, count: int, key: str, device):
    # Exactly the dict Linear.init_H_data builds, after capture_H has run.
    # A fresh copy per (tensor, K): finalize_capture_H mutates H in place and
    # caches L/su/diag on the dict.
    return {
        "H": H.clone().to(device),
        "first_key": key,
        "count": count,
        "finalized": False,
        "num_total": 0,
        "inf_nan": torch.zeros(2, dtype=torch.long, device=device),
        "device": device,
    }


# The two configurations that get run.  "default" is convert_model.make_quant_args
# with the CLI defaults (--out_scales always, --codebook mul1); sigma_reg is
# deliberately absent from both so the library's own .get(..., 0.025) applies.
# "prior" reproduces the settings experiments/exl3_rate_sweep.py used for the
# 0.05653 number this run is checked against (mcg codebook, out-scales auto).
CONFIGS = {
    "default": {"apply_out_scales": True, "mul1": True},
    "prior":   {"apply_out_scales": None, "mcg": True},
}


def make_quant_args(K: int, seed: int, debug_dir: str, config: str):
    qa = {
        "seed": seed,
        "K": K,
        "devices": [0],
        "device_ratios": None,
        "debug_dir": debug_dir,
    }
    qa.update(CONFIGS[config])
    return qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 20, 42])
    ap.add_argument("--projs", nargs="+", default=["gate_proj", "up_proj"])
    ap.add_argument("--rates", type=int, nargs="+", default=[4, 5, 6, 8])
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-check", type=int, default=1,
                    help="second seed, run at K=4 on the first tensor only, as a "
                         "variance bar; negative to skip the check jobs")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--act", default=ACT, help="activation capture directory")
    ap.add_argument("--split", choices=["holdout", "half"], default="holdout",
                    help="holdout: H from all but the last --eval-rows rows. "
                         "half: H from the first half, score on the second (what "
                         "experiments/exl3_rate_sweep.py did for the 0.05653 prior)")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="quant_args configurations to run at every rate "
                         f"(default: the shipped 'default' arm plus a 'prior' arm at K=4)")
    ap.add_argument("--no-ship", action="store_true",
                    help="measure only; write no .pt files and no summary.json")
    ap.add_argument("--replicate-prior", action="store_true",
                    help="shorthand: --act ACT_PROBE --split half --configs prior "
                         "--rates 4 --no-ship, i.e. reproduce exl3_rate_sweep.py exactly")
    ap.add_argument("--fit-rows", type=int, default=None,
                    help="use only the FIRST N of the fit rows to build H (the "
                         "held-out eval split is unchanged). Isolates 'how many "
                         "calibration rows' from 'which eval set'")
    ap.add_argument("--merge-into", default=None,
                    help="path of an existing summary.json; this run's entries are "
                         "merged into it under --merge-key instead of being shipped")
    ap.add_argument("--merge-key", default="prior_replication")
    ap.add_argument("--gaussian-rates", type=int, nargs="*", default=[4, 5],
                    help="rates for the i.i.d. Gaussian sanity arm (empty to skip)")
    ap.add_argument("--verbose", action="store_true",
                    help="library verbose output, including its own quant nmse (a basis check)")
    a = ap.parse_args()
    if a.replicate_prior:
        a.act, a.split, a.configs, a.rates, a.no_ship = ACT_PROBE, "half", ["prior"], [4], True
        a.seed_check = None

    sys.path.insert(0, EXL3_SRC)
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3  # noqa: E402

    dev = torch.device("cuda:0")
    os.makedirs(a.out, exist_ok=True)
    debug_dir = os.path.join(a.out, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    commit = library_commit()
    print(f"exllamav3 commit {commit}")
    print(f"torch {torch.__version__} cuda {torch.version.cuda} "
          f"cap {torch.cuda.get_device_capability()}")

    index = json.load(open(f"{SRC}/model.safetensors.index.json"))["weight_map"]
    results = []

    for layer in a.layers:
        blob = torch.load(f"{a.act}/model__language_model__layers__{layer}__mlp__experts.pt",
                          map_location="cpu", weights_only=False)
        xa = blob["inputs"].float()
        n_fit = xa.shape[0] // 2 if a.split == "half" else xa.shape[0] - a.eval_rows
        n_eval_from = n_fit
        if a.fit_rows is not None:
            n_fit = min(n_fit, a.fit_rows)
        x_fit = xa[:n_fit].contiguous().to(dev)
        x_ev = xa[n_eval_from:].contiguous().to(dev)
        H0, count = build_H(x_fit)
        del x_fit
        torch.cuda.empty_cache()
        print(f"\n== L{layer}  H from {count} rows, eval {x_ev.shape[0]} rows")

        for proj in a.projs:
            name = f"model.language_model.layers.{layer}.mlp.experts.0.{proj}.weight"
            with safe_open(f"{SRC}/{index[name]}", framework="pt") as f:
                w = f.get_tensor(name).contiguous().to(dev).float()   # [out, in]
            R, C = w.shape
            nw = w.norm()
            y_ref = x_ev @ w.T
            ny = y_ref.norm()

            # (K, seed, config, shipped).  Only the shipped rows write a .pt and
            # enter the headline table.
            if a.configs is not None:
                jobs = [(K, a.seed, c, not a.no_ship)
                        for c in a.configs for K in a.rates]
            else:
                jobs = [(K, a.seed, "default", True) for K in a.rates]
                # The prior-settings arm at K=4 on every tensor: the reconciliation
                # against experiments/results/exl3_rate_sweep_K4.json.
                jobs.append((4, a.seed, "prior", False))
            if a.seed_check is not None and a.seed_check >= 0 \
                    and layer == a.layers[0] and proj == a.projs[0]:
                # Ran LAST, identical to the first job: exl3_rate_sweep.py records
                # that quantize_exl3 is stateful across K in one process (its own
                # arms went to rel_err ~55-65 after the first). If this reproduces
                # the first K=4 row exactly, this loop is not carrying that state.
                jobs.append((a.rates[0], a.seed, "default", False))
                jobs.append((4, a.seed_check, "default", False))

            for K, seed, config, shipped in jobs:
                H_data = make_H_data(H0, count, name, dev)
                quant_args = make_quant_args(K, seed, debug_dir, config)
                w_in_out = w.T.contiguous().clone()   # [in, out], mutated in place
                torch.cuda.synchronize()
                t0 = time.time()
                weight_q, proxy_err, out_tensors = quantize_exl3(
                    w_in_out, H_data, quant_args,
                    return_weight_q=True, progress_str=None, verbose=a.verbose,
                )
                torch.cuda.synchronize()
                dt = time.time() - t0

                w_hat = weight_q.T.contiguous().float()   # [out, in], original basis
                rel = float((w_hat - w).norm() / nw)
                out_leg = float(((x_ev @ w_hat.T) - y_ref).norm() / ny)

                bits = 0
                for t in out_tensors.values():
                    bits += t.numel() * t.element_size() * 8
                bpw = bits / (R * C)
                trellis_bits = (out_tensors["trellis"].numel()
                                * out_tensors["trellis"].element_size() * 8) / (R * C)

                tag = f"L{layer}.{proj}"
                rec = {
                    "tensor": tag, "layer": layer, "proj": proj,
                    "shape": [R, C], "K": K, "seed": seed,
                    "config": config, "shipped": shipped,
                    "rel_weight_err": rel,
                    "nmse": rel * rel,
                    "out_leg_rel_err": out_leg,
                    "proxy_err": float(proxy_err),
                    "bpw_bytes_exact": bpw,
                    "bpw_trellis_only": trellis_bits,
                    "bpw_library_K": float(K),
                    "apply_out_scales": bool(quant_args["apply_out_scales"]),
                    "g_scale": float(quant_args["g_scale"]),
                    "q_fallback": bool(quant_args["q_fallback"]),
                    "sigma_reg": 0.025,
                    "codebook": "mul1" if quant_args.get("mul1") else "mcg",
                    "wall_s": dt,
                    "h_fit_rows": count,
                    "commit": commit,
                }
                if shipped:
                    path = f"{a.out}/L{layer}_{proj}_K{K}.pt"
                    torch.save(w_hat.cpu(), path)
                    rec["path"] = path
                results.append(rec)
                print(f"   K={K} seed={seed} {config:<8}{'*' if shipped else ' '} "
                      f"rel={rel:.5f}  out={out_leg:.5f}  "
                      f"proxy={float(proxy_err):.6f}  bpw={bpw:.4f}  "
                      f"g_scale={quant_args['g_scale']:.5f}  {dt:.1f}s")
                del weight_q, w_hat, H_data, out_tensors
                torch.cuda.empty_cache()

            del w, y_ref
            torch.cuda.empty_cache()
        del H0, x_ev
        torch.cuda.empty_cache()

    # ---- i.i.d. Gaussian sanity arm -------------------------------------
    # A [2048, 4096] standard normal has no structure for a Hessian to exploit,
    # so the achievable relative RMS error at R bits/weight is the Gaussian
    # rate-distortion bound sqrt(D(R)) = 2^-R: 0.0625 at K=4.  Any real coder
    # must land at or above it.  Two H conventions, because they take different
    # code paths through the library:
    #   identity     H = I -> block_ldl gives L = I, whose diagonal is then
    #                zeroed, so LDLQ applies no compensation.  Same ldlq path as
    #                the real tensors, no feedback.
    #   no_hessian   H on the meta device -> the library's own uncalibrated
    #                fallback (finalize_capture_H q_fallback -> fallback_quant).
    # Error is measured in the ORIGINAL basis, after the library undoes its own
    # Hadamard rotation and su/sv scaling, against the same W it was handed.
    gaussian = []
    if a.gaussian_rates:
        gk, gn = 4096, 2048
        g = torch.Generator(device="cpu").manual_seed(0)
        wg = torch.randn(gn, gk, generator=g, dtype=torch.float32).to(dev)   # [out, in]
        nwg = wg.norm()
        print(f"\n== i.i.d. Gaussian [{gn}, {gk}] sanity arm  (2^-K bound: "
              + ", ".join(f"K={K}:{2.0 ** -K:.5f}" for K in a.gaussian_rates) + ")")
        for h_mode in ("identity", "no_hessian"):
            for K in a.gaussian_rates:
                if h_mode == "identity":
                    H_data = make_H_data(torch.eye(gk, device=dev), 1,
                                         "gaussian", dev)
                else:
                    H_data = {
                        "H": torch.empty((gk, gk), device="meta"),
                        "first_key": "gaussian", "count": 0, "finalized": False,
                        "num_total": 0,
                        "inf_nan": torch.zeros(2, dtype=torch.long, device=dev),
                        "device": dev,
                    }
                quant_args = make_quant_args(K, a.seed, debug_dir, "default")
                w_in_out = wg.T.contiguous().clone()
                torch.cuda.synchronize()
                t0 = time.time()
                weight_q, proxy_err, out_tensors = quantize_exl3(
                    w_in_out, H_data, quant_args,
                    return_weight_q=True, progress_str=None, verbose=a.verbose,
                )
                torch.cuda.synchronize()
                dt = time.time() - t0
                w_hat = weight_q.T.contiguous().float()
                rel = float((w_hat - wg).norm() / nwg)
                bits = sum(t.numel() * t.element_size() * 8 for t in out_tensors.values())
                bpw = bits / (gn * gk)
                sign_bits = ((out_tensors["suh"].numel() + out_tensors["svh"].numel())
                             * 2 * 8) / (gn * gk)
                gaussian.append({
                    "h_mode": h_mode, "K": K, "shape": [gn, gk], "seed": a.seed,
                    "config": "default", "codebook": "mul1",
                    "rel_rms_err": rel,
                    "gaussian_rd_bound": 2.0 ** -K,
                    "ratio_to_bound": rel / (2.0 ** -K),
                    "bpw_bytes_exact": bpw,
                    "bpw_trellis_only": (out_tensors["trellis"].numel()
                                         * out_tensors["trellis"].element_size()
                                         * 8) / (gn * gk),
                    "bpw_suh_svh_overhead": sign_bits,
                    "bpw_library_K": float(K),
                    "g_scale": float(quant_args["g_scale"]),
                    "q_fallback": bool(quant_args["q_fallback"]),
                    "apply_out_scales": bool(quant_args["apply_out_scales"]),
                    "proxy_err": float(proxy_err),
                    "wall_s": dt,
                })
                print(f"   {h_mode:<11} K={K}  rel_rms={rel:.5f}  "
                      f"bound=2^-{K}={2.0 ** -K:.5f}  ratio={rel / 2.0 ** -K:.4f}  "
                      f"bpw={bpw:.4f} (trellis {K:.1f} + suh/svh {sign_bits:.4f})  "
                      f"fallback={bool(quant_args['q_fallback'])}  {dt:.1f}s")
                del weight_q, w_hat, out_tensors, H_data
                torch.cuda.empty_cache()
        del wg
        torch.cuda.empty_cache()

    main_rows = [r for r in results if r["shipped"]]
    summary = {
        "commit": commit,
        "torch": torch.__version__,
        "source_model": SRC,
        "act_capture": a.act,
        "split": a.split,
        "fit_rows_cap": a.fit_rows,
        "eval_rows_held_out": a.eval_rows,
        "exl3_k4_prior": EXL3_K4_PRIOR,
        "quant_args_template": {
            "seed": a.seed, "devices": [0], "device_ratios": None,
            "apply_out_scales": True, "mul1": True,
            "sigma_reg": "library default 0.025 (key absent)",
        },
        "entries": results,
        "gaussian_sanity": gaussian,
        "args": vars(a),
    }
    by_k, by_k_out = {}, {}
    for r in main_rows:
        by_k.setdefault(r["K"], []).append(r["rel_weight_err"])
        by_k_out.setdefault(r["K"], []).append(r["out_leg_rel_err"])
    summary["mean_rel_err_by_K"] = {str(k): sum(v) / len(v) for k, v in sorted(by_k.items())}
    summary["mean_out_leg_by_K"] = {str(k): sum(v) / len(v) for k, v in sorted(by_k_out.items())}
    prior = [r["out_leg_rel_err"] for r in results if r["config"] == "prior"]
    prior_w = [r["rel_weight_err"] for r in results if r["config"] == "prior"]
    if prior:
        summary["prior_config_K4_mean_out_leg"] = sum(prior) / len(prior)
        summary["prior_config_K4_mean_rel_weight_err"] = sum(prior_w) / len(prior_w)
    if a.merge_into:
        with open(a.merge_into) as f:
            base = json.load(f)
        base[a.merge_key] = {
            "why": "the exact configuration and data experiments/exl3_rate_sweep.py "
                   "used for the 0.05653 prior, re-run against this build: probe "
                   "capture, half/half split, mcg codebook, out-scales auto",
            "act_capture": a.act, "split": a.split, "configs": a.configs,
            "recorded_prior_mean_out_leg": EXL3_K4_PRIOR,
            "replicated_mean_out_leg": (sum(r["out_leg_rel_err"] for r in results)
                                        / len(results)),
            "replicated_mean_rel_weight_err": (sum(r["rel_weight_err"] for r in results)
                                               / len(results)),
            "entries": results,
        }
        with open(a.merge_into, "w") as f:
            json.dump(base, f, indent=2)
        print(f"\nmerged {len(results)} entries into {a.merge_into} "
              f"under '{a.merge_key}'")
    elif not a.no_ship:
        with open(f"{a.out}/summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    tags = [f"L{l}.{p}" for l in a.layers for p in a.projs]
    for label, key, agg in (("relative WEIGHT error  ||W_hat-W||_F / ||W||_F", "rel_weight_err", by_k),
                            ("output-space WEIGHT LEG  ||X(W_hat-W)^T|| / ||XW^T||", "out_leg_rel_err", by_k_out)):
        print(f"\n{label}")
        print(f"{'tensor':<16}" + "".join(f"{'K=' + str(k):>10}" for k in a.rates) + f"{'bpw K4':>9}")
        for tag in tags:
            row = f"{tag:<16}"
            for K in a.rates:
                m = [r for r in main_rows if r["tensor"] == tag and r["K"] == K]
                row += f"{m[0][key]:>10.5f}" if m else f"{'-':>10}"
            m4 = [r for r in main_rows if r["tensor"] == tag and r["K"] == 4]
            row += f"{m4[0]['bpw_bytes_exact']:>9.4f}" if m4 else f"{'-':>9}"
            print(row)
        row = f"{'MEAN':<16}"
        for K in a.rates:
            v = agg.get(K)
            row += f"{sum(v) / len(v):>10.5f}" if v else f"{'-':>10}"
        print(row)

    print(f"\nThe 0.05653 prior (experiments/results/exl3_rate_sweep_K4.json) is an")
    print(f"OUTPUT-SPACE weight leg, on a different capture and a half/half split,")
    print(f"with the mcg codebook and out-scales auto. Compare it to:")
    print(f"  this run, library defaults, K=4 out leg : "
          f"{summary['mean_out_leg_by_K'].get('4', float('nan')):.5f}")
    if prior:
        print(f"  this run, prior settings,  K=4 out leg : "
              f"{summary['prior_config_K4_mean_out_leg']:.5f}")
    print(f"  prior artifact                          : {EXL3_K4_PRIOR:.5f}")
    print(f"\nchecks (not shipped):")
    for r in results:
        if not r["shipped"]:
            print(f"  {r['tensor']:<16} K={r['K']} seed={r['seed']} {r['config']:<8} "
                  f"rel={r['rel_weight_err']:.6f}  out={r['out_leg_rel_err']:.6f}")
    if a.merge_into:
        pass
    elif a.no_ship:
        print(f"\nmeasure-only run ({a.act}, split={a.split}); nothing written")
    else:
        print(f"\nwrote {a.out}/summary.json ({len(main_rows)} shipped entries)")


if __name__ == "__main__":
    main()
