#!/usr/bin/env python3
"""The tessera#514 excess, read against the BF16 reference instead of against TP1.

The world-size receipt (``experiments/results/glm53_a4_stub_tp_single_rank_kl_514.json``)
compares each checkpoint with ITSELF at another world size.  On a quantized
model that comparison cannot tell added error from re-drawn error: a small
perturbation entering a quantizer flips codes near decision boundaries, and
every flip re-draws that element's quantization error without changing its
distribution.  This script reads the same four ``kl_tool`` dumps the receipt
names (A4 TP1, A4 TP2, BF16 TP1, BF16 TP2), checks their sha256 against the
committed table, and measures the quantity the receipt could not: how far the
A4 stub is from the BF16 stub at each world size, and how much of the A4
TP2-vs-TP1 delta is a re-draw of the quantization noise versus error added on
top of it.

On the ids all four dumps returned at a position (the shared support), with
q1 = A4TP1 - BF16TP1, q2 = A4TP2 - BF16TP1, dA = A4TP2 - A4TP1 = q2 - q1 and
dB = BF16TP2 - BF16TP1:

* ``var_ratio`` = var(q2)/var(q1): 1 when TP2 adds nothing to the quantization
  error; ``additive_share`` = (var(q2)-var(q1))/var(dA) = 1 + 2 cov(q1,dA)/var(dA)
  is the share of the A4 delta that is added error rather than a re-draw (0
  for a pure re-draw, 1 for error independent of the quantization noise);
* ``reroll_fraction`` = 1 - corr(q1, q2), the share of the quantization-noise
  variance that TP2 re-drew;
* ``corr_dA_dB``: whether the A4 delta tracks the BF16 control's delta;
* the kurtosis of dA and dB: many small flips against a few chaotic positions.

Per position, the renormalized shared-support KL(BF16TP1 || A4) at TP1 and TP2
is paired, so the ratio of means carries a bootstrap interval over positions
(the pairs of one position are not independent).  The lumped top-1024 lower
bound is copied from ``kl_tool compare`` JSONs when given.

    python3 experiments/tp2_reference_kl_514.py \\
        --a4-tp1 A4/tp1.json.npz --a4-tp2 A4/tp2.json.npz \\
        --bf16-tp1 BF16/tp1.json.npz --bf16-tp2 BF16/tp2.json.npz \\
        --topk-compare bf16_tp1_vs_a4_tp1=kl-B1A1.json --topk-compare bf16_tp1_vs_a4_tp2=kl-B1A2.json \\
        --out experiments/results/glm53_a4_stub_tp2_reference_kl_514.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RECEIPT_TABLE = REPO / "experiments/results/glm53_a4_stub_tp_single_rank_kl_514.json"
PAD_ID = -1
SEED = 514
BOOTSTRAPS = 500


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return json.loads(str(z["meta"])), z["ids"], z["lps"].astype(np.float64)


def row(ids, lps):
    """One logprob per id, ids sorted ascending (the same dedupe tp-equivalence.py uses)."""
    keep = (ids != PAD_ID) & np.isfinite(lps)
    ids, lps = ids[keep].astype(np.int64), lps[keep]
    order = np.argsort(-lps, kind="stable")
    ids, lps = ids[order], lps[order]
    ids, first = np.unique(ids, return_index=True)
    return ids, lps[first]


def renorm_kl(p_ids, p_lps, q_ids, q_lps):
    common, ia, ib = np.intersect1d(p_ids, q_ids, assume_unique=True, return_indices=True)
    lp, lq = p_lps[ia], q_lps[ib]
    rp, rq = lp - np.log(np.exp(lp).sum()), lq - np.log(np.exp(lq).sum())
    return float(np.sum(np.exp(rp) * (rp - rq))), np.abs(lp - lq)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for arm in ("a4-tp1", "a4-tp2", "bf16-tp1", "bf16-tp2"):
        ap.add_argument(f"--{arm}", required=True, type=Path)
    ap.add_argument("--topk-compare", action="append", default=[],
                    help="name=kl_tool-compare.json, copied into the result as the top-K lower bound")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    table = json.loads(RECEIPT_TABLE.read_text(encoding="utf-8"))
    expect = {
        "a4_tp1": table["arms"]["a4_tp2_vs_tp1"]["reference"]["npz_sha256"],
        "a4_tp2": table["arms"]["a4_tp2_vs_tp1"]["compared"]["npz_sha256"],
        "bf16_tp1": table["arms"]["bf16_tp2_vs_tp1"]["reference"]["npz_sha256"],
        "bf16_tp2": table["arms"]["bf16_tp2_vs_tp1"]["compared"]["npz_sha256"],
    }
    paths = {"a4_tp1": args.a4_tp1, "a4_tp2": args.a4_tp2, "bf16_tp1": args.bf16_tp1, "bf16_tp2": args.bf16_tp2}
    digests = {k: sha256(p) for k, p in paths.items()}
    for k in expect:
        if digests[k] != expect[k]:
            raise SystemExit(f"refused: {k} payload sha256 {digests[k]} is not the receipt's {expect[k]}")

    arms = {}
    meta = {}
    for k, p in paths.items():
        m, ids, lps = load(p)
        meta[k] = m
        arms[k] = [row(ids[i], lps[i]) for i in range(ids.shape[0])]
    corpus = {(m.get("corpus") or {}).get("contract_sha256") for m in meta.values()}
    tokenizer = {(m.get("tokenizer") or {}).get("identity_sha256") for m in meta.values()}
    if len(corpus) != 1 or None in corpus or len(tokenizer) != 1 or None in tokenizer:
        raise SystemExit("refused: the four dumps do not share one corpus contract and tokenizer")
    n = {len(a) for a in arms.values()}
    if len(n) != 1:
        raise SystemExit(f"refused: positions differ {n}")
    n = n.pop()

    # Per position: renormalized KL(BF16TP1 || A4) at TP1 and at TP2, and the four-way deltas.
    kl_tp1, kl_tp2, kl_b2_tp1, kl_b2_tp2 = [], [], [], []
    abs_tp1, abs_tp2 = [], []
    per_pos = []
    for i in range(n):
        b1, b2, a1, a2 = arms["bf16_tp1"][i], arms["bf16_tp2"][i], arms["a4_tp1"][i], arms["a4_tp2"][i]
        k, d = renorm_kl(*b1, *a1); kl_tp1.append(k); abs_tp1.append(d)
        k, d = renorm_kl(*b1, *a2); kl_tp2.append(k); abs_tp2.append(d)
        kl_b2_tp1.append(renorm_kl(*b2, *a1)[0]); kl_b2_tp2.append(renorm_kl(*b2, *a2)[0])
        c = a1[0]
        for other in (a2, b1, b2):
            c = np.intersect1d(c, other[0], assume_unique=True)
        v = {name: arm[1][np.searchsorted(arm[0], c)] for name, arm in
             (("a1", a1), ("a2", a2), ("b1", b1), ("b2", b2))}
        per_pos.append((v["a1"] - v["b1"], v["a2"] - v["b1"], v["a2"] - v["a1"], v["b2"] - v["b1"]))
    kl_tp1, kl_tp2 = np.array(kl_tp1), np.array(kl_tp2)
    kl_b2_tp1, kl_b2_tp2 = np.array(kl_b2_tp1), np.array(kl_b2_tp2)

    def decomposition(idx):
        q1 = np.concatenate([per_pos[i][0] for i in idx]); q2 = np.concatenate([per_pos[i][1] for i in idx])
        dA = np.concatenate([per_pos[i][2] for i in idx]); dB = np.concatenate([per_pos[i][3] for i in idx])
        vq1, vq2, vdA, vdB = q1.var(), q2.var(), dA.var(), dB.var()
        cov = np.mean((q1 - q1.mean()) * (dA - dA.mean()))
        kurt = lambda x, v: float(np.mean((x - x.mean()) ** 4) / v ** 2)
        return {
            "pairs": int(q1.size),
            "var_q1": float(vq1), "var_q2": float(vq2), "var_dA": float(vdA), "var_dB": float(vdB),
            "var_ratio": float(vq2 / vq1),
            "additive_share": float(1.0 + 2.0 * cov / vdA),
            "reroll_fraction": float(1.0 - np.corrcoef(q1, q2)[0, 1]),
            "corr_dA_dB": float(np.corrcoef(dA, dB)[0, 1]),
            "var_dA_over_var_dB": float(vdA / vdB),
            "abs_dA_p50": float(np.median(np.abs(dA))), "abs_dB_p50": float(np.median(np.abs(dB))),
            "abs_dA_p99": float(np.quantile(np.abs(dA), 0.99)), "abs_dB_p99": float(np.quantile(np.abs(dB), 0.99)),
            "kurtosis_dA": kurt(dA, vdA), "kurtosis_dB": kurt(dB, vdB),
            "kl_ratio_tp2_over_tp1": float(kl_tp2[idx].mean() / kl_tp1[idx].mean()),
            "kl_ratio_tp2_over_tp1_vs_bf16_tp2": float(kl_b2_tp2[idx].mean() / kl_b2_tp1[idx].mean()),
        }

    full = decomposition(np.arange(n))
    rng = np.random.default_rng(SEED)
    boots = [decomposition(rng.integers(0, n, n)) for _ in range(BOOTSTRAPS)]
    with_ci = {}
    for key, value in full.items():
        if key == "pairs":
            with_ci[key] = value
            continue
        samples = np.array([b[key] for b in boots])
        with_ci[key] = {"value": value, "ci95": [float(np.quantile(samples, 0.025)),
                                                 float(np.quantile(samples, 0.975))]}
    diff = kl_tp2 - kl_tp1
    abs_tp1 = np.concatenate(abs_tp1); abs_tp2 = np.concatenate(abs_tp2)

    topk = {}
    for spec in args.topk_compare:
        name, _, path = spec.partition("=")
        c = json.load(open(path))["all"]
        topk[name] = {"kl_lower_mean": c["kl_lower_mean"], "kl_lower_p99": c["kl_lower_p99"],
                      "kl_lower_max": c["kl_lower_max"], "teacher_tail_mass_mean": c["teacher_tail_mass_mean"],
                      "source_sha256": sha256(Path(path))}

    result = {
        "schema": "tessera.tp2_reference_kl/1",
        "issue": "tessera#514",
        "resolves": "experiments/results/glm53_a4_stub_tp_single_rank_kl_514.json",
        "scope": "the four kl_tool dumps the world-size receipt names (A4 and BF16 4-layer GLM-5.3-Flash stubs, "
                 "TP1 and TP2, prefill 8x512, eager, top-1024), read against the BF16 TP1 stub as the reference",
        "positions": n,
        "corpus_contract_sha256": corpus.pop(),
        "tokenizer_identity_sha256": tokenizer.pop(),
        "payloads": {k: {"npz_sha256": digests[k], "host": meta[k].get("host")} for k in paths},
        "reference_kl": {
            "definition": "renormalized shared-support KL(BF16 TP1 || A4) per position, TP1 and TP2 paired",
            "tp1_mean": float(kl_tp1.mean()), "tp2_mean": float(kl_tp2.mean()),
            "ratio_tp2_over_tp1": round(float(kl_tp2.mean() / kl_tp1.mean()), 4),
            "ratio_ci95": with_ci["kl_ratio_tp2_over_tp1"]["ci95"],
            "paired_mean_diff": float(diff.mean()),
            "paired_mean_diff_sem": float(diff.std(ddof=1) / np.sqrt(n)),
            "tp2_worse_share_of_positions": float(np.mean(diff > 0)),
            "abs_dlogprob_p50_tp1": float(np.median(abs_tp1)), "abs_dlogprob_p50_tp2": float(np.median(abs_tp2)),
            "abs_dlogprob_p99_tp1": float(np.quantile(abs_tp1, 0.99)), "abs_dlogprob_p99_tp2": float(np.quantile(abs_tp2, 0.99)),
            "against_bf16_tp2": {"tp1_mean": float(kl_b2_tp1.mean()), "tp2_mean": float(kl_b2_tp2.mean()),
                                 "ratio_tp2_over_tp1": round(float(kl_b2_tp2.mean() / kl_b2_tp1.mean()), 4)},
            "topk_lower_bound": topk,
        },
        "decomposition": {
            "definition": "on the ids all four dumps returned: q1 = A4TP1-BF16TP1, q2 = A4TP2-BF16TP1, "
                          "dA = A4TP2-A4TP1, dB = BF16TP2-BF16TP1; bootstrap over positions",
            "bootstraps": BOOTSTRAPS, "seed": SEED,
            **with_ci,
        },
    }
    args.out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    r, d = result["reference_kl"], result["decomposition"]
    print(f"positions {n}; KL(BF16TP1||A4): TP1 {r['tp1_mean']:.5f}, TP2 {r['tp2_mean']:.5f}, "
          f"ratio {r['ratio_tp2_over_tp1']} CI95 {r['ratio_ci95']}, TP2 worse at {100*r['tp2_worse_share_of_positions']:.1f}% of positions")
    for key in ("var_ratio", "additive_share", "reroll_fraction", "corr_dA_dB", "var_dA_over_var_dB",
                "abs_dA_p50", "abs_dB_p50", "abs_dA_p99", "abs_dB_p99", "kurtosis_dA", "kurtosis_dB"):
        print(f"  {key:22s} {d[key]['value']:+.5f}  CI95 [{d[key]['ci95'][0]:+.5f}, {d[key]['ci95'][1]:+.5f}]")
    for name, t in topk.items():
        print(f"  top-1024 lower bound {name}: mean {t['kl_lower_mean']:.6f} p99 {t['kl_lower_p99']:.4f} max {t['kl_lower_max']:.4f}")
    print("->", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
