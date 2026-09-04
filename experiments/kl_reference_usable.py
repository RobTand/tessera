#!/usr/bin/env python3
"""Is this teacher dump a usable reference, or is the KL beside it noise?

A top-K KL against a BF16 teacher only means something when the teacher has an
opinion.  On 2026-09-01 a 4-layer structural cut of GLM-5.3-Flash served, and
produced a KL, and the KL was meaningless; on 2026-09-04 the same thing
happened again on a 16-expert cut of the same model, and the second time it
cost three serves to find out.  The check is cheap and the dump already has
everything it needs, so run it on the teacher BEFORE spending a student serve:

    python3 experiments/kl_reference_usable.py <teacher.json.npz> [corpus.json]

It reports, and refuses (exit 2) when the reference cannot carry a comparison:

  next-token top-1   how often the teacher's argmax IS the corpus's next token.
                     A trained model is tens of percent.  Both cuts scored
                     0.00% of 4088 -- not one.  This is the load-bearing number
                     and it needs the corpus contract, which is why the corpus
                     argument is worth passing.
  true-token rank    where the actual next token sits among the K returned.
                     Median 1024 of 1025 means the teacher ranks the truth
                     below essentially everything it was asked about.
  support mass       how much probability the compared support holds.  Below
                     this the DPI bound is too slack to read: the 09-04 run
                     bounded its KL between 0.0058 and 78.96.
  confident positions  positions where the teacher's top-1 clears `--confident`.
                     Zero of them means there is no subset to fall back to --
                     the 09-01 run still had 1709, the 09-04 cut had none.

The thresholds are deliberately loose.  This is a refusal for a reference that
cannot say anything at all, not a quality gate: a model that passes here has
only earned the right to be compared against.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def load_targets(corpus_path: str) -> tuple[np.ndarray, dict]:
    """The corpus's own next tokens, in the order the dump scores them."""
    contract = json.load(open(corpus_path))
    chunks = contract["chunks"]
    prepends_bos = contract.get("prepends_bos", False)
    if not isinstance(prepends_bos, bool):
        raise SystemExit(
            f"corpus prepends_bos must be a boolean, got {prepends_bos!r}"
        )
    # Without an injected BOS the first token in each chunk has no predecessor,
    # so prompt-logprob dumping begins at c[1].  With one, BOS is that
    # predecessor and the runtime returns a score for every corpus token.  This
    # is a property of the corpus contract, not something to infer from a dump's
    # shape: inference would let a malformed pair choose its own interpretation.
    targets = np.concatenate([
        np.asarray(c if prepends_bos else c[1:], dtype=np.int64) for c in chunks
    ])
    declared = contract.get("scored_positions")
    if declared is not None and declared != targets.shape[0]:
        raise SystemExit(
            f"corpus declares {declared} scored positions, but prepends_bos="
            f"{prepends_bos} and its chunks define {targets.shape[0]}"
        )
    return targets, contract


def require_same_contract(dump_meta: dict, contract: dict) -> None:
    """Refuse a corpus that is not the one the dump was taken on.

    A dump and a contract built for different tokenizers have the same shape
    and different meanings: passing the GLM corpus against a Qwen dump scored
    1.54% next-token top-1 on a model that is fine, which reads exactly like
    the refusal this tool exists to raise.  The identities are recorded in
    both files; compare them rather than the token counts.
    """
    d_sha = dump_meta.get("corpus", {}).get("contract_sha256")
    c_sha = contract.get("contract_sha256")
    if d_sha and c_sha and d_sha != c_sha:
        raise SystemExit(
            f"the dump was taken on corpus {d_sha[:12]} and you passed {c_sha[:12]}.\n"
            f"  dump tokenizer:   {dump_meta.get('tokenizer', {}).get('path')}\n"
            f"  corpus tokenizer: {contract.get('tokenizer', {}).get('path')}\n"
            "  These score the same positions and mean different things."
        )


def summarise(npz_path: str, targets: np.ndarray | None, confident: float,
              contract: dict | None = None) -> dict:
    z = np.load(npz_path)
    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    if contract is not None:
        require_same_contract(meta, contract)
    ids, lps = z["ids"], z["lps"]
    order = np.argsort(-lps, axis=1)
    sorted_ids = np.take_along_axis(ids, order, axis=1)
    sorted_lps = np.take_along_axis(lps, order, axis=1)
    top1_prob = np.exp(sorted_lps[:, 0])
    out = {
        "positions": int(ids.shape[0]),
        "returned_k": int(ids.shape[1]),
        "support_mass_mean": float(np.exp(lps).sum(axis=1).mean()),
        "top1_prob_median": float(np.median(top1_prob)),
        "confident_positions": int((top1_prob >= confident).sum()),
    }
    if targets is not None:
        if targets.shape[0] != ids.shape[0]:
            raise SystemExit(
                f"corpus gives {targets.shape[0]} scored positions, the dump has "
                f"{ids.shape[0]}: these are not the same contract"
            )
        present = sorted_ids == targets[:, None]
        has = present.any(axis=1)
        rank = np.where(has, present.argmax(axis=1), -1)
        out["next_token_top1"] = float((sorted_ids[:, 0] == targets).mean())
        out["true_token_in_support"] = float(has.mean())
        out["true_token_rank_median"] = int(np.median(rank[has])) if has.any() else -1
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dump", help="a kl_tool position dump, <name>.json.npz")
    ap.add_argument("corpus", nargs="?", help="the corpus contract the dump was taken on")
    ap.add_argument("--confident", type=float, default=0.5,
                    help="top-1 probability at which a position counts as confident")
    ap.add_argument("--min-support-mass", type=float, default=0.60,
                    help="refuse below this much probability inside the compared support")
    ap.add_argument("--min-next-token-top1", type=float, default=0.05,
                    help="refuse below this next-token top-1 accuracy (needs the corpus)")
    ap.add_argument("--json", help="write the summary here")
    args = ap.parse_args()

    targets, contract = load_targets(args.corpus) if args.corpus else (None, None)
    s = summarise(args.dump, targets, args.confident, contract)

    print(f"{args.dump}")
    print(f"  positions {s['positions']}  returned k {s['returned_k']}")
    print(f"  probability mass inside the compared support   {s['support_mass_mean']:.4f}")
    print(f"  median top-1 probability                       {s['top1_prob_median']:.4f}")
    print(f"  confident positions (top-1 >= {args.confident})            {s['confident_positions']}")
    if targets is not None:
        print(f"  next-token top-1 accuracy                      {100*s['next_token_top1']:.2f}%")
        print(f"  true token present in support                  {100*s['true_token_in_support']:.2f}%")
        print(f"  median rank of the true token                  {s['true_token_rank_median']}")

    refusals = []
    if s["support_mass_mean"] < args.min_support_mass:
        refusals.append(
            f"only {s['support_mass_mean']:.2%} of the teacher's mass is inside the compared "
            f"support, so the DPI bound is too slack to read"
        )
    if s["confident_positions"] == 0:
        refusals.append("no position is confident, so there is no subset to fall back to")
    if targets is not None and s["next_token_top1"] < args.min_next_token_top1:
        refusals.append(
            f"next-token top-1 is {s['next_token_top1']:.2%}: this reference does not predict "
            f"its own corpus, so a KL against it measures agreement with noise"
        )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({**s, "refusals": refusals}, fh, indent=1)
    if refusals:
        print("\nREFUSED as a KL reference:")
        for r in refusals:
            print(f"  - {r}")
        print("  A KL taken against this dump is a plumbing receipt, not a quality number.")
        return 2
    print("\nusable as a KL reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
