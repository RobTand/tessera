"""Measure each candidate's offline pick as one arm, and bootstrap it.

``score_candidates.py`` evaluates a knapsack pick by summing the measured
single-unit ``dkl`` rows.  On these seven units that sum reproduces only 84.8%
of the jointly measured effect (the receipt's two-bundle decomposition
reproduced 99.3%, so the residual is real and this experiment must not hide
behind it).  So every pick a candidate makes is also *built* and measured as a
single arm, and compared to the byte-matched uniform control by a paired
bootstrap over the 4088 scored positions -- the same instrument, the same
positions, one difference per position.

Geometry, weights, teacher and metric are ``rd_table.py``'s; only the set of
arms differs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rd_table import (  # noqa: E402
    ALLOC, CORPUS, MODEL, ROLES, TesseraFp8Linear, UNIFORM_RUNG,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks", required=True, help="JSON {name: {role: rung}}")
    ap.add_argument("--cache", default="/home/rob/tmp/ts-rung-rd-out/tiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=4000)
    ap.add_argument("--reference", default="uniform_R1006",
                    help="the byte-matched uniform arm every other arm is "
                         "bootstrapped against")
    args = ap.parse_args()
    device = torch.device("cuda")
    picks = json.loads(Path(args.picks).read_text())
    picks = {"uniform_R1006": {r: UNIFORM_RUNG for r in ROLES},
             "served_allocation": dict(ALLOC), **picks}
    if args.reference not in picks:
        raise SystemExit(f"reference arm {args.reference!r} is not in the pick set")

    corpus = json.loads(CORPUS.read_text())
    chunks = torch.tensor(corpus["chunks"], dtype=torch.long, device=device)
    n_chunks, seqlen = chunks.shape
    scored = n_chunks * (seqlen - 1)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()

    @torch.no_grad()
    def logits_for(chunk):
        return model(chunk.unsqueeze(0)).logits[0, : seqlen - 1].float()

    teacher_lp = torch.empty((n_chunks, seqlen - 1, model.config.vocab_size),
                             dtype=torch.float32, device=device)
    with torch.no_grad():
        for i in range(n_chunks):
            teacher_lp[i] = torch.log_softmax(logits_for(chunks[i]), dim=-1)
    teacher_p = teacher_lp.exp()

    cache = Path(args.cache)
    tiles, wire = {}, {}
    for name, pick in picks.items():
        for role, rung in pick.items():
            if (role, rung) in tiles:
                continue
            b = torch.load(cache / f"{role.replace('.', '__')}_R{rung}.pt", map_location=device)
            tiles[(role, rung)] = (b["tile"].view(torch.uint8).contiguous(),
                                   b["scale"].contiguous())
            wire[(role, rung)] = int(b["bytes"])

    layer0 = model.model.layers[0]
    holders = {}
    for role in ROLES:
        parent, leaf = role.split(".")
        p = getattr(layer0, parent)
        old = getattr(p, leaf)
        holders[role] = TesseraFp8Linear(old.bias, old.out_features).to(device)
        setattr(p, leaf, holders[role])

    @torch.no_grad()
    def arm(pick):
        for role in ROLES:
            holders[role].load(*tiles[(role, pick[role])])
        pos = torch.empty(scored, dtype=torch.float32, device=device)
        agree = 0
        for i in range(n_chunks):
            slp = torch.log_softmax(logits_for(chunks[i]), dim=-1)
            pos[i * (seqlen - 1):(i + 1) * (seqlen - 1)] = (
                teacher_p[i] * (teacher_lp[i] - slp)).sum(-1)
            agree += int((slp.argmax(-1) == teacher_lp[i].argmax(-1)).sum())
        return pos.cpu().numpy(), agree / scored

    positions, out = {}, {"schema": "tessera.rung_pick_verification/1", "arms": {}}
    for name, pick in picks.items():
        pos, ag = arm(pick)
        positions[name] = pos
        out["arms"][name] = {"pick": pick, "kl_full": float(pos.mean()),
                             "top1_agree": ag,
                             "wire_bytes": sum(wire[(r, pick[r])] for r in ROLES)}
        print(f"{name:26s} kl {pos.mean():.6f}  top1 {ag*100:.2f}%  "
              f"{out['arms'][name]['wire_bytes']}B", flush=True)

    rng = np.random.default_rng(0)
    idx = rng.integers(0, scored, size=(args.bootstrap, scored))
    ref = positions[args.reference]
    for name, pos in positions.items():
        d = pos - ref
        draws = d[idx].mean(axis=1)
        out["arms"][name].update(
            reference=args.reference,
            vs_uniform=float(pos.mean() / ref.mean()),
            dkl_vs_uniform=float(d.mean()),
            dkl_bootstrap_se=float(draws.std(ddof=1)),
            dkl_ci95=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
            p_worse_than_uniform=float((draws > 0).mean()))
    Path(args.out).write_text(json.dumps(out, indent=2))
    np.savez_compressed(Path(args.out).with_suffix(".positions.npz"), **positions)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
