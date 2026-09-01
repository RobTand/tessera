"""Allocate the GLM-5.3-Flash body over the real menu, under Mia's byte budget.

Two cost sources, kept separate on purpose:

* **W+A joint, measured** (`glm53_full_body_cost.py`) -- every arm scored as it
  serves: `Q_w(W) @ Q_a(x)` on held-out cached activations, so NVFP4's W4A4 and
  FP8's W8A8 are paid for rather than assumed free.  Sampled: 8 of 288 experts
  per layer, paired across arms.
* **AQUA A-side, analytic** (`aqua_activation_cost`) -- every unit, every
  expert, no sampling and no render, against the `compressed_tensors` lane's
  *attested* `served_activation_quantization.executes`.  This is the production
  mechanism and it covers 100% of `aqua_priceable_params`.

They answer the same question two ways, so the run reports both and the DP is
solved on each.  Where they disagree the disagreement is the finding.

**Per-expert Fisher.** The W-side run stored the raw squared error per sampled
expert, so the Fisher weighting is applied HERE, from
`h_trace_per_expert` -- not the layer-pooled `h_trace` the measurement loop
used. Pooling would assume error and sensitivity are uncorrelated across the
288 experts of a layer, which is exactly what an allocator is looking for.

**The budget is the accountant's.** `solve_allocation` is documented as a
projection whose achieved bits can overshoot, so its bpp is not the claim: the
chosen assignment is re-priced from `FormatSpec.effective_bits_for_shape` and
the Tessera family accountant, and that is what is compared to Mia.
"""
import argparse
import collections
import json
import pickle
import sys

W = "/mnt/shared/dq-runs/glm53-tessera-alloc-20260901/artifacts"
PROBE = "/mnt/shared/dq-runs/glm53-bf16-pread-probe-1469b9b-20260830/artifacts/probe.pkl"
COST_A = "/home/rob/tessera/experiments/results/glm53_full_body_cost_A.json"
PLAN = f"{W}/glm53_tessera_plan.json"
MODEL = "/mnt/shared/models/GLM-5.3-Flash-BF16"
# Mia, measured: body excluding vision and MTP, minus embed_tokens (vLLM's
# VocabParallelEmbedding has no quantized path) -- docs/measurements/
# glm53-body-budget-2026-09-01.md
MIA_BODY_GIB = 157.601


def unit_bits(fmt, shape):
    import prismaquant.format_registry as fr
    if fmt == "BF16":
        return 16.0
    return float(fr.get_format(fmt).effective_bits_for_shape(shape))


#: Every parameter the cost run could not price is charged here.  4.0 bpp is
#: the top serialisable Tessera rung, which is what the uniform export writes,
#: so the reserve is the size those units actually take if they ship as
#: Tessera -- not an optimistic floor.
UNCOVERED_FORMAT = "TESSERA_E2M1_K2_R896"


def uncovered_params(priced):
    """Parameters in the export plan that no priced unit covers.

    Leaf tensors are mapped to the serving unit the cost run named -- fused
    siblings collapse to ``gate_up_proj``, and every routed expert's leaf
    collapses to its layer's packed unit -- because that is the granularity the
    DP decides at.
    """
    import re
    from safetensors import safe_open

    plan = json.load(open(PLAN))
    plan = plan["plan"] if isinstance(plan, dict) and "plan" in plan else plan
    index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

    def unit_of(name):
        name = name.replace(".weight", "")
        match = re.match(r"(.*\.mlp\.experts)\.\d+\.(gate_proj|up_proj|down_proj)$",
                         name)
        if match:
            leaf = ".down_proj" if match.group(2) == "down_proj" else ".gate_up_proj"
            return match.group(1) + leaf
        for leaf in (".gate_proj", ".up_proj"):
            if name.endswith(leaf):
                return name[: -len(leaf)] + ".gate_up_proj"
        return name

    by_file = collections.defaultdict(list)
    for name in plan:
        by_file[index[name]].append(name)
    total = 0
    for shard, names in by_file.items():
        with safe_open(f"{MODEL}/{shard}", framework="pt") as handle:
            for name in names:
                if unit_of(name) in priced:
                    continue
                count = 1
                for dim in handle.get_slice(name).get_shape():
                    count *= dim
                total += count
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="",
                    help="comma list of bpp targets; default sweeps the frontier")
    args = ap.parse_args()

    from prismaquant.allocator_solver import Candidate, solve_allocation

    cost = json.load(open(COST_A))
    stats = pickle.load(open(PROBE, "rb"))["stats"]
    menu = [(m[0], m[1]) for m in cost["menu"]]
    aqua = {}
    for name in ("aqua_activation_dloss.json", "aqua_packed_activation_dloss.json"):
        try:
            aqua.update(json.load(open(f"{W}/{name}"))["table"])
        except FileNotFoundError:
            print(f"note: {name} absent -- A-side arm will be partial", flush=True)

    def sampled_experts(name, n_experts, k):
        """Reproduce the measurement's expert choice.

        `glm53_full_body_cost.py` seeds `20260901 + layer` and takes the first
        `k` of a `randperm`, so the identity of the sampled experts is
        recoverable rather than stored -- which matters, because the Fisher
        weight below is per expert and pairing it with the wrong one would be a
        silent mis-weighting rather than an error.
        """
        import torch
        layer = int(name.split(".layers.")[1].split(".")[0])
        gen = torch.Generator().manual_seed(20260901 + layer)
        return torch.randperm(n_experts, generator=gen)[:k].tolist()

    def probe_row(name):
        """The probe's stats for a cost unit, fusing siblings where it must.

        The cost run scores `gate_up_proj` as one unit because that is the
        serving unit -- fused siblings must share a format -- but the probe
        recorded `gate_proj` and `up_proj` separately for the *shared* experts
        (the packed routed experts are already fused there).  `h_trace` is a
        sum over output rows of `E_t[||g||^2 ||x||^2]`, and fused siblings see
        the same input, so the fused row is the sum of its siblings' and the
        out_features add.  Guessing one sibling's h_trace for the pair would
        halve the Fisher weight of every shared expert in the model.
        """
        if name in stats:
            return stats[name]
        if not name.endswith(".gate_up_proj"):
            raise KeyError(name)
        stem = name[: -len("gate_up_proj")]
        parts = [stats[stem + leaf] for leaf in ("gate_proj", "up_proj")]
        row = dict(parts[0])
        row["h_trace"] = sum(float(p["h_trace"]) for p in parts)
        row["out_features"] = sum(int(p["out_features"]) for p in parts)
        return row

    units, candidates, coverage = {}, {}, collections.Counter()
    for name, entry in cost["units"].items():
        n_params = entry["n_params"]
        scale = entry["scale"]
        samples = entry["samples"]
        row = probe_row(name)
        per_expert_h = row.get("h_trace_per_expert")
        picked = (sampled_experts(name, int(row["num_experts"]), len(samples))
                  if per_expert_h is not None and ".layers." in name else None)
        rows = {}
        for fmt, _bits in menu:
            joint = 0.0
            for index, sample in enumerate(samples):
                sq = sample[fmt][1]
                # Fisher weight: per-expert where the probe has it, else the
                # unit's own h_trace (dense units are one "sample").
                if per_expert_h and picked and index < len(picked):
                    h = float(per_expert_h[picked[index]])
                else:
                    h = float(row["h_trace"])
                joint += 0.5 * h * sq
            rows[fmt] = joint / max(1, len(samples)) * scale
        units[name] = {"n_params": n_params}
        shape = (row["out_features"], row["in_features"])
        candidates[name] = [
            Candidate(fmt=fmt, bits_per_param=unit_bits(fmt, shape),
                      memory_bytes=int(n_params * unit_bits(fmt, shape) / 8),
                      predicted_dloss=rows[fmt])
            for fmt, _bits in menu
        ]
        coverage["units"] += 1
        coverage["params"] += n_params
        if name in aqua:
            coverage["aqua_priced"] += 1

    # --- what the cost run does NOT cover ---------------------------------
    #
    # The DP can only place units it has a cost for, and the cost run priced
    # the MoE experts, the shared experts, the dense MLPs and lm_head -- 95.8%
    # of the export plan's parameters.  The rest (attention q/k/v/o and the
    # expert units outside the measured layers) has no probe activation capture
    # at all, so it has no honest cost and cannot be allocated.  Charging it at
    # zero would make the frontier's GiB a number for a different artifact than
    # the one that ships, which is exactly the accounting that gets a size
    # claim retracted.  So it is charged at a DECLARED default, subtracted from
    # the budget before the DP runs, and reported.
    uncovered = uncovered_params(cost["units"])
    # A square-ish shape: the Tessera accountant's bits depend on the shape
    # only through the scale/forest/header planes, which are a fraction of a
    # percent at these sizes, so one representative shape is honest here and
    # the per-unit exact price is what the exporter charges.
    reserve_bits = unit_bits(UNCOVERED_FORMAT, (4096, 4096))
    reserve_gib = uncovered * reserve_bits / 8 / 2 ** 30
    budget = MIA_BODY_GIB - reserve_gib
    print(f"units {coverage['units']}  params {coverage['params']:,}  "
          f"aqua-priced {coverage['aqua_priced']}")
    print(f"uncovered by the cost run: {uncovered/1e9:.2f} B params "
          f"({100*uncovered/(uncovered+coverage['params']):.1f}% of the plan), "
          f"charged at {UNCOVERED_FORMAT} = {reserve_gib:.3f} GiB")
    total = coverage["params"]
    floor = min(sum(min(c.bits_per_param for c in cs) * units[n]["n_params"]
                    for n, cs in candidates.items()) / total for _ in (0,))
    print(f"menu floor {floor:.4f} bpp   Mia body budget {MIA_BODY_GIB} GiB   "
          f"available to the DP {budget:.3f} GiB\n")

    targets = ([float(v) for v in args.targets.split(",")] if args.targets
               else [floor + 0.05 * k for k in range(0, 24)])
    print(f"{'target':>8}{'achieved':>10}{'GiB':>10}{'dloss':>14}   mix")
    frontier = []
    for target in targets:
        solved = solve_allocation(units, candidates, target)
        if solved is None:
            continue
        assignment, chosen = solved
        bits = sum(chosen[n].bits_per_param * units[n]["n_params"]
                   for n in assignment) / total
        gib = sum(chosen[n].memory_bytes for n in assignment) / 2 ** 30
        dloss = sum(chosen[n].predicted_dloss for n in assignment)
        mix = collections.Counter(assignment.values())
        frontier.append(dict(target=target, bits=bits, gib=gib, dloss=dloss,
                             mix=dict(mix), assignment=assignment))
        short = "  ".join(f"{k.replace('TESSERA_E2M1_','T').replace('_R768','').replace('_R896','')}:{v}"
                          for k, v in mix.most_common())
        print(f"{target:>8.3f}{bits:>10.4f}{gib:>10.3f}{dloss:>14.6g}   {short}")

    json.dump(frontier, open(f"{W}/glm53_frontier.json", "w"), indent=1)
    print(f"\nwrote {W}/glm53_frontier.json")
    fits = [f for f in frontier if f["gib"] <= budget]
    if fits:
        best = min(fits, key=lambda f: f["dloss"])
        whole = best["gib"] + reserve_gib
        print(f"\nBest under Mia's {MIA_BODY_GIB} GiB body budget: "
              f"{best['bits']:.4f} bpp over the priced units, "
              f"{best['gib']:.3f} GiB + {reserve_gib:.3f} GiB reserved "
              f"= {whole:.3f} GiB whole body "
              f"({100*(1-whole/MIA_BODY_GIB):+.2f}% vs Mia), "
              f"dloss {best['dloss']:.6g}")
        print(f"  mix: {best['mix']}")


if __name__ == "__main__":
    sys.exit(main())
