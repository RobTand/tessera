"""Real GLM-5.3 routed units in the register-direct fragment order (eng-regdirect-build, stage 1 harness).

Packs one layer's routed experts, cut to one TP2 rank (``real_units.layer_units``), into the planes
``fragment_synth.make_stack`` returns, so the stage-1 bench runs the kernel on the disk codes.  It
repacks each expert with ``tessera.fragment_wire.repack_fragment`` and only rearranges its
output into the kernel's planes: the history units (before tile 0) become ``hist``, the rest
``wire``; the sorted slot rates become the (ra, rb, ksa) profile.

CPU self-check (D38 for the real-wire path): ``reference_decode`` of the packed planes equals the
window-rule decode of the disk unit (``real_units.reference_weights``) for the checked experts.

  python real_stack.py --artifact DIR --layer 10 --rank 0 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fragment_synth import GROUPS, pack_rate, reference_decode  # noqa: E402
from real_units import Artifact, layer_units, reference_weights  # noqa: E402


def pack_expert(mode, units):
    """One expert -> (wire words, history words, (ra, rb, ksa), kperm, table, wscale)."""
    from tessera.fragment_wire import repack_fragment
    from tessera.wire import pack_body

    projs = ("gate_proj", "up_proj") if mode == 0 else ("down_proj",)
    us = [units[p] for p in projs]
    rates = us[0].rates
    if any(u.rates != rates for u in us):
        raise ValueError("gate and up differ in rate")
    rows, cols = us[0].codes.shape
    fw = repack_fragment(tuple(pack_body(u.codes.to(torch.int64), rates) for u in us), rates,
                         rows=rows, cols=cols, projection_group="gate_up" if mode == 0 else "down",
                         start_state=torch.stack([u.start for u in us]))
    hist_words = int(fw.unit_offsets[0, 0, 0] - fw.expert_offsets[0])
    slot_rates = fw.rates.tolist()
    ra, rb = slot_rates[0], slot_rates[-1]
    ksa = slot_rates.count(ra) if ra != rb else len(slot_rates)
    return (fw.words[hist_words:].clone(), fw.words[:hist_words].clone(), (ra, rb, ksa), fw.perm,
            torch.stack([u.table for u in us]), torch.stack([u.scale for u in us]))


def build(art, layer, rank, experts):
    """{mode: planes} for modes 0 and 2, each expert's units read once."""
    acc = {m: dict(words=[], hist=[], w0=[], h0=[], prof=[], kperm=[], table=[], wscale=[], wo=0, ho=0) for m in (0, 2)}
    for e in range(experts):
        units = layer_units(art, layer, e, rank)
        for mode, a in acc.items():
            w, h, pr, kp, tb, ws = pack_expert(mode, units)
            a["w0"].append(a["wo"]); a["h0"].append(a["ho"])
            a["wo"] += w.numel(); a["ho"] += h.numel()
            for key, v in zip(("words", "hist", "prof", "kperm", "table", "wscale"), (w, h, pr, kp, tb, ws)):
                a[key].append(v)
    return {mode: dict(mode=mode, wire=torch.cat(a["words"]), expert_word0=torch.tensor(a["w0"], dtype=torch.int64),
                       hist=torch.cat(a["hist"]), expert_hist0=torch.tensor(a["h0"], dtype=torch.int64),
                       rate=torch.tensor([pack_rate(*p) for p in a["prof"]], dtype=torch.int32),
                       kperm=torch.stack(a["kperm"]).contiguous(), table=torch.stack(a["table"]).contiguous(),
                       wscale=torch.stack(a["wscale"]).contiguous(), ks=a["kperm"][0].numel() // GROUPS[mode])
            for mode, a in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="/mnt/shared/tessera-measurements/pact-e4m3-accuracy-20260928/release-t8/exported")
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--experts", type=int, default=288)
    ap.add_argument("--check", default="0,1,143,287")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", help="path prefix of earlier stacks: require bit-equal planes")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    art = Artifact(a.artifact)
    rep = {"artifact": a.artifact, "layer": a.layer, "rank": a.rank, "experts": a.experts}
    t0 = time.time()
    stacks = build(art, a.layer, a.rank, a.experts)
    rep["pack_seconds"] = round(time.time() - t0, 1)
    for mode, st in stacks.items():
        path = os.path.join(a.out, f"stack-L{a.layer}-rank{a.rank}-mode{mode}.pt")
        torch.save(st, path)
        checks = {}
        for e in [int(v) for v in a.check.split(",") if int(v) < a.experts]:
            units = layer_units(art, a.layer, e, a.rank)
            want = ([reference_weights(units["gate_proj"]), reference_weights(units["up_proj"])] if mode == 0
                    else [reference_weights(units["down_proj"])])
            got = reference_decode(st, e)
            checks[e] = all(torch.equal(got[i], want[i]) for i in range(len(want)))
        if a.compare:
            old = torch.load(f"{a.compare}-mode{mode}.pt")
            checks["equal_to_" + os.path.basename(a.compare)] = all(
                torch.equal(old[k], st[k]) if torch.is_tensor(st[k]) else old[k] == st[k] for k in st)
        v = st["rate"]
        rep[f"mode{mode}"] = {"path": path, "ks": st["ks"],
                              "wire_words": st["wire"].numel(), "hist_words": st["hist"].numel(),
                              "profiles": sorted({(int(x) & 15, (int(x) >> 4) & 15, int(x) >> 8) for x in v}),
                              "fragment_equals_disk": checks}
        print(mode, rep[f"mode{mode}"], flush=True)
    json.dump(rep, open(os.path.join(a.out, f"real-stack-rank{a.rank}.json"), "w"), indent=1)
    assert all(all(rep[f"mode{m}"]["fragment_equals_disk"].values()) for m in (0, 2)), "packed planes differ from the disk unit"
    print("real stack packed and checked")


if __name__ == "__main__":
    main()
