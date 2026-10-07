"""Real GLM-5.3 routed units in the register-direct fragment order (eng-regdirect-build, stage 1 harness).

Packs one layer's routed experts, cut to one TP2 rank (``real_units.layer_units``), into the planes
``fragment_synth.make_stack`` returns, so the stage-1 bench runs the kernel on the disk codes.  It
is the inverse of ``fragment_synth.reference_decode``.  It is a harness, not the production
repack: ``tessera.fragment_wire`` replaces it when that lands.

Column groups (32 columns) are sorted by rate, ascending and stable: segment A holds the lowest
rate.  Gate/up: slot s holds group kperm[s] of both projections, which must share the rate.  Down:
slot s holds groups kperm[2s] and kperm[2s + 1], which must share the rate.  The history block
holds rows -4..-1 that the cut start state implies: row -k = (start >> (k - 1) R) & (2^R - 1).

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
from fragment_synth import GROUPS, HIST_LANES, KSTEP, TILE, WARPS, pack_rate, reference_decode  # noqa: E402
from real_units import Artifact, layer_units, reference_weights  # noqa: E402


def _words(fields, r):
    """int64 fields [..., lanes, 32] (R-bit codes, MSB-first) -> int32 words [..., R, lanes]."""
    sh = r - 1 - torch.arange(r)
    bits = ((fields.unsqueeze(-1) >> sh) & 1).flatten(-2)                  # [..., lane, 32R]
    bits = bits.reshape(*bits.shape[:-1], r, 32)
    w = (bits << (31 - torch.arange(32))).sum(-1)                           # [..., lane, R]
    w = torch.where(w >= 2**31, w - 2**32, w).to(torch.int32)
    return w.movedim(-1, -2)


def _group_rates(rates, what):
    g = torch.tensor(rates, dtype=torch.int64).reshape(-1, KSTEP)
    if not bool((g == g[:, :1]).all()):
        raise ValueError(f"{what}: a 32-column group holds two rates")
    return g[:, 0]


def _segments(group_rate, pair):
    """Slots ordered by rate: (kperm [slots * pair], [(rate, first slot, slots)])."""
    order = torch.argsort(group_rate, stable=True)
    gr = group_rate[order].reshape(-1, pair)
    if not bool((gr == gr[:, :1]).all()):
        raise ValueError("down: a unit k-step pairs two groups of different rates")
    rates = gr[:, 0]
    segs = []
    for r in torch.unique_consecutive(rates).tolist():
        idx = (rates == r).nonzero().reshape(-1)
        segs.append((int(r), int(idx[0]), idx.numel()))
    if len(segs) > 2:
        raise ValueError(f"stage 1 holds two rates per expert, got {[s[0] for s in segs]}")
    return order, segs


def pack_expert(mode, units):
    """One expert -> (tile-major words [T, words], history words, (ra, rb, ksa), kperm, table, wscale)."""
    if mode == 0:
        gate, up = units["gate_proj"], units["up_proj"]
        gr = _group_rates(gate.rates, "gate")
        if not torch.equal(gr, _group_rates(up.rates, "up")):
            raise ValueError("gate and up differ in rate on one column group")
        planes, starts, table, wscale = (gate.codes, up.codes), (gate.start, up.start), (gate.table, up.table), (gate.scale, up.scale)
    else:
        down = units["down_proj"]
        gr = _group_rates(down.rates, "down")
        planes, starts, table, wscale = (down.codes, down.codes), (down.start, down.start), (down.table,), (down.scale,)
    ng = GROUPS[mode]
    kperm, segs = _segments(gr, ng)
    rows = planes[0].shape[0]
    nt = rows // TILE
    if rows % TILE:
        raise ValueError(f"rows {rows} is not a multiple of {TILE}")
    tile_parts, hist = [], []
    for r, s0, n in segs:
        f = torch.empty(n, 2, rows, KSTEP, dtype=torch.int64)
        h = torch.empty(n, 2, 4, KSTEP, dtype=torch.int64)
        for p in range(2):
            grp = kperm[(s0 + torch.arange(n)) * ng + (0 if mode == 0 else p)]
            cols = (grp[:, None] * KSTEP + torch.arange(KSTEP)).reshape(-1)
            f[:, p] = planes[p][:, cols].to(torch.int64).reshape(rows, n, KSTEP).permute(1, 0, 2)
            st = starts[p][cols].to(torch.int64).reshape(n, KSTEP)
            for k in range(1, 5):
                h[:, p, 4 - k] = (st >> ((k - 1) * r)) & ((1 << r) - 1)
        # [slot, p, T, w, g, row, t, j] -> [T, slot, w, lane (g, t), field (p, j, row)]
        f = f.reshape(n, 2, nt, WARPS, 8, 2, 4, 8).permute(2, 0, 3, 4, 6, 1, 7, 5).reshape(nt, n, WARPS, 32, 32)
        tile_parts.append(_words(f, r).reshape(nt, -1))
        # [slot, p, g (6, 7), row, t, j] -> [slot, lane (g, t), field (p, j, row)]
        h = h.reshape(n, 2, 2, 2, 4, 8).permute(0, 2, 4, 1, 5, 3).reshape(n, HIST_LANES, 32)
        hist.append(_words(h, r).reshape(-1))
    ra, rb = segs[0][0], segs[-1][0]
    ksa = segs[0][2] if len(segs) == 2 else len(kperm) // ng
    return (torch.cat(tile_parts, 1).reshape(-1), torch.cat(hist), (ra, rb, ksa), kperm.to(torch.int16),
            torch.stack(table), torch.stack(wscale))


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
