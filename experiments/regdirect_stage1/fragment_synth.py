"""Synthetic fragment-order planes and their reference decode (eng-regdirect-build, stage 1 harness).

A random wire in the layout ``tessera.regdirect_routed`` reads (design note section 2):
per expert, per 128-row tile, the units of rate segment A (slots [0, ksa)) then segment B; a unit
is 8 warps x R words x 32 lanes, word i of lane L at i * 32 + L; lane (g, t) holds pair q = p * 8 + j
at bit q * 2R (MSB-first), rows 2g and 2g + 1 of the warp's 16 rows, column 8t + j of group p's
32-column group.  The history block holds, per slot, R words x 8 lanes: lanes (6, t) and (7, t) of a
block before row 0, i.e. the codes of rows -4..-1 that the column's start state implies.

``reference_decode`` reads only these planes and the window rule; it does not share code with the
kernel.  When ``tessera.fragment_wire`` lands, the disk-to-fragment tests replace this generator.
"""
from __future__ import annotations

import torch

WARPS, TILE, KSTEP, HIST_LANES, WIN = 8, 128, 32, 8, 14
GROUPS = {0: 1, 2: 2}
TABLES = {0: 2, 2: 1}


def pack_rate(ra, rb, ksa):
    return ra | (rb << 4) | (ksa << 8)


def make_stack(mode, experts, rows, ks, profiles, seed, device):
    """``profiles[e] = (ra, rb, ksa)``.  Returns a dict of the FragmentStack planes."""
    g = torch.Generator(device=device).manual_seed(seed)
    nt = rows // TILE
    ng = GROUPS[mode]
    words, hist, w0, h0 = [], [], [], []
    wo = ho = 0
    for e in range(experts):
        ra, rb, ksa = profiles[e]
        tile_words = ksa * WARPS * ra * 32 + (ks - ksa) * WARPS * rb * 32
        w = torch.randint(-2**31, 2**31 - 1, (nt * tile_words,), generator=g, device=device, dtype=torch.int32)
        h = torch.randint(-2**31, 2**31 - 1, (ksa * ra * HIST_LANES + (ks - ksa) * rb * HIST_LANES,),
                          generator=g, device=device, dtype=torch.int32)
        words.append(w); hist.append(h); w0.append(wo); h0.append(ho)
        wo += w.numel(); ho += h.numel()
    kperm = torch.stack([torch.randperm(ks * ng, generator=g, device=device) for _ in range(experts)]).to(torch.int16)
    t = torch.randint(0, 256, (experts, TABLES[mode], 1 << WIN), generator=g, device=device, dtype=torch.int32)
    table = torch.where((t & 0x7F) == 0x7F, t - 1, t).to(torch.uint8)              # no E4M3 NaN
    wscale = torch.rand(experts, TABLES[mode], rows, generator=g, device=device) * 1e-2 + 1e-3
    return dict(mode=mode, wire=torch.cat(words), expert_word0=torch.tensor(w0, dtype=torch.int64, device=device),
                hist=torch.cat(hist), expert_hist0=torch.tensor(h0, dtype=torch.int64, device=device),
                rate=torch.tensor([pack_rate(*p) for p in profiles], dtype=torch.int32, device=device),
                kperm=kperm.contiguous(), table=table, wscale=wscale, ks=ks)


def _lane_fields(words, r):
    """[..., R, 32 lanes] int32 -> fields [..., 8 g, 4 t, 2 p, 8 j, 2 rows] (R-bit codes)."""
    w = (words.to(torch.int64) & 0xFFFFFFFF).movedim(-1, -2)                 # [..., lane, R]
    sh = 31 - torch.arange(32, device=w.device)
    bits = ((w.unsqueeze(-1) >> sh) & 1).to(torch.int32).flatten(-2)          # [..., lane, 32R]
    bits = bits.reshape(*bits.shape[:-2], 8, 4, 2, 8, 2, r)
    weight = (1 << (r - 1 - torch.arange(r, device=w.device))).to(torch.int32)
    return (bits * weight).sum(-1)


def reference_decode(st, e):
    """E4M3 weight bytes [tables, rows, K] of expert ``e`` from the fragment planes alone."""
    mode, ks = st["mode"], st["ks"]
    ng = GROUPS[mode]
    rows = st["wscale"].shape[2]
    nt = rows // TILE
    v = int(st["rate"][e])
    ra, rb, ksa = v & 15, (v >> 4) & 15, v >> 8
    out = torch.zeros(TABLES[mode], rows, ks * ng * KSTEP, dtype=torch.uint8, device=st["wire"].device)
    kp = st["kperm"][e].long()
    w0, h0 = int(st["expert_word0"][e]), int(st["expert_hist0"][e])
    tile_words = ksa * WARPS * ra * 32 + (ks - ksa) * WARPS * rb * 32
    tiles = st["wire"][w0:w0 + nt * tile_words].reshape(nt, tile_words)
    for r, lo, n, woff, hoff in ((ra, 0, ksa, 0, h0), (rb, ksa, ks - ksa, ksa * WARPS * ra * 32, h0 + ksa * ra * HIST_LANES)):
        if n == 0:
            continue
        seg = tiles[:, woff:woff + n * WARPS * r * 32].reshape(nt, n, WARPS, r, 32)
        f = _lane_fields(seg, r)                         # [T, slot, w, g, t, p, j, row]
        f = f.permute(1, 5, 0, 2, 3, 7, 4, 6)              # slot, p, T, w, g, row, t, j
        f = f.reshape(n, 2, rows, KSTEP)                  # rows n = T*128 + w*16 + 2g + row; col 8t + j
        hseg = st["hist"][hoff:hoff + n * r * HIST_LANES].reshape(n, r, HIST_LANES)
        hb = (hseg.to(torch.int64) & 0xFFFFFFFF).movedim(-1, -2)                  # [slot, 8 lanes, R]
        sh = 31 - torch.arange(32, device=hb.device)
        bits = ((hb.unsqueeze(-1) >> sh) & 1).to(torch.int32).flatten(-2).reshape(n, 2, 4, 2, 8, 2, r)
        hf = (bits * (1 << (r - 1 - torch.arange(r, device=hb.device))).to(torch.int32)).sum(-1)
        hf = hf.permute(0, 3, 1, 5, 2, 4).reshape(n, 2, 4, KSTEP)   # slot, p, rows -4..-1, col
        full = torch.cat([hf, f], 2).to(torch.int64)                # [slot, p, 4 + rows, 32]
        state = torch.zeros(n, 2, rows, KSTEP, dtype=torch.int64, device=full.device)
        for i in range(-(-WIN // r)):
            state |= full[:, :, 4 - i:4 - i + rows, :] << (i * r)
        state &= (1 << WIN) - 1
        for p in range(2):
            proj = p if mode == 0 else 0
            vals = st["table"][e, proj][state[:, p]]                  # [slot, rows, 32]
            cg = kp[(lo + torch.arange(n, device=kp.device)) * ng + (0 if mode == 0 else p)]
            cols = (cg[:, None] * KSTEP + torch.arange(KSTEP, device=kp.device)[None]).reshape(-1)
            out[proj][:, cols] = vals.permute(1, 0, 2).reshape(rows, -1)
    return out
