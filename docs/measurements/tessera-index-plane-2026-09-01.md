# The scale plane's redundancy is cashable: a 4-bit index plane loses nothing

**Date:** 2026-09-01. **Script:** `experiments/tessera_index_plane.py`
(`--codebook subset`, results `experiments/results/tessera_index_plane_subset.json`;
the first run, `tessera_index_plane.json`, used a k-means codebook whose E4M3
snapping collapsed centroids to 11 distinct values of 16 — 3.4% worse; kept
as the record of why the codebook must be a *distinct* E4M3 subset).
**Tensors:** six GLM-5.3-Flash routed experts (L5/20/42 × gate/up, expert 0,
2048×4096). **Eval:** the last 1024 rows (two documents) of the 16-document
pread capture, held out from nothing here — every arm is a pure weight-side
change, so the fit rows are not used; the served activation quantiser's
global scale is fit on the first 14 documents. **Metric:** output-space
weight leg (`|x(ŵ−w)ᵀ| / |xwᵀ|`) and the served W4A4 error; ratios are the
mean over tensors of the per-tensor ratio.

## Why this was run

`tessera-theoretical-limits-2026-09-01.md` §2 found the per-16 E4M3 plane
carries 0.215 bpp of information in 0.5 bpp of bytes (27–30 distinct values
over 3.5 octaves), and §4 concluded the difference could not be cashed —
but the only arm that tried (per-32 E4M3 + Wei L=2 at 4.0 bpp, 1.028× on
this eval) paid for its bits by halving the loading granularity. The
reviewer's question: keep per-16 loading, cash the redundancy with a
per-tensor codebook of `k` E4M3 values and a `log2(k)`-bit index per 16
weights. Decode is still E2M1 codes × a per-16 E4M3 scale: the kernel lane
reads the scale through a 16-entry LUT, the stock lane materialises the
same E4M3 tile. FP4-native either way.

**Codebook fit.** With the codes fixed, a half's error is `A s² − 2 B s + C`
(`A = <u,u>`, `B = <w,u>`), so the SSE-optimal codebook for the LS targets
`s* = B/A` is weighted 1-D k-means with weights `A` — but a continuous
Lloyd whose centroids are snapped to E4M3 each step *collapses* (16 → 11
distinct). The right fit is over the finite candidate set: greedy backward
elimination from every in-range E4M3 value down to `k` distinct ones, then
swap passes (`e4m3_subset`). Quantisation is nearest-in-linear (the
parabola's minimiser). The plane alternates with the trellis exactly as the
refit schedule does (3 rounds).

## Result

| arm (plane, trellis) | bpp | out-space weight leg | vs flat E4M3/16, L=1 (4.0) | W4A4 | vs EXL3@A4 |
|---|---:|---:|---:|---:|---:|
| idx8 per-16, L=1 | 3.6875 | 0.09225 | 0.967× | 0.12629 | 1.217× |
| E4M3 per-32, L=1 | 3.7500 | 0.09610 | 0.928× | 0.12912 | 1.244× |
| **idx16 per-16, L=1** | **3.7500** | **0.08925** | **1.000×** | 0.12415 | 1.196× |
| idx32 per-16, L=1 | 3.8125 | 0.08913 | 1.001× | 0.12405 | 1.195× |
| idx8 per-16, L=2 | 3.9375 | 0.08359 | 1.068× | 0.12016 | 1.157× |
| flat E4M3 per-16, L=1 (today's bytes, refit) | 4.0000 | 0.08924 | 1.000× | 0.12413 | 1.196× |
| E4M3 per-32, L=2 (the frontier's "compact plane") | 4.0000 | 0.08672 | 1.028× | 0.12231 | 1.178× |
| **idx16 per-16, L=2** | **4.0000** | **0.08030** | **1.111×** (1.100–1.114) | **0.11791** | **1.136×** |
| idx32 per-16, L=2 | 4.0625 | 0.08022 | 1.112× | 0.11786 | 1.135× |
| E4M3 per-16, L=2 | 4.2500 | 0.08026 | 1.112× | 0.11789 | 1.136× |

Three facts, each holding on all six tensors:

1. **A 16-entry per-tensor E4M3 codebook is lossless.** idx16 at L=1
   (3.75 bpp) reproduces the full 0.5-bpp E4M3 plane to the third digit
   (1.000×, min 0.997, max 1.002); idx32 adds nothing. The plane's
   information is ≤ 4 bits per 16 weights, as the entropy said.
2. **The freed 0.25 bpp, spent on Wei L=2, buys 1.111× at the same 4.0 bpp**
   — the number the frontier had priced at 4.25 bpp. Rob's "tepid 1.05"
   was the per-32 plane's loading loss, not the alphabet's rate cap:
   the same L=2 payload is worth 1.028× behind per-32 and 1.111× behind
   per-16.
3. **idx8 is a real 3.6875 rung** (3.3% under the per-16 plane) and
   idx8 + L=2 at 3.9375 is 1.068× — the plane axis is now
   `{3 bits, 4 bits}` per 16 with a known price, and per-32 E4M3 is
   dominated at every rate.

Against today's *default* bytes (the S6b two-tier plane, refit-4, 1.7%
behind flat E4M3 under refit), idx16 + L=2 at 4.0 bpp is ≈1.13× on the
weight leg. On the served W4A4 metric the gap to the hypothetical EXL3@A4
moves from 1.196× to 1.136×.

## What the wire change is

Today's S6b plane is one E8M0 byte per 32 plus a 4-bit `(d, m)` word per
16. The index plane is that same 4-bit word per 16 with the E8M0 tier
**deleted** and the word's meaning changed from `(d, m)` relative to the
base to an index into a 16-entry per-unit E4M3 LUT (16 bytes in the
manifest, plus the fp32 global the stock lane already carries). The
0.25 bpp the E8M0 tier occupied moves to the trellis as Wei L=2: one
convolutional-code bit per *pair* of positions, two bits choosing the
first position's subset (the second is the pair's label minus it, mod 4),
and the usual `R−1` point bits per position — `2R+1` bits per super-symbol
at rung `R`, i.e. `R + ½` per position, 3.75 b/wt at the cap. The decode
tile changes from "one select bit per code" to "one select bit per two
codes plus a 2-bit subset word"; the LUTs the kernel lane already splits
(`build_tuple_index_lut` / `build_anchor_values`) carry over.

## Evidence class

A per-tensor screen (six tensors, weight leg + served activation
quantiser, 1024 held-out rows). It is the same class as every other
number in the rate/plane frontier and the refit tables; promotion to the
served KL harness follows the wire build, like the refit A/B.

---

**Built 2026-09-01 (later the same evening):** the wire change is in
`src/tessera/` as schema minor 1 and is the exporter's default. The
production encoder, cold-starting the table from the amax targets, measures
1.125× over the span-1 S6b default at 4.0 bpp on the same six tensors —
`docs/measurements/tessera-wire-default-2026-09-01.md`.
