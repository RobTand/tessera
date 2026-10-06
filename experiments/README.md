# Measurement scripts

The code behind `docs/measurements/`. These were written as scratch and are
preserved here because the docs cite their numbers: a result whose reproduction
code has been deleted is not a reproducible result.

| script | backs |
|---|---|
| `curve.py`, `pair.py`, `pair_seed.py` | `release-vs-tuple-trellis.md` — the k-tuple rate sweep and the 2.12 dB pair-trellis measurement |
| `t8.py`, `t8curve.py` | `tessera-8-and-the-payload-grid.md` results 1 and 2 |
| `freegrid.py` | result 3, and the `lloyd_max` construction the free-grid arms use |
| `kwide.py`, `ksliced.py`, `kverify.py` | `kernel-lane.md` — block sweeps, one-hot exactness, the matched comparator |
| `loadcost.py`, `bw.py` | `nvfp4-kernel-attestation.md` — decode cost, and the box's achievable bandwidth |
| `rotfull.py` | `rotation-decision.md` — the full-tensor re-measurement |
| `tessera_dominated_rungs.py` | `tessera-dominated-rungs-2026-09-02.md` — the dominated-rung table, the accountant-vs-exporter identity, and the both-axes quality leg |
| `t8r_speed/bench_geometry.py`, `t8r_speed/rung_quality.py`, `t8r_speed/rung_allowability_table.py` | D41 (#989): every scalar q256 rung 768–1152 at M=1/16/2048/4096; forward/reverse CUDA timing includes balanced and existing recorded M2048/M4096 routing, separate CPU actual-expert weight-space quality, and schema/semantic table production; never served KL |

They expect `PYTHONPATH=src` and a writable `TRITON_CACHE_DIR`. They read
Qwen3.8-27B from the local HF cache and are not hermetic.

**The standing uniform-control gate is installed, not scratch.** Invoke
`python -m tessera.uniform_control plan|verify` (RobTand/tessera#886). It is the
standing gate of RobTand/tessera#3: `plan` writes the byte-matched control with
the match asserted; `verify` re-asserts it on exported manifests and records
the unchanged `tessera.uniform_control.v1` verdict. Its library owner remains
`tessera.control`. Run-specific reproduction drivers stay verbatim in
`allocated_serve_2026-09-02/` and belong to their historical source revision.

**`lloyd_max` in `freegrid.py` is the one to promote.** If free grids become a
lane, the level construction is wire — two artifacts over different grids decode
differently — and it must be deterministic and versioned, not a scratch
function. See the fail-closed note in `tessera-8-and-the-payload-grid.md`.
