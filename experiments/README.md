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

They expect `PYTHONPATH=src` and a writable `TRITON_CACHE_DIR`. They read
Qwen3.8-27B from the local HF cache and are not hermetic.

**`lloyd_max` in `freegrid.py` is the one to promote.** If free grids become a
lane, the level construction is wire — two artifacts over different grids decode
differently — and it must be deterministic and versioned, not a scratch
function. See the fail-closed note in `tessera-8-and-the-payload-grid.md`.
