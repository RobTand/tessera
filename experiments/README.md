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
| `uniform_control.py` | `tessera-allocated-served-2026-09-02.md` §4 and §7 — the byte-matched uniform arm, and the verdict against it |
| `tessera_dominated_rungs.py` | `tessera-dominated-rungs-2026-09-02.md` — the dominated-rung table, the accountant-vs-exporter identity, and the both-axes quality leg |

They expect `PYTHONPATH=src` and a writable `TRITON_CACHE_DIR`. They read
Qwen3.8-27B from the local HF cache and are not hermetic.

**`uniform_control.py` is not scratch.** It is the standing gate of
RobTand/tessera#3, and the one check that saw the 2026-09-02 failure: `plan` writes the
byte-matched uniform control for a candidate `--plan-json` with the match
asserted, `verify` re-asserts it on the two exported manifests and records the
verdict. Its library half is `tessera.control`, which `plan_from_layer_config.py`
now prices into every sidecar it writes. The run-specific drivers that produced
the receipt stay verbatim in `allocated_serve_2026-09-02/`.

**`lloyd_max` in `freegrid.py` is the one to promote.** If free grids become a
lane, the level construction is wire — two artifacts over different grids decode
differently — and it must be deterministic and versioned, not a scratch
function. See the fail-closed note in `tessera-8-and-the-payload-grid.md`.
