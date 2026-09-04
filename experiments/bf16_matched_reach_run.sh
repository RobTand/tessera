#!/usr/bin/env bash
# Issue #18, the comparison its joint (L, ratio) grid could not make.
#
# That grid priced every window width against a byte-matched shipped
# reference and found the two axes entangled: at ratio 1 a wider table has
# more entries AND its outermost entry reaches further, so an `L` move is a
# bundle -- resolution and reach at once -- and only `L` costs bytes.  The
# receipt says so in the section that names this as a follow-up: "every L
# comparison here moves resolution and reach together, and the grid cannot
# say which of the two the winning cell is buying."
#
# This unbundles it.  `experiments/matched_reach.py` searches for the spread
# at which one width realises ANOTHER width's reach exactly (a search, not a
# division: the reach is the snapped outermost quantile, so it is a step
# function of the spread), and the rows below encode at those spreads.  The
# resulting factorial has
#
#   * rows  = table entry count (`L`), which costs bytes and is byte-matched
#             by the sweep against the shipped pair at the rung that spends
#             the same bytes;
#   * columns = spread, which costs NOTHING -- the artifact is the same size
#             at every value of it, so its ratio is read directly.
#
# The cheap row is `L=14`, the shipped width: its two off-diagonal cells are
# the shipped table at another width's reach, at zero byte cost.  If the L
# wins the grid found are spread wins, they are free and belong to #48's
# reach term; if they are entry-count wins, they cost bytes and belong to a
# wire decision.
#
# THE READING, REGISTERED BEFORE THE RUN.  For a (population, rung), let A be
# the landed byte-matched effect of the L-arm on its gate (GLM experts R=8:
# `L=16 r=1` at 0.9332x; dense R=4: `L=12 r=1` at 0.8916x) and B the effect
# of the `L=14` arm at that width's matched reach.  The recovered fraction is
# f = log B / log A, on the geomean and reported per unit.
#
#   f >= 0.5   the majority of the L win is spread, and it is available at
#              zero bytes and no change to the table's size.
#   f <= 0.15  the win is entry count; the spread constant is not where it
#              lives, and the earlier one-sided ratio grid (every PAIR_RATIO
#              is >= 1.0) missed nothing.
#   between    both halves are real and neither is dismissible.
#   f < 0      the two axes moved the gate in OPPOSITE directions, so no
#              fraction of one is recoverable from the other.  Both
#              orientations count: `B > 1 > A` is an L win the spread alone
#              reverses, and `B < 1 < A` an L arm that hurts while the spread
#              alone helps.  Reported as the sign it is, never as a band.
#
# WHERE THAT READING APPLIES, also registered before the run.  f is a ratio of
# two logarithms, so it is dominated by its own third digit wherever A is near
# 1: at GLM R=4 the landed L arms are 1.0029x and 1.0134x, and splitting a
# 0.3% effect into halves would report precision the quantity does not have.
# So f is COMPUTED and printed for every cell -- nothing is hidden -- but the
# verdict above is only READ where |log A| >= log 1.01, i.e. where the
# byte-matched L arm moves its gate by at least 1%.  Applied to the landed
# grids that admits all six dense cells (0.8916x to 1.1882x) and five of the
# six GLM ones, excluding exactly one: GLM R=4 `L=16` at 1.0029x.  This is
# a reporting convention chosen before the numbers, not a derived constant and
# not a noise floor: the encoder is deterministic and the repeat control is
# byte-identity, so there is no noise here to clear -- only conditioning.
#
# A physical check comes free: `rows_over_reach` at (14, r*) must equal it at
# (L, 1.0) exactly, on every unit -- the same reach clips the same rows.  A
# run where those disagree has not matched anything.
#
# The controls are the sweep's own -- the shipped pair (L=14, ratio 1.0) runs
# first and is repeated last in every process, byte- and tensor-identical --
# plus one this script adds: the shipped baseline here must be byte-identical
# to the same arm in the landed `pair_glm.json` / `pair_dense.json`, which is
# a cross-RUN control the in-process repeat cannot give.
# `experiments/matched_reach_report.py` checks it and refuses to summarise
# without it.
#
# Nothing here can move a default.  Both axes change `encoder_profile_id`,
# there is no BF16 serving lane, and every metric below is weight space or
# H-weighted columns.  House principle 3: this is a screen.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WT=${WT:-$HERE}
OUT=${OUT:-/mnt/shared/tessera-runs/reach/matched}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
export PYTHONPATH="$WT/src:$WT/experiments"
export TMPDIR=/home/rob/tmp
export TRITON_CACHE_DIR=/home/rob/.triton-cache

mkdir -p "$OUT" /home/rob/tmp
cd "$WT"

WIDTHS=${WIDTHS:-"12 14 16"}

# The ratios are COMPUTED here and passed in, never typed: `--ratios-for L`
# prints L's ratio at each width's reach, in `--widths` order, with the
# diagonal spelled exactly 1.0 so that cell runs the shipped `window_sigma
# is None` path rather than a re-spelling of it.
row() {
  local L=$1 pop=$2; shift 2
  local ratios
  ratios=$("$PY" experiments/matched_reach.py --widths $WIDTHS --ratios-for "$L")
  echo "=== $pop row L=$L  ratios=$ratios  $(date -Is)"
  "$PY" experiments/bf16_l_sigma_sweep.py --stage "pair-$pop" \
      --pair-bits "$L" --pair-ratios $ratios \
      --out "$OUT/mr_${pop}_L${L}.json" "$@"
}

# The eight dense Qwen Linears the landed `pair_dense.json` swept, named
# rather than inherited: `DENSE_UNITS` in `bf16_route_weight_space.py` is a
# DIFFERENT six, so a row taking the default would sweep a population the
# landed run never touched -- no shared shipped baseline, and therefore no
# cross-run control and no A to compare a B against.
DENSE=(model.layers.2.mlp.down_proj model.layers.2.self_attn.q_proj
       model.layers.2.self_attn.k_proj model.layers.14.mlp.gate_proj
       model.layers.2.mlp.up_proj model.layers.14.mlp.down_proj
       model.layers.27.self_attn.o_proj model.layers.14.self_attn.v_proj)

case "${1:-all}" in
  dense-14) row 14 dense --units "${DENSE[@]}" --rungs 1024 1536 2048 ;;
  dense-12) row 12 dense --units "${DENSE[@]}" --rungs 1024 1536 2048 ;;
  dense-16) row 16 dense --units "${DENSE[@]}" --rungs 1024 1536 2048 ;;
  glm-14)   row 14 glm --layers 5 20 42 --projs gate_proj up_proj \
                       --experts 0 --rungs 1024 1536 2048 ;;
  glm-16)   row 16 glm --layers 5 20 42 --projs gate_proj up_proj \
                       --experts 0 --rungs 1024 2048 ;;
  glm-12)   row 12 glm --layers 5 20 42 --projs gate_proj up_proj \
                       --experts 0 --rungs 1024 2048 ;;
  *) echo "usage: $0 {dense,glm}-{12,14,16}" >&2; exit 2 ;;
esac
echo "=== done $(date -Is)"
