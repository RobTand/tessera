#!/usr/bin/env bash
# The three receipt legs the campaign itself does not print (issue #102).
#
#   (b) the prefill regime is UNCHANGED.  A dump file cannot be byte-compared
#       across runs -- its meta carries a wall clock, an elapsed time and an
#       argv -- so `kl_tool fingerprint` hashes the scored values plus the
#       metric identity and nothing else.  The before-state is a dump frozen
#       on disk BEFORE the tool was touched (#83's arms); the after-state is
#       this campaign's prefill dump of the same bytes on the same corpus.
#   (c) a cross-regime compare is REFUSED, and --allow-mismatch does not
#       reach it.  Printed here as a transcript with its exit status, because
#       a refusal nobody has seen fire is a claim.
#   (d) the position-matched contrast: the prefill pair restricted to the
#       decode regime's positions, so the decode number can be read against a
#       prefill number over the SAME positions and the only difference left
#       is which forward ran.
#
# usage: decode_regime_evidence.sh
set -uo pipefail
KLDIR=/mnt/shared/tessera-kl
RUNS=${RUNS:-/home/rob/tessera-runs/ts102}
PY=${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}
KL=${KL:-/home/rob/dq-runs/kl_tool.py}
BASE=${BASE:-$KLDIR/qwen_tessera_ts83-armA-streamed-eager.json.npz}  # frozen pre-change
mkdir -p "$RUNS"

echo "=== (b) prefill fingerprints: the pre-change dump, and this campaign's"
for P in "$BASE" "$KLDIR/qwen_ts102_ts102-armA_prefill.json.npz" \
         "$KLDIR/qwen_ts102_ts102-armB_prefill.json.npz"; do
  [ -f "$P" ] || { echo "  MISSING $P"; continue; }
  echo "--- $(basename "$P")"
  $PY "$KL" fingerprint "$P"
done

echo
echo "=== (c) cross-regime compare, refused"
for EXTRA in "" "--allow-mismatch"; do
  echo "--- kl_tool compare <prefill teacher> <decode student> $EXTRA"
  $PY "$KL" compare "$KLDIR/qwen_teacher_bf16_v028.json.npz" \
      "$KLDIR/qwen_ts102_ts102-armA_decode.json.npz" $EXTRA
  echo "  exit status: $?"
done

echo
echo "=== (d) the prefill pair restricted to the decode regime's positions"
$PY "$(dirname "$0")/decode_regime_subset.py" \
  --teacher-prefill "$KLDIR/qwen_teacher_bf16_v028.json.npz" \
  --student-prefill "$KLDIR/qwen_ts102_ts102-armA_prefill.json.npz" \
  --decode-student "$KLDIR/qwen_ts102_ts102-armA_decode.json.npz" \
  --json "$RUNS/subset_armA.json"
