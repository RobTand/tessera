#!/usr/bin/env bash
# SASS and resource usage of one library across A/B arms built by build_ext.sh,
# disassembled inside the measurement image (cuobjdump), addresses and
# encodings stripped so two arms compare by instruction.
#   sass_arms.sh <out_root> <reference arm> <arm> [<arm> ...]
# SASS_LIB: the library (default tessera_routed_fused_mma_e4m3).
# Writes <out_root>/sass/<arm>.{sass,res,norm}; prints, per arm, whether its
# normalized SASS equals the reference arm's and the routed kernels' register,
# stack and local-memory usage.  Exit 0 whatever the comparison says.
set -uo pipefail
OUT=$(realpath "$1"); REF=$2; shift 2
IMG=${ORACLE_IMAGE:?set ORACLE_IMAGE to the immutable PB-declared measurement image}
NAME=${SASS_LIB:-tessera_routed_fused_mma_e4m3}
mkdir -p "$OUT/sass"
for a in "$REF" "$@"; do
  : > "$OUT/sass/$a.norm"   # a failed disassembly leaves nothing comparable
  lib=$(ls "$OUT/ext-$a"/${NAME}_*/${NAME}.so 2>/dev/null | head -1)
  [[ -n "$lib" ]] || { echo "$a: no $NAME library under $OUT/ext-$a"; continue; }
  docker run --rm --network=none --user "$(id -u):$(id -g)" -v "$OUT":"$OUT" --entrypoint bash "$IMG" -c \
    "cuobjdump -sass '$lib' > '$OUT/sass/$a.sass' && cuobjdump -res-usage '$lib' > '$OUT/sass/$a.res'" \
    || { echo "$a: cuobjdump failed"; continue; }
  sed -E 's@/\*[0-9a-f]{4,}\*/@@; s@/\* 0x[0-9a-f]+ \*/@@; s@[[:space:]]+$@@' "$OUT/sass/$a.sass" \
    | grep -v '^[[:space:]]*$' > "$OUT/sass/$a.norm"
  echo "$a: $(grep -c 'Function :' "$OUT/sass/$a.sass") functions, $(wc -l < "$OUT/sass/$a.norm") lines"
done
for a in "$@"; do
  if [[ ! -s "$OUT/sass/$REF.norm" || ! -s "$OUT/sass/$a.norm" ]]; then echo "SASS $a vs $REF: NOT COMPARED (a disassembly is missing)"
  elif cmp -s "$OUT/sass/$REF.norm" "$OUT/sass/$a.norm"; then echo "SASS $a == $REF: IDENTICAL"
  else echo "SASS $a vs $REF: $(diff "$OUT/sass/$REF.norm" "$OUT/sass/$a.norm" | grep -c '^[<>]') lines differ"; fi
done
for a in "$REF" "$@"; do
  echo "== $a routed_fused_kernel resource usage (count of instantiations per value)"
  grep -A1 "routed_fused_kernel" "$OUT/sass/$a.res" | grep -oE "REG:[0-9]+|STACK:[0-9]+|LOCAL:[0-9]+" | sort | uniq -c | sort -k2
done
exit 0
