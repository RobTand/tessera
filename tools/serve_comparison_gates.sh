#!/usr/bin/env bash
# Serve-comparison gates published from the reviewed deployment orchestrators.
#
#   artifact_gate    (issue #885) The existing shared identity owner
#                    (comparison_arm_identity.py) authenticates the ordered
#                    owned arm population and the actual loaded arm/audit
#                    bytes against the frozen comparison binding. The
#                    orchestrator calls it BEFORE any resource is taken --
#                    before memory checks, preflight, locks or fences -- and
#                    takes the artifact path from its output; a hardcoded or
#                    stale export identity is the defect this gate refuses.
#   generation_gate  (issue #872) Requires the complete paired generated
#                    population (decoded UTF-8 text + length finish, warmup
#                    included) before an arm may SHIP. Actual timed outputs,
#                    never teacher-forced TR3.
#   lever_chain      (issue #872) The base arm, then the gate arm and (only
#                    on a gate failure) the fallbacks in order; the first arm
#                    whose TR3 is BITWISE against the base and whose
#                    lever_check is clean AND whose generation gate passes is
#                    SHIP. A generation failure never falls through to SHIP.
#
# Sourcing context must provide: COMPARISON_MANIFEST, LEAD_PIN, ARMS, MTP_ARM
# and H (harness directory holding arms/*.env) for artifact_gate, and
# lever_chain's collaborators (say, run_val, tr3_gate, lever_check, and the
# arm variables). artifact_gate and generation_gate prefer the
# SERVE_GATE_TOOL_DIR / SERVED_GENERATION_CLIENT overrides and otherwise use
# this script's own directory.
#
# These gates bind input/model identity only. Quality, native, admission,
# image and serve authorization each keep their existing owner.

artifact_gate() {  # existing shared owner authenticates actual loaded arm/audit bytes
  local a owner_dir
  owner_dir=${SERVE_GATE_TOOL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
  local -a command=(python3 "$owner_dir/comparison_arm_identity.py" comparison-artifact --manifest "$COMPARISON_MANIFEST" --pin "$LEAD_PIN")
  for a in $ARMS ${MTP_ARM:-}; do command+=(--arm "$H/arms/$a.env"); done
  "${command[@]}"
}

generation_gate() {  # candidate/reference run dirs, candidate arm; actual timed outputs, not TR3
  local client gate_dir
  gate_dir=${SERVE_GATE_TOOL_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
  client=${SERVED_GENERATION_CLIENT:-$gate_dir/served_generation_client.py}
  python3 "$client" compare-generation --manifest "$COMPARISON_MANIFEST" \
    --reference "$2/speed" --candidate "$1/speed" --out "$1/generation-exactness.json"
}

lever_chain() {
  local a g lvs grc R=${CHAIN_ROOT:-$U/runs}
  say "stage 5: base arm $BASE_ARM"; run_val "$BASE_ARM"
  if [ "$RV_STATUS" != ran ] || [ "$RV_KV" != 0 ] || [[ "$RV_LV" = VOID* ]]; then
    GATE_TRAIL="$BASE_ARM: reference source/dispatch evidence refused; candidate comparison not decidable"
    say "$GATE_TRAIL"; return 1
  fi
  local BASE_TR3; BASE_TR3=$(tr3_gate "$R/$WID-$BASE_ARM" "$R/$WID-$BASE_ARM"); BASE_TR3=${BASE_TR3%% *}
  for a in $GATE_ARM $FALLBACK_ARMS; do
    say "stage 5: lever-gate arm $a"
    run_val "$a"
    if [ "$RV_STATUS" != ran ]; then
      GATE_TRAIL="$GATE_TRAIL | $a: $RV_STATUS (box state, not a lever result; chain stopped)"
      say "lever gate: $a $RV_STATUS; no fallback (a fallback would face the same box state)"; break
    fi
    if [ "$BASE_TR3" != BITWISE ]; then
      GATE_TRAIL="$GATE_TRAIL | $a: base $BASE_ARM has no TR3 result; gate not decidable, fallbacks not run"
      say "lever gate: base $BASE_ARM has no TR3 result: $a is not decidable; no fallback"; break
    fi
    g=$(tr3_gate "$R/$WID-$a" "$R/$WID-$BASE_ARM")
    lvs=levers-ok; case "$RV_LV" in VOID*) lvs="levers ${RV_LV%% | *}" ;; esac
    GATE_TRAIL="$GATE_TRAIL | $a: ${g%%;*}; $lvs; kernel rc $RV_KV"
    say "lever gate: $a TR3 vs $BASE_ARM: $g; $lvs; kernel evidence rc $RV_KV"
    printf '\nLEVER GATE %s (%s: TR3 vs %s %s; %s; kernel rc %s)\n' "$(date -u +%FT%TZ)" "$WID-$a" "$BASE_ARM" "${g%%;*}" "$lvs" "$RV_KV" >> "$N"
    grc=1
    if [ "${g%% *}" = BITWISE ] && [ "$lvs" = levers-ok ] && [ "$RV_KV" = 0 ]; then
      generation_gate "$R/$WID-$a" "$R/$WID-$BASE_ARM" "$a"; grc=$?
      GATE_TRAIL="$GATE_TRAIL; generation rc $grc"
      say "lever gate: $a generated-output evidence rc=$grc"
      if [ "$grc" = 0 ]; then SHIP=$a; SHIP_RUN=$R/$WID-$a; say "lever gate: $a PASSES (ship arm)"; break; fi
    fi
    say "lever gate: $a does not pass; next fallback: $(echo " $FALLBACK_ARMS " | sed "s/.* $a //" | awk '{print $1}')"
  done
  [ -n "$SHIP" ] || say "lever gate: no arm passed ($GATE_TRAIL)"
  GATE_TRAIL=${GATE_TRAIL# | }
}
