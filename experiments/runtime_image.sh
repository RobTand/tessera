#!/usr/bin/env bash
# Refuse a floating serve image, and hand the resolved digest to the receipt.
#
# Sourced by every wrapper that starts a container.  Two functions, no side
# effects at source time.
#
# WHY (issue #100).  Every wrapper here named `vllm/vllm-openai:latest`, and a
# tag is not a pin: upstream can repoint it, two boxes can hold two builds
# under it, and every receipt would still record the same four words.  The pin
# is a digest reference and it lives in ONE place --
# src/tessera/serving/runtime_contract.json's versions.attested_on.image -- read
# from here, never copied into a script.  See src/tessera/serving/runtime_image.py
# for the rule, the RepoDigests-not-.Id trap, and why an unpinned repository is
# stamped rather than refused.
#
#   runtime_image_pin                -> print the pinned pull reference
#   runtime_image_require IMAGE      -> refuse (exit 2) unless IMAGE is the pin;
#                                       sets RUNTIME_IMAGE_{DIGEST,LOCAL_ID,JSON}
#
# The refusal happens BEFORE serve_lock_acquire in every caller: a wrapper that
# is going to refuse must not first take the box's one serve lock and make
# fourteen other agents queue behind a failure.

# Resolve from THIS file, not from a caller's $TS: a worktree's wrapper must
# gate on its own checkout's pin, not the main checkout's.
_RUNTIME_IMAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_IMAGE_PY="${RUNTIME_IMAGE_PY:-python3}"

_runtime_image_cli() {
  PYTHONPATH="$_RUNTIME_IMAGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$RUNTIME_IMAGE_PY" -m tessera.serving.runtime_image "$@"
}

runtime_image_pin() { _runtime_image_cli pin; }

runtime_image_require() {
  local image="$1"
  # stdout is the machine-readable record (refused or not); stderr is the prose.
  # Keep both: a gate only a human can read gets skipped by the script that
  # should have honoured it.
  local json rc=0
  json="$(_runtime_image_cli resolve --image "$image")" || rc=$?
  RUNTIME_IMAGE_JSON="$json"
  if [ "$rc" != "0" ]; then
    # Unlike build_identity_stamp, this IS fatal.  A warning nothing reads is a
    # confession log, not a gate (principle 9), and the whole point of #100 is
    # that the previous behaviour -- run whatever `latest` happens to be --
    # recorded nothing about what ran.  The record goes to stdout so a caller
    # that captures the wrapper's output has the refusal in a form it can
    # parse; the CLI has already written the prose to stderr for a reader.
    echo "$json"
    return 2
  fi
  RUNTIME_IMAGE_DIGEST="$(printf '%s' "$json" | _runtime_image_field resolved_digest)"
  RUNTIME_IMAGE_LOCAL_ID="$(printf '%s' "$json" | _runtime_image_field local_id)"
  echo "image $image -> ${RUNTIME_IMAGE_DIGEST:-<no manifest digest>} (local id ${RUNTIME_IMAGE_LOCAL_ID:-unknown})"
}

_runtime_image_field() {
  "$RUNTIME_IMAGE_PY" -c 'import json,sys; v=json.load(sys.stdin).get(sys.argv[1]); print("" if v is None else v)' "$1"
}
