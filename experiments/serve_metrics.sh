#!/usr/bin/env bash
# The serving gates that read /metrics, in one place (tessera#247).
#
# Spec-decode poisons a logprob readout: /v1/completions returns the DRAFT
# model's numbers when vLLM serves with a speculative config, so every KL
# wrapper refuses before it dumps a position.  Three of them spelled that
# refusal as
#
#     if curl -s ".../metrics" | grep -q 'vllm:spec_decode'; then
#
# and that pipeline cannot detect the condition it owns.  `grep -q` exits at
# its FIRST match and closes the pipe; a real metrics response is far longer
# than the marker, so curl is still writing and fails (write error / SIGPIPE).
# Under `set -o pipefail` the pipeline's status is then curl's, the `if` reads
# false, and the refusal body is skipped -- the marker was found and the wrapper
# dumped anyway.  Because the pipeline is an `if` condition, errexit does not
# abort either.  The same defect was already found and fixed one gate down, in
# `serve_and_dump_kl.sh`'s startup-log check: grep the FILE, never a pipe.
#
# A failed fetch took the accepting branch too.  `curl -s` prints nothing and
# exits non-zero on a connection failure, and does not even reject an HTTP
# error response -- so "the serve could not be asked" and "the serve is not
# speculative" were the same answer.  A gate that cannot read its evidence
# refuses; it does not pass.
#
# usage:  source "$(dirname "$0")/serve_metrics.sh"
#         if ! serve_require_no_spec_decode "$PORT" "$METRICS_FILE"; then ...

#: The metric family vLLM publishes only when a speculative config is active.
SERVE_SPEC_DECODE_MARKER='vllm:spec_decode'

# serve_metrics_fetch PORT DEST
#   Write the COMPLETE /metrics body to DEST.  0 only on a 200 whose body was
#   received in full; non-zero, with the reason on stderr, otherwise.
serve_metrics_fetch() {
  local port="$1" dest="$2"
  local code status
  rm -f "$dest"
  code=$(curl -s --show-error -o "$dest" -w '%{http_code}' \
         "http://127.0.0.1:${port}/metrics") && status=0 || status=$?
  if [ "$status" -ne 0 ]; then
    echo "REFUSED: could not read the serve's /metrics on port ${port}" \
         "(curl exit $status).  A gate that cannot fetch its evidence" \
         "refuses; an unreachable serve is not a serve without spec-decode" >&2
    return 1
  fi
  if [ "$code" != "200" ]; then
    echo "REFUSED: the serve's /metrics on port ${port} answered HTTP" \
         "${code}.  An error response is not evidence that spec-decode is" \
         "off" >&2
    return 1
  fi
  if [ ! -s "$dest" ]; then
    echo "REFUSED: the serve's /metrics on port ${port} answered 200 with an" \
         "empty body; there is nothing to check" >&2
    return 1
  fi
  return 0
}

# serve_require_no_spec_decode PORT DEST
#   0 when the serve published a complete metrics response with no speculative
#   markers in it.  DEST keeps that response beside the run as the evidence the
#   gate read.
serve_require_no_spec_decode() {
  local port="$1" dest="$2"
  serve_metrics_fetch "$port" "$dest" || return 1
  # The FILE, never a pipe: the whole response is on disk before it is read,
  # so an early match cannot break the producer that is still writing.
  if grep -q "$SERVE_SPEC_DECODE_MARKER" "$dest"; then
    echo "REFUSED: serve has spec-decode active; the logprobs would be the" \
         "draft model's (${SERVE_SPEC_DECODE_MARKER} in $dest)" >&2
    return 1
  fi
  return 0
}
