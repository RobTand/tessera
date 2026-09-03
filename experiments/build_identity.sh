#!/usr/bin/env bash
# Stamp WHICH COMPILED BUILD served an arm, beside that arm's KL dump.
#
# Sourced by every wrapper that starts its own serve and writes a dump.  Two
# functions, no side effects at source time.
#
# WHY (issue #30).  A compiled vLLM artifact replayed is bit-identical
# (0.000000 / 100%); the same graph rebuilt is not (0.017117 / 95.65%, 120 of
# 196 autotuned Triton kernels retuned) -- docs/measurements/serving-compile-
# divergence-2026-09-02.md.  That is the size of an ordinary result on this
# box, so until an arm records its build, a rebuild and a regression are the
# same number.  The twelve-arm chain of 2026-09-02 shared one cache root by
# convention and still needed a forensic session over two surviving cache
# directories to attribute its one odd arm.
#
# Note what is NOT stamped: the AOT key alone.  vLLM keys its cache by its own
# inputs, and both of the divergent builds above sit under one key with a
# byte-identical cache_key_factors.json, so a key-matched stamp would certify a
# rebuild as a replay.  The stamper digests the CONTENT of the cache slot; see
# src/tessera/serving/build_identity.py.  A serve with no cache-root mount
# stamps `complete: false` and refuses to certify anything either way.
#
#   build_identity_docker_env       -> the -e flags to splice into `docker run`
#   build_identity_stamp LOG OUT [CACHE_ROOT] [IMAGE] [SERVE_MODE] [EAGER] [ARTIFACT]
#
# TESSERA_SERVE_DETERMINISTIC=1 forwards TORCHINDUCTOR_DETERMINISTIC=1 into the
# container (torch/_inductor/config.py: "skips any on device benchmarking in
# Inductor if we know they affect numerics.  WARNING: Expect perf hit") and
# stamps the flag into the identity, so a campaign cannot silently mix arms
# built with it and without.  Whether two builds under the flag actually agree
# is unmeasured; setting it on a warm cache changes nothing at all, and the
# stamp records the `fresh_compiles: 0` that says so.

# Resolve from THIS file, not from a caller's $TS: a worktree's wrapper must
# stamp with its own checkout's stamper, not the main checkout's.
_BUILD_IDENTITY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_IDENTITY_PY="${BUILD_IDENTITY_PY:-python3}"

build_identity_docker_env() {
  if [ "${TESSERA_SERVE_DETERMINISTIC:-0}" = "1" ]; then
    printf -- '-e TORCHINDUCTOR_DETERMINISTIC=1'
  fi
}

build_identity_stamp() {
  local log="$1" out="$2" cache_root="${3:-}" image="${4:-}" mode="${5:-}" \
        eager="${6:-}" artifact="${7:-}"
  local args=(--log "$log" --out "$out"
              --deterministic "${TESSERA_SERVE_DETERMINISTIC:-0}")
  [ -n "$cache_root" ] && args+=(--cache-root "$cache_root")
  [ -n "$image" ] && args+=(--image "$image")
  [ -n "$mode" ] && args+=(--serve-mode "$mode")
  [ -n "$eager" ] && args+=(--eager "$eager")
  [ -n "$artifact" ] && args+=(--artifact-path "$artifact")
  # Never fatal: the stamp is provenance, and a wrapper that dies here would
  # lose the dump it just paid a serve for.  A missing sidecar is loud on the
  # reading side -- serving_compile_divergence.py reports it as a problem.
  PYTHONPATH="$_BUILD_IDENTITY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$BUILD_IDENTITY_PY" -m tessera.serving.build_identity stamp "${args[@]}" \
    || echo "WARNING: build identity not stamped for $out"
}
