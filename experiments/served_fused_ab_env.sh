#!/usr/bin/env bash
# Arm-env decisions for the #545 served A/B, in one sourced place so the
# driver wrapper, the action script and the CPU tests read ONE definition.
#
# Historical defect (parent review of 2026-10-04, finding 4): the wrapper
# spelled ``GLM=0`` and expanded ``${GLM:+-e ...}`` -- and ``:+`` tests
# NONEMPTY, not TRUE, so the value "0" exported TESSERA_RESEARCH_GLM53_NOPE=1
# into the h1 arms too.  These helpers make off/on explicit and testable.

# served_fused_ab_glm_env GLM_FLAG -> prints one "-e KEY=VALUE" per line
served_fused_ab_glm_env() {
  if [ "${1:-}" = "1" ]; then
    printf '%s\n' "-e TESSERA_RESEARCH_GLM53_NOPE=1"
  fi
}

# served_fused_ab_profiler_env DIR -> prints the in-engine profiler env
served_fused_ab_profiler_env() {
  printf '%s\n' "-e VLLM_TORCH_PROFILER_DIR=$1"
}
