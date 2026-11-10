#!/usr/bin/env bash
# Root Docker is permitted in Flux only by the reviewed serving contract.
refuse_flux_docker() {
  if [[ ${FLUX_JOB_ID+x} ]]; then
    echo 'FLUX_JOB_ID: root Docker inside a Flux job is allowed only through the reviewed serving wrapper; send every other use to the lead.' >&2
    return 2
  fi
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  refuse_flux_docker
fi
