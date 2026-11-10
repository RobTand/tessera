#!/usr/bin/env bash
# This branch has no reviewed serving wrapper or root-Docker exemption.
refuse_flux_docker() {
  if [[ ${FLUX_JOB_ID+x} ]]; then
    echo 'FLUX_JOB_ID: root Docker inside a Flux job is refused; no reviewed serving wrapper exists on this branch. Send this use to the lead.' >&2
    return 2
  fi
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  refuse_flux_docker
fi
