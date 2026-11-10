#!/bin/sh
# The caller owns sharding; only this pytest invocation receives these options.
export MAX_JOBS=1
exec "${TESSERA_CPU_PYTHON:?set TESSERA_CPU_PYTHON to the scoped CPU interpreter}" "$@" --dist=worksteal --durations=10
