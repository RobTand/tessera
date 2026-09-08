#!/bin/sh
# pbtest owns sharding; only this pytest invocation receives these options.
export MAX_JOBS=1
exec /home/rob/venvs/pb-cpu/bin/python "$@" --dist=worksteal --durations=10
