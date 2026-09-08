#!/bin/sh
# pbtest owns sharding; this interpreter entrypoint supplies the suite options.
export PYTEST_ADDOPTS="--dist=worksteal --durations=10"
export MAX_JOBS=1
exec /home/rob/venvs/pb-cpu/bin/python "$@"
