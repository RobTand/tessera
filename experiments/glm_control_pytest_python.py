#!/usr/bin/python3
"""CPU interpreter adapter preserving the repository's xdist test mode.

The caller supplies the test partition and Python arguments. This versioned adapter
only adds the required worksteal/duration options and selects the scoped CPU
interpreter; it neither partitions tests nor executes outside a Flux job.
"""
import os
import sys

if __name__ == '__main__':
    os.environ['PYTEST_ADDOPTS'] = '--dist worksteal --durations=10'
    interpreter = '/home/rob/venvs/pb-cpu/bin/python'
    os.execv(interpreter, [interpreter, *sys.argv[1:]])
