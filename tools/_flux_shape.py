"""Pure resource and environment grammar shared by Flux command renderers."""
from __future__ import annotations

import math
import re

THREADS = dict.fromkeys(("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                         "TORCH_NUM_THREADS", "NUMEXPR_NUM_THREADS", "MAX_JOBS",
                         "CMAKE_BUILD_PARALLEL_LEVEL"), "1")
THREADS.update(MAKEFLAGS="-j1", NINJAFLAGS="-j1")


def duration_seconds(value):
    """Parse a positive, finite Flux duration with an explicit unit."""
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?[smhd]", value) is None:
        raise ValueError("walltime requires a positive explicit-unit duration, e.g. 10m")
    seconds = float(value[:-1]) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[value[-1]]
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("walltime must be positive and finite")
    return seconds


def resource_shape(cores, memory, gpus=0):
    """The stock slot shape; any core class satisfies it, so none is named."""
    return f"slot/[core={cores};memory={memory}" + (";gpu=1]" if gpus else "]")
