"""Match the actually observed Nsight name, refusing other native modes."""
from pathlib import Path
import re
import shlex

import pytest

# Available Kernels in retained actual PB2726 profile/ncu.csv. Nsight's
# demangling prints explicit typed constants, unlike torch.profiler's spelling.
ACTUAL = 'void <unnamed>::routed_fused_kernel<(bool)1, (int)0, (bool)0, (bool)0, (int)4, (bool)0, (int)128>(<unnamed>::Params)'


def pattern():
    path=Path(__file__).resolve().parents[1]/'experiments/t8r_speed/routed_gate_diagnosis.sh'
    line=next(l for l in path.read_text().splitlines() if l.startswith('export BENCH_NCU_KERNELS='))
    return shlex.split(line[len('export BENCH_NCU_KERNELS='):])[0]


def test_exact_observed_ncu_mode0_matches():
    assert re.search(pattern(),ACTUAL)


@pytest.mark.parametrize('old,new', [('(int)0','(int)1'),('(int)0','(int)2'),
    ('(int)4','(int)5'),('(int)128','(int)64'),('(bool)1','(bool)0')])
def test_other_native_specializations_do_not_match(old,new):
    assert re.search(pattern(),ACTUAL.replace(old,new,1)) is None
