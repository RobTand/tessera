"""Real CPU dlopen before/after regression; no CUDA execution (#874)."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

PROBE = r'''
import ctypes, json, os, sys
old, new, hold = sys.argv[1], sys.argv[2], sys.argv[3] == 'hold'
a = os.open(old, os.O_RDONLY)
first = ctypes.CDLL('/proc/self/fd/' + str(a))
assert first.identity() == 0
if not hold:
    os.close(a)
b = os.open(new, os.O_RDONLY)
second = ctypes.CDLL('/proc/self/fd/' + str(b))
print(json.dumps({'held': hold, 'first_fd': a, 'second_fd': b,
                  'first': first.identity(), 'second': second.identity()}))
if hold:
    os.close(a)
os.close(b)
'''


@pytest.mark.parametrize('hold', [False, True], ids=['before-reused-path', 'after-held-distinct-paths'])
def test_real_dlopen_binary_selection_changes_with_fd_lifetime(tmp_path, hold):
    compiler = shutil.which('cc')
    assert compiler, 'CPU qualification requires a real C compiler'
    libraries = []
    for number in (0, 4):
        source = tmp_path / f'arm{number}.c'
        library = tmp_path / f'arm{number}.so'
        source.write_text(f'int identity(void) {{ return {number}; }}\n')
        subprocess.run([compiler, '-shared', '-fPIC', str(source), '-o', str(library)], check=True)
        libraries.append(str(library))
    result = subprocess.run([sys.executable, '-c', PROBE, *libraries, 'hold' if hold else 'reuse'],
                            check=True, text=True, capture_output=True)
    observed = json.loads(result.stdout)
    print('REAL_CPU_DLOPEN ' + result.stdout.strip())
    assert observed['first'] == 0
    if hold:
        assert observed['first_fd'] != observed['second_fd']
        assert observed['second'] == 4
    else:
        assert observed['first_fd'] == observed['second_fd']
        # Demonstrates the formerly plausible wrong-binary selection, not a CUDA repro.
        assert observed['second'] == 0
