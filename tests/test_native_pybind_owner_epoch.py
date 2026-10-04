"""Real CPU PyBind import epochs; not CUDA/native-serving qualification."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import pytest
import torch

PROBE = r'''
import importlib.machinery, importlib.util, json, os, sys
name, path, mode = sys.argv[1:]
fds, objects, rows = [], [], []
def load():
    fd = os.open(path, os.O_RDONLY); fds.append(fd)
    origin = '/proc/self/fd/' + str(fd)
    loader = importlib.machinery.ExtensionFileLoader(name, origin)
    spec = importlib.util.spec_from_file_location(name, origin, loader=loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    loader.exec_module(module)
    objects.append(module)
    return module, origin
try:
    if mode == 'once':
        module, origin = load()
    for epoch in range(3):
        if mode == 'reload':
            module, origin = load()
        rows.append({'epoch': epoch, 'origin': origin, 'file': module.__file__,
                     'spec': module.__spec__.origin, 'identity': module.identity(),
                     'bound': module.__file__ == origin and module.__spec__.origin == origin})
        if mode == 'reload':
            del sys.modules[name]
    print(json.dumps({'mode': mode, 'loads': len(fds), 'rows': rows}))
finally:
    for fd in fds: os.close(fd)
'''


@pytest.fixture(scope="module")
def pybind_probe_library(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("real-pybind-epoch")
    compiler = shutil.which('c++')
    assert compiler, 'real CPU PyBind regression requires C++ compiler'
    include = Path(torch.__file__).parent / 'include'
    assert (include / 'pybind11/pybind11.h').is_file(), 'missing actual Torch PyBind headers'
    name = 'native_epoch_probe'
    source = tmp_path / 'probe.cpp'
    library = tmp_path / (name + '.so')
    source.write_text('#include <pybind11/pybind11.h>\n'
                      'PYBIND11_MODULE(native_epoch_probe, m) { m.def("identity", [] { return 4; }); }\n')
    subprocess.run([compiler, '-shared', '-fPIC', '-std=c++17', '-O2',
                    '-I' + str(include), '-I' + sysconfig.get_paths()['include'],
                    str(source), '-o', str(library)], check=True)
    return library


@pytest.mark.parametrize("mode", ["reload", "once"])
def test_real_pybind_three_epochs_require_one_native_owner_lifetime(pybind_probe_library, mode):
    name, library = "native_epoch_probe", pybind_probe_library
    completed = subprocess.run([sys.executable, "-c", PROBE, name, str(library), mode],
                               check=True, capture_output=True, text=True)
    observed = json.loads(completed.stdout)
    print('REAL_CPU_PYBIND_EPOCHS ' + completed.stdout.strip())
    assert [row['identity'] for row in observed['rows']] == [4, 4, 4]
    assert observed['rows'][0]['bound']
    if mode == 'reload':
        assert observed['loads'] == 3
        assert any(not row['bound'] for row in observed['rows'][1:])
    else:
        assert observed['loads'] == 1
        assert all(row['bound'] for row in observed['rows'])
