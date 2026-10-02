"""A fresh extension directory is a valid pre-build measurement state."""
from pathlib import Path
import os
import subprocess


def extension_listing(directory):
    path=Path(__file__).resolve().parents[1]/'experiments/t8r_speed/bench_t8r.sh'
    source=path.read_text()
    start=source.index('ext_libs() {')
    end=source.index('\nEXT_BEFORE=',start)
    command=source[start:end]+'\nEXT_DIR=$1\nresult=$(ext_libs)\nprintf "%s" "$result"\n'
    return subprocess.run(['bash','-euo','pipefail','-c',command,'listing',str(directory)],
                          capture_output=True,text=True,check=True).stdout


def test_fresh_extension_directory_is_valid(tmp_path):
    assert extension_listing(tmp_path)==''


def test_existing_extension_identity_stays_visible(tmp_path):
    directory=tmp_path/'tessera_native';directory.mkdir()
    library=directory/'fixture.so';library.write_bytes(b'not-a-build-probe')
    result=extension_listing(tmp_path)
    timestamp,name=result.strip().split(maxsplit=1)
    assert int(timestamp)==int(library.stat().st_mtime)
    assert name.endswith('tessera_native/fixture.so')


def test_gpu_test_wrapper_reuses_native_namespace_and_xdist(tmp_path):
    wrapper = Path(__file__).resolve().parents[1] / 'experiments/routed_fused_tests.sh'
    checkout = tmp_path / 'checkout'
    (checkout / 'experiments').mkdir(parents=True)
    (checkout / 'experiments/runtime_image.sh').write_text(
        'runtime_image_require() { RUNTIME_IMAGE_CONTAINER_ENV=""; }\n')
    source = tmp_path / 'frozen-src'
    kernel = source / 'tessera/serving/csrc/routed_fused_window.cu'
    kernel.parent.mkdir(parents=True)
    kernel.write_text('// reviewed frozen source\n')
    extensions = tmp_path / 'retained-extensions'
    packages = tmp_path / 'packages'
    packages.mkdir()
    for name in ('pytest', '_pytest', 'pluggy', 'iniconfig', 'packaging', 'xdist', 'execnet'):
        (packages / name).mkdir()
    (packages / 'py.py').write_text('')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    docker = binaries / 'docker'
    docker.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$DOCKER_ARGV_PATH"\n')
    docker.chmod(0o755)
    argv_path = tmp_path / 'argv'
    env = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               ORACLE_IMAGE='inert-image', TEST_RUNNER_SP=str(packages),
               BENCH_SRC=str(source), BENCH_EXT_DIR=str(extensions), TEST_XDIST='1',
               DOCKER_ARGV_PATH=str(argv_path))
    subprocess.run(['bash', str(wrapper), str(checkout), str(tmp_path / 'out'),
                    '-n', '2', '--dist', 'worksteal', 'owned-case'],
                   env=env, check=True, capture_output=True, text=True)
    args = argv_path.read_bytes().decode().rstrip('\0').split('\0')
    assert str(source) + ':/work/src:ro' in args
    assert str(extensions) + ':' + str(extensions) in args
    assert 'TORCH_EXTENSIONS_DIR=' + str(extensions) in args
    assert 'xdist.plugin' in args
    assert args[-5:] == ['-n', '2', '--dist', 'worksteal', 'owned-case']
