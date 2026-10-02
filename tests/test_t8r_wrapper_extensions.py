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


import pytest


@pytest.mark.parametrize('choice', [None, '1'])
@pytest.mark.parametrize('direct',[False,True])
def test_bench_wrapper_retains_paired_build_choice(tmp_path, choice,direct):
    wrapper = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/bench_t8r.sh'
    checkout = tmp_path/'checkout'
    (checkout/'experiments').mkdir(parents=True)
    (checkout/'experiments/runtime_image.sh').write_text(
        'runtime_image_require() { RUNTIME_IMAGE_CONTAINER_ENV=""; }\n')
    kernel=checkout/'src/tessera/serving/csrc/routed_fused_window.cu'
    kernel.parent.mkdir(parents=True)
    kernel.write_text('// frozen source\n')
    artifact=tmp_path/'artifact'
    artifact.mkdir()
    (artifact/'config.json').write_text('{}')
    binaries=tmp_path/'bin'
    binaries.mkdir()
    docker=binaries/'docker'
    docker.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$DOCKER_ARGV_PATH"\n')
    docker.chmod(0o755)
    argv_path=tmp_path/'argv'
    env=dict(os.environ,PATH=str(binaries)+os.pathsep+os.environ['PATH'],
             ORACLE_IMAGE='inert-image',DOCKER_ARGV_PATH=str(argv_path))
    env.pop('TESSERA_ROUTED_FUSED_PAIRED_K32',None)
    if choice is not None:
        env['TESSERA_ROUTED_FUSED_PAIRED_K32']=choice
    extra=[]
    if direct:
        sdk=tmp_path/'published-sdk';(sdk/'src/prismabuild').mkdir(parents=True)
        env.update(BENCH_DIRECT_VLLM='1',BENCH_OWNER_TOKEN='a'*32,PB_CLIENT_ROOT=str(sdk))
        env.pop('BENCH_STRICT_STAGED',None)
        extra=['--paired-k32-numerics','--direct-vllm-inputs']
    subprocess.run(['bash',str(wrapper),str(checkout),str(tmp_path/'out'),
                    '--artifact',str(artifact),*extra],env=env,check=True,capture_output=True,text=True)
    args=argv_path.read_bytes().decode().rstrip('\0').split('\0')
    values=[arg for arg in args if arg.startswith('TESSERA_ROUTED_FUSED_PAIRED_K32=')]
    assert values == ([] if choice is None else ['TESSERA_ROUTED_FUSED_PAIRED_K32='+choice])
    if direct:
        for key,value in [('--memory','16g'),('--memory-swap','16g'),('--pids-limit','512'),('--cpus','2')]:
            assert args[args.index(key)+1]==value
        assert args[args.index('--cidfile')+1]==str(tmp_path/'out/owned.cid')
        assert 'tessera.paired_numeric_owner='+'a'*32 in args
        assert str(sdk)+':'+str(sdk)+':ro' in args
        assert not any(x.startswith('PRISMABUILD_') for x in args)
