"""A fresh extension directory is a valid pre-build measurement state."""
from pathlib import Path
import os
import subprocess

import pytest


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


def wrapper_environment(tmp_path):
    checkout = tmp_path / 'checkout'
    (checkout / 'experiments').mkdir(parents=True)
    (checkout / 'experiments/runtime_image.sh').write_text(
        'runtime_image_require() { RUNTIME_IMAGE_CONTAINER_ENV=""; RUNTIME_IMAGE_JSON="{}"; }\n')
    (checkout / 'pyproject.toml').write_text('[project]\nname="tessera-quant"\nversion="0.0.0"\n')
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
    env = {k: v for k, v in os.environ.items() if not k.startswith(('BENCH_', 'NATIVE_', 'TEST_'))}
    env.update(PATH=str(binaries) + os.pathsep + os.environ['PATH'],
               ORACLE_IMAGE='inert-image', TEST_RUNNER_SP=str(packages),
               BENCH_SRC=str(source), BENCH_EXT_DIR=str(extensions), TEST_XDIST='1',
               DOCKER_ARGV_PATH=str(argv_path), PRISMABUILD_ACTION_KEY="0" * 64)
    return checkout, source, extensions, argv_path, env


@pytest.mark.parametrize('canonical', [False, True])
def test_gpu_test_wrapper_reuses_native_namespace_and_xdist(tmp_path, canonical):
    wrapper = Path(__file__).resolve().parents[1] / 'experiments/routed_fused_tests.sh'
    checkout, source, extensions, argv_path, env = wrapper_environment(tmp_path)
    container_source = '/tessera/src' if canonical else '/work/src'
    container_extensions = '/ext' if canonical else str(extensions)
    if canonical:
        env.update(NATIVE_CONTAINER_SRC=container_source, NATIVE_CONTAINER_EXT=container_extensions)
    subprocess.run(['bash', str(wrapper), str(checkout), str(tmp_path / 'out'),
                    '-n', '2', '--dist', 'worksteal', 'owned-case'],
                   env=env, check=True, capture_output=True, text=True)
    args = argv_path.read_bytes().decode().rstrip('\0').split('\0')
    assert str(source) + ':' + container_source + ':ro' in args
    assert str(extensions) + ':' + container_extensions in args
    assert 'TORCH_EXTENSIONS_DIR=' + container_extensions in args
    assert 'NATIVE_CONTAINER_SRC=' + container_source in args
    if canonical:
        assert 'native_test_source' in args
        assert str(checkout / 'pyproject.toml') + ':/tessera/pyproject.toml:ro' in args
    assert 'xdist.plugin' in args
    assert args[-5:] == ['-n', '2', '--dist', 'worksteal', 'owned-case']


@pytest.mark.parametrize('canonical', [False, True])
def test_build_wrapper_matches_the_consumer_namespace(tmp_path, canonical):
    wrapper = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/build_ext.sh'
    checkout, source, extensions, argv_path, env = wrapper_environment(tmp_path)
    container_source = '/tessera/src' if canonical else '/work/src'
    container_extensions = '/ext' if canonical else str(extensions)
    work = tmp_path / 'owned-library-work'
    env['NATIVE_BUILD_WORK_DIR'] = str(work)
    if canonical:
        env.update(NATIVE_CONTAINER_SRC=container_source, NATIVE_CONTAINER_EXT=container_extensions)
    subprocess.run(['bash', str(wrapper), str(checkout), str(extensions), 'e4m3mma'],
                   env=env, check=True, capture_output=True, text=True)
    args = argv_path.read_bytes().decode().rstrip('\0').split('\0')
    assert str(source) + ':' + container_source + ':ro' in args
    assert str(extensions) + ':' + container_extensions in args
    assert 'TORCH_EXTENSIONS_DIR=' + container_extensions in args
    assert 'PYTHONPATH=' + container_source in args
    assert 'HOME=' + str(work / 'home') in args
    assert 'TMPDIR=' + str(work / 'tmp') in args
    if canonical:
        assert str(checkout / 'pyproject.toml') + ':/tessera/pyproject.toml:ro' in args


@pytest.mark.parametrize('selection', ['0', '1'])
@pytest.mark.parametrize('canonical', [False, True])
def test_actual_wrapper_forwards_resident_layout_selection(tmp_path, selection, canonical):
    """Execute the real wrapper; image inspection and Docker only are inert."""
    wrapper = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/bench_t8r.sh'
    checkout, source, extensions, argv_path, env = wrapper_environment(tmp_path)
    artifact = tmp_path / 'artifact'
    artifact.mkdir()
    (artifact / 'config.json').write_text('{}')
    extensions.mkdir()
    env['TESSERA_ROUTED_PIECE_MAJOR'] = selection
    container_source = '/tessera/src' if canonical else '/work/src'
    container_extensions = '/ext' if canonical else str(extensions)
    if canonical:
        env.update(NATIVE_CONTAINER_SRC=container_source, NATIVE_CONTAINER_EXT=container_extensions)
    subprocess.run(['bash', str(wrapper), str(checkout), str(tmp_path / 'out'),
                    '--artifact', str(artifact), '--groups', 'experts.R1024.L10'],
                   env=env, check=True, capture_output=True, text=True)
    args = argv_path.read_bytes().decode().rstrip('\0').split('\0')
    forwarded = [args[i + 1] for i, value in enumerate(args[:-1]) if value == '-e']
    assert 'TESSERA_ROUTED_PIECE_MAJOR=' + selection in forwarded
    assert str(source) + ':' + container_source + ':ro' in args
    assert str(extensions) + ':' + container_extensions in args
    assert 'TORCH_EXTENSIONS_DIR=' + container_extensions in forwarded
    assert 'PYTHONPATH=' + container_source + ':/work/tests' in forwarded
    if canonical:
        assert str(checkout / 'pyproject.toml') + ':/tessera/pyproject.toml:ro' in args


@pytest.mark.parametrize('phase,deadline', [('numeric', '240s'), ('timing', '600s'), ('repeatability', '600s')])
def test_direct_pm_numeric_reuses_owned_container_and_canonical_namespace(tmp_path, phase, deadline):
    wrapper = Path(__file__).resolve().parents[1] / 'experiments/t8r_speed/bench_t8r.sh'
    checkout, source, extensions, argv_path, env = wrapper_environment(tmp_path)
    artifact = tmp_path / 'artifact'; artifact.mkdir()
    (artifact / 'config.json').write_text('{}')
    extensions.mkdir()
    sdk = tmp_path / 'published'; (sdk / 'src/prismabuild').mkdir(parents=True)
    timeout = Path(env['PATH'].split(os.pathsep)[0]) / 'timeout'
    timeout.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$TIMEOUT_ARGV_PATH"\nshift 3\nexec "$@"\n')
    timeout.chmod(0o755)
    env['TIMEOUT_ARGV_PATH'] = str(tmp_path / 'timeout-argv')
    env.update(BENCH_DIRECT_VLLM='1', BENCH_OWNER_TOKEN='e' * 32,
               PB_CLIENT_ROOT=str(sdk), BENCH_FIXTURE_RUNNER_SP=env['TEST_RUNNER_SP'],
               NATIVE_CONTAINER_SRC='/tessera/src',
               NATIVE_CONTAINER_EXT='/ext')
    subprocess.run(['bash', str(wrapper), str(checkout), str(tmp_path / 'out'),
                    '--artifact', str(artifact), '--comparison-protocol', '/owned/protocol.json',
                    '--comparison-phase', phase],
                   env=env, check=True, capture_output=True, text=True)
    args = argv_path.read_bytes().decode().rstrip('\0').split('\0')
    timeout_args = Path(env['TIMEOUT_ARGV_PATH']).read_bytes().decode().rstrip('\0').split('\0')
    assert timeout_args[:3] == ['--signal=TERM', '--kill-after=15s', deadline]
    assert args[args.index('--label') + 1] == 'tessera.paired_numeric_owner=' + 'e' * 32
    assert args[args.index('--memory') + 1] == args[args.index('--memory-swap') + 1] == '16g'
    assert args[args.index('--cpus') + 1] == '2'
    assert str(sdk) + ':' + str(sdk) + ':ro' in args
    assert 'PYTHONPATH=/tessera/src:/work/tests:' + str(sdk) + '/src:' + env['TEST_RUNNER_SP'] in args
    assert env['TEST_RUNNER_SP'] + ':' + env['TEST_RUNNER_SP'] + ':ro' in args
    assert (tmp_path / 'out/owner-token.txt').read_text().strip() == 'e' * 32


@pytest.mark.parametrize("key,on,library", [
    ("TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH", "1", "e4m3mma"),
    ("TESSERA_ROUTED_FUSED_VALUE_A_PREFETCH", "4", "value"),
    ("TESSERA_ROUTED_FUSED_FP4_A_PREFETCH", "4", "e2m1")])
@pytest.mark.parametrize('kind', ['build', 'benchmark'])
@pytest.mark.parametrize('selection', [None, '', '0', 'on', 'bad'],
                         ids=['unset', 'empty', 'off', 'on', 'invalid'])
def test_native_build_choice_wrapper_preserves_declared_value(tmp_path, kind, selection, key, on, library):
    """Real shell paths preserve strict Python choice; only Docker/image are inert."""
    if selection == "on":
        selection = on
    root = Path(__file__).resolve().parents[1]
    checkout, source, extensions, argv_path, env = wrapper_environment(tmp_path)
    env.pop(key, None)
    if selection is not None:
        env[key] = selection
    if kind == 'build':
        wrapper = root / 'experiments/t8r_speed/build_ext.sh'
        command = ['bash', str(wrapper), str(checkout), str(extensions), library]
    else:
        wrapper = root / 'experiments/t8r_speed/bench_t8r.sh'
        artifact = tmp_path / 'artifact'; artifact.mkdir()
        (artifact / 'config.json').write_text('{}')
        extensions.mkdir()
        command = ['bash', str(wrapper), str(checkout), str(tmp_path / 'out'),
                   '--artifact', str(artifact), '--groups', 'experts.R1024.L10']
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    args = argv_path.read_bytes().decode().rstrip('\0').split('\0')
    forwarded = [args[i + 1] for i, arg in enumerate(args[:-1]) if arg == '-e']
    choices = [value for value in forwarded if value.startswith(key + '=')]
    if selection is None:
        assert choices == ([key + '=0'] if kind == 'build' else [])
    else:
        assert choices == [key + '=' + selection]
