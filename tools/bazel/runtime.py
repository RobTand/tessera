"""Give a remote test the environment the hosted gate gives a shard."""
import os
from pathlib import Path
import sys
import tempfile
import venv

# .github/workflows/gate.yml and tools/ci/gate_run.py set these for every shard.
GATE_ENVIRONMENT = {
    'CUDA_VISIBLE_DEVICES': '',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'GIT_AUTHOR_NAME': 'Tessera CI',
    'GIT_AUTHOR_EMAIL': 'ci@example.invalid',
    'GIT_COMMITTER_NAME': 'Tessera CI',
    'GIT_COMMITTER_EMAIL': 'ci@example.invalid',
    # A remote test has no .git. Exports stamp the commit they came from
    # (tessera.serving_parts.git_hash), and this is how a caller without git
    # supplies one. A fixed value keeps results cacheable across commits.
    'TESSERA_GIT': 'bazel-remote-test',
}
# What Bazel's test runner adds. A hosted shard has none of it.
BAZEL_PREFIXES = ('TEST_', 'RUNFILES_', 'BASH_FUNC_', 'GTEST_', 'COVERAGE_', 'TESTBRIDGE_', 'LD_', 'TS_BAZEL_')
BAZEL_NAMES = {'JAVA_RUNFILES', 'PYTHON_RUNFILES', 'XML_OUTPUT_FILE', 'BAZEL_TEST', 'PYTHONSAFEPATH',
               'RUNTEST_PRESERVE_CWD', 'TZ', 'BUILD_EXECROOT'}


def compilers(cc):
    """CC and CXX from the cc toolchain's C compiler: torch's cpp_extension and
    Inductor run CXX, and the C++ driver sits beside the C one."""
    if not cc:
        return {}
    path = Path(cc)
    cxx = path.with_name(path.name.replace('gcc', 'g++').replace('clang', 'clang++')) \
        if path.name.endswith(('gcc', 'clang')) else path.with_name('c++')
    return {'CC': str(path), 'CXX': str(cxx)}


def gate_environment(environ, root, scratch):
    """The pytest process's environment: the caller's minus Bazel's, plus the gate's.

    TMPDIR and HOME are short and on the execroot's filesystem: Bazel's
    TEST_TMPDIR is too long for the AF_UNIX sockets tests bind, and torch's
    build directories must allow execution.
    """
    environment = {name: value for name, value in environ.items()
                   if not name.startswith(BAZEL_PREFIXES) and name not in BAZEL_NAMES}
    environment.update(GATE_ENVIRONMENT)
    environment.update(compilers(environ.get('TS_BAZEL_CC')))
    environment['TMPDIR'] = environment['HOME'] = str(scratch)
    # gate_run puts tools/ci first; the gate workflow adds src.
    environment['PYTHONPATH'] = os.pathsep.join([str(root / 'tools/ci'), str(root / 'src')])
    return environment


def short_scratch(private):
    """A new directory as near the root as possible, on TEST_TMPDIR's filesystem."""
    device, base = private.stat().st_dev, private
    for parent in private.parents:
        if parent.stat().st_dev != device or not os.access(parent, os.W_OK | os.X_OK):
            break
        base = parent
    return Path(tempfile.mkdtemp(prefix='t', dir=base))


def checkout_of(runfiles):
    """The runner's git checkout: runfiles of a local, unsandboxed test link into it."""
    root = (runfiles / 'pyproject.toml').resolve().parent
    if not (root / '.git').exists():
        raise RuntimeError(f'runner-lane test needs a git checkout; {root} is not one. '
                           'Run it on a Remote Bazel runner (docs/bazel.md).')
    return root


def prepare_remote_test(root):
    """Short scratch and an interpreter whose children keep Bazel's packages.

    Returns the scratch directory and the environment for pytest.
    """
    scratch = short_scratch(Path(os.environ['TEST_TMPDIR']).resolve())
    # Python's -I ignores PYTHONPATH, and tests start children with their own
    # environment. A real venv with a .pth keeps declared Bazel dependencies.
    environment = Path(os.environ['TEST_TMPDIR']).resolve() / 'python-environment'
    if not (environment / 'pyvenv.cfg').exists():
        venv.EnvBuilder(with_pip=False, symlinks=True).create(environment)
    site = next(environment.glob('lib/python*/site-packages'))
    paths = [str(Path(path).resolve()) for path in sys.path if path]
    (site / 'bazel-runfiles.pth').write_text('\n'.join(paths) + '\n')
    sys.executable = str(environment / 'bin/python')
    variables = gate_environment(os.environ, root, scratch)
    # Tests run python3 and python from PATH.
    variables['PATH'] = os.pathsep.join([str(environment / 'bin'), os.environ.get('PATH', '')])
    return scratch, variables
