"""Reconcile a completed build's Ninja log without relinking retained bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def binding(path):
    before = path.stat()
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    after = path.stat()
    if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
        raise RuntimeError(f'artifact changed while hashing: {path}')
    return dict(sha256=digest, bytes=after.st_size, mtime_ns=after.st_mtime_ns)


def finalize(directory, source):
    from tessera.jit_build_lock import jit_build_lock

    directory, source = Path(directory), Path(source)
    with jit_build_lock(directory):
        files = [source, directory / 'build.ninja', *sorted(directory.glob('*.o')),
                 *sorted(directory.glob('*.so'))]
        libraries = list(directory.glob('*.so'))
        if len(libraries) != 1 or not list(directory.glob('*.o')):
            raise RuntimeError('finalization requires one completed library and its objects')
        before = {str(path): binding(path) for path in files}

        def ninja(*args):
            return subprocess.run(['ninja', '-C', str(directory), *args],
                                  check=True, capture_output=True, text=True)

        dry = ninja('-n', '-d', 'explain')
        planned = [re.sub(r'^\[\d+/\d+\] ', '', line) for line in dry.stdout.splitlines()
                   if re.match(r'^\[\d+/\d+\] ', line)]
        old_log = (directory / '.ninja_log').read_bytes()
        old_digest = hashlib.sha256(old_log).hexdigest()
        if planned:
            # Reconcile only the observed link-only timestamp condition. Any
            # compile, changed command, missing dependency or ambiguity refuses.
            commands = ninja('-t', 'commands', libraries[0].name).stdout.splitlines()
            explanation = dry.stderr
            if (len(planned) != 1 or not commands or planned != commands[-1:]
                    or ' -shared ' not in planned[0]
                    or f'recorded mtime of {libraries[0].name} older than most recent input' not in explanation
                    or 'command line changed' in explanation):
                raise RuntimeError('unqualified pending native work: ' + dry.stdout + dry.stderr)
            saved = directory / f'native-finalization-before-{old_digest}.log'
            if not saved.exists():
                with saved.open('xb') as handle:
                    handle.write(old_log)
            # Supported Ninja operation: refresh log records from the actual
            # completed outputs. It does not touch source/object/ELF timestamps.
            ninja('-t', 'restat')
        after_dry = ninja('-n', '-d', 'explain')
        if 'ninja: no work to do.' not in after_dry.stdout:
            raise RuntimeError('native build remains dirty: ' + after_dry.stdout + after_dry.stderr)
        after = {str(path): binding(path) for path in files}
        if before != after:
            raise RuntimeError('native bytes or timestamps changed during log finalization')
        receipt = dict(schema='tessera.native_build_finalization.v1',
                       bindings=after, log_before_sha256=old_digest,
                       log_after_sha256=hashlib.sha256((directory / '.ninja_log').read_bytes()).hexdigest(),
                       before_explanation=dry.stdout + dry.stderr,
                       after_explanation=after_dry.stdout + after_dry.stderr,
                       reconciled=bool(planned), no_pending_work=True)
        (directory / 'native-finalization.json').write_text(json.dumps(receipt, indent=2) + '\n')
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    for directory in args.directories:
        print(json.dumps(finalize(directory, args.source)))


if __name__ == '__main__':
    main()
