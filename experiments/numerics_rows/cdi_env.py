"""Apply CDI environment edits to a CPU container without GPU devices or hooks."""
import argparse
import json
import os
from pathlib import Path
import re
import sys

import yaml

SPECS = (Path('/etc/cdi/nvidia-podman-compat.yaml'), Path('/var/run/cdi/nvidia.yaml'))


def read_environment(paths=SPECS):
    records = []
    try:
        for path in paths:
            path = Path(path)
            if not path.is_file():
                continue
            document = yaml.safe_load(path.read_text())
            if document['kind'] != 'nvidia.com/gpu':
                raise ValueError()
            device, = [d for d in document['devices'] if d['name'] == 'all']
            environment = {}
            for edits in (document.get('containerEdits', {}), device.get('containerEdits', {})):
                for item in edits.get('env', []):
                    name, value = item.split('=', 1)
                    if not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', name) or any(c in value for c in '\r\n\0'):
                        raise ValueError()
                    if name in environment and environment[name] != value:
                        raise ValueError()
                    environment[name] = value
            if not environment:
                raise ValueError()
            records.append((str(path), environment))
        if not records or any(env != records[0][1] for _, env in records):
            raise ValueError()
    except Exception:
        raise ValueError('CDI environment is missing, conflicting or unsupported') from None
    return records[0][1], {'paths': [path for path, _ in records], 'names': sorted(records[0][1])}


def cpu_options(scratch, *, paths=SPECS, record=None):
    environment, diagnostic = read_environment(paths)
    scratch = Path(scratch)
    env_file = scratch / 'cdi.env'
    descriptor = os.open(env_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(''.join(f'{name}={value}\n' for name, value in sorted(environment.items())))
    if record:
        Path(record).write_text(json.dumps(diagnostic, indent=2) + '\n')
    return ['--env-file', str(env_file)]


def isolate_cpu(arguments, scratch):
    if arguments[:2] != ['podman', 'run'] or any(
            a.startswith(('--device', '--privileged', '--hooks-dir', '--runtime')) for a in arguments[2:]):
        raise ValueError('CPU CDI launcher requires Podman without devices, privileged mode or runtime overrides')
    hooks = Path(scratch) / 'empty-hooks'
    hooks.mkdir(exist_ok=True, mode=0o700)
    return ['podman', '--hooks-dir', str(hooks), *arguments[1:]]


def command(arguments, scratch, *, paths=SPECS, record=None):
    isolated = isolate_cpu(arguments, scratch)
    return [*isolated[:4], *cpu_options(scratch, paths=paths, record=record), *isolated[4:]]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        call = command(args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments,
                       args.scratch, record=args.record)
        os.execvp(call[0], call)
    except Exception as error:
        print('CPU CDI launcher refused: ' + type(error).__name__, file=sys.stderr)
        raise SystemExit(3) from None
