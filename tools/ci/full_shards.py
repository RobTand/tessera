#!/usr/bin/env python3
"""Generate and validate four balanced CPU target lists per architecture.

The disjoint union includes every eligible rule from `bazel query //...`,
including non-test build targets. Each separate Run request retains the PR
gate's 45-minute limit, without putting long label lists in Run actions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_partition  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path('tools/bazel/full_targets')
ARCHES = ('amd64', 'arm64')
COUNT = 4


def populations(query_xml):
    records = {}
    for rule in ET.fromstring(query_xml):
        if rule.tag != 'rule':
            continue
        label = rule.attrib['name'].removeprefix('@@').removeprefix('@')
        if not label.startswith('//') or label in records or any(c.isspace() for c in label):
            raise ValueError('query contains unsupported or duplicate labels')
        records[label] = {
            'test': rule.attrib['class'].endswith('_test'),
            'tags': {tag.attrib['value'] for tag in rule.findall("list[@name='tags']/string")},
        }
    if not records:
        raise ValueError('full query contains no rules')
    return {arch: {
        'rules': sorted(label for label, info in records.items()
                        if not info['tags'] & ({'manual', 'gpu', 'runner'} if arch == 'arm64'
                                               else {'manual', 'gpu'})),
        'tests': sorted(label for label, info in records.items() if info['test']
                        and not info['tags'] & ({'manual', 'gpu', 'runner'} if arch == 'arm64'
                                               else {'manual', 'gpu'})),
    } for arch in ARCHES}


def partition(population, durations):
    unknown = {lane: gate_partition.default_unknown_seconds(costs)
               for lane, costs in durations.items()}
    costs = {}
    for label in population['tests']:
        package, name = label.removeprefix('//').split(':')
        lane = 'bytes' if name.endswith('.bytes') else 'torch'
        path = f'{package}/{name.removesuffix(".bytes")}.py'
        costs[label] = durations[lane].get(path, unknown[lane])
    parts = [{'rules': list(row['files']), 'tests': row['files'],
              'estimated_seconds': row['estimated_seconds']}
             for row in gate_partition.shards(population['tests'], costs, COUNT, unknown_seconds=1)]
    for label in sorted(set(population['rules']) - set(population['tests'])):
        min(parts, key=lambda part: len(part['rules']))['rules'].append(label)
    for part in parts:
        part['rules'].sort()
    return parts


def planned(root, query_xml):
    durations = json.loads((root / 'tools/ci/gate_durations.json').read_text())['lanes']
    files, manifest = {}, {'schema': 'tessera.full-shards.v1', 'architectures': {}}
    for arch, population in populations(query_xml).items():
        shards = []
        for index, part in enumerate(partition(population, durations)):
            path = DIRECTORY / f'{arch}-{index}.targets'
            data = ''.join(label + '\n' for label in part['rules'])
            files[path] = data
            shards.append({**part, 'index': index, 'file': path.as_posix(),
                           'sha256': hashlib.sha256(data.encode()).hexdigest()})
        manifest['architectures'][arch] = shards
    return files, manifest


def generate(root, query_xml):
    files, manifest = planned(root, query_xml)
    (root / DIRECTORY).mkdir(parents=True, exist_ok=True)
    for path, data in files.items():
        (root / path).write_text(data)
    (root / DIRECTORY / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def validate(root, query_xml, arch, index):
    if arch not in ARCHES or type(index) is not int or index not in range(COUNT):
        raise ValueError('invalid full-shard coordinates')
    files, expected = planned(root, query_xml)
    try:
        actual = json.loads((root / DIRECTORY / 'manifest.json').read_text())
        if actual != expected or any((root / path).read_text() != data for path, data in files.items()):
            raise ValueError('full target lists differ from live query; regenerate full_shards.py')
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError('full target lists are absent or malformed; regenerate full_shards.py') from error
    part = expected['architectures'][arch][index]
    return {**part, 'path': str(root / part['file'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--query-xml', type=Path)
    args = parser.parse_args()
    xml = args.query_xml.read_text() if args.query_xml else subprocess.check_output(
        ['bazel', '--max_idle_secs=300', 'query', '--lockfile_mode=error', '--output=xml', '//...'],
        cwd=ROOT, text=True)
    if args.check:
        validate(ROOT, xml, 'amd64', 0)
    else:
        generate(ROOT, xml)


if __name__ == '__main__':
    main()
