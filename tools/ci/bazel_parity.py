#!/usr/bin/env python3
"""Hold Bazel's test population to the hosted gate's, statically (#1589).

The hosted gate runs every tests/test_*.py file (tools/ci/gate_select.py
population) in two lanes, bytes and torch, minus the node-level quarantine in
tools/ci/gate_quarantine.json. Bazel runs one target per file and lane:
`test_x` (torch) and `test_x.bytes`. This fails, naming the file and the fix,
when a gate file has no target in a lane, when a target the cloud gate leaves
out (`manual`) has no recorded reason, or when a table entry in
tools/bazel/nonhermetic.bzl or tools/bazel/test_inputs.bzl names no target or
gives no reason. Every BUILD.bazel is generated: the fix is
tools/bazel/generate.sh, or a reasoned table entry.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
FIX = 'tools/bazel/generate.sh'
TARGET = re.compile(r'^ts_py_test\(\n(.*?)^\)', re.MULTILINE | re.DOTALL)
FIELD = re.compile(r'^    (name|path|lane) = "([^"]*)",$', re.MULTILINE)
TAGS = re.compile(r'^    tags = \[(.*?)\],$', re.MULTILINE | re.DOTALL)
ENTRY = re.compile(r'^    "(//[^"]+)": struct\((.*?)\),$', re.MULTILINE | re.DOTALL)
#: Tags that explain why the cloud gate leaves a target out.
EXPLAINED = {'gpu': 'tools/bazel/nonhermetic.bzl', 'gate-quarantine': 'tools/ci/gate_quarantine.json'}


def parse_targets(text):
    """[{'name', 'path', 'tags'}] for every ts_py_test in a generated BUILD file."""
    targets = []
    for block in TARGET.findall(text):
        fields = dict(FIELD.findall(block))
        tags = TAGS.search(block)
        targets.append({'name': fields.get('name'), 'path': fields.get('path'),
                        'tags': re.findall(r'"([^"]+)"', tags.group(1)) if tags else []})
    return targets


def population(root):
    spec = importlib.util.spec_from_file_location('gate_select', root / 'tools/ci/gate_select.py')
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / 'tools/ci'))
    spec.loader.exec_module(module)
    return module.population(root)


def missing_lanes(targets, files):
    names = {target['name'] for target in targets}
    missing = []
    for path in files:
        stem = path.removeprefix('tests/').removesuffix('.py')
        missing += [f'{path}: no {lane} target //tests:{name}'
                    for lane, name in (('torch', stem), ('bytes', stem + '.bytes')) if name not in names]
    return missing


def table(root, path):
    """{label: entry text} from a .bzl table of struct entries."""
    return dict(ENTRY.findall((root / path).read_text()))


def exclusions_without_reason(targets, nonhermetic, quarantine):
    """Labels of `manual` targets with no tag that explains them, or no table entry behind the tag."""
    found = []
    for target in targets:
        if 'manual' not in target['tags']:
            continue
        label = '//tests:' + target['name']
        base = label.removesuffix('.bytes')
        if 'gpu' in target['tags'] and base in nonhermetic:
            continue
        if 'gate-quarantine' in target['tags'] and target['path'] in quarantine:
            continue
        found.append(label)
    return found


def _targets(root):
    return parse_targets((root / 'tests/BUILD.bazel').read_text())


def missing_targets(root=ROOT):
    return missing_lanes(_targets(root), population(root))


def unexplained_exclusions(root=ROOT):
    lanes = json.loads((root / 'tools/ci/gate_quarantine.json').read_text())['lanes']
    quarantine = {node for nodes in lanes.values() for node in nodes}
    return exclusions_without_reason(_targets(root), table(root, 'tools/bazel/nonhermetic.bzl'), quarantine)


def table_problems(root=ROOT):
    names = {'//tests:' + target['name'] for target in _targets(root)}
    problems = []
    for path in ('tools/bazel/nonhermetic.bzl', 'tools/bazel/test_inputs.bzl'):
        for label, entry in table(root, path).items():
            if label not in names:
                problems.append(f'{path}: {label} names no test target')
            reason = re.search(r'reason = (\S.*?)(?:,\s*$|$)', entry, re.MULTILINE | re.DOTALL)
            if not reason or reason.group(1).strip() in ('""', "''"):
                problems.append(f'{path}: {label} gives no reason')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    problems = missing_targets() + [f'{label} is left out of the cloud gate without a reason'
                                    for label in unexplained_exclusions()] + table_problems()
    for problem in problems:
        print(problem)
    if problems:
        print(f'Fix: {FIX}, or a reasoned entry in tools/bazel/nonhermetic.bzl')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
