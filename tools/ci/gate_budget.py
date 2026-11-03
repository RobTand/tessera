"""Per-test budget validation and JUnit timing rules, without policy-file reads.

The hosted gate reads its complete policy before calling these rules. Bazel
passes only the current test file's policy as an action argument.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

from gate_junit import test_name

BUDGET_ENFORCEMENT_TOLERANCE = 1.5


class BudgetFailure(ValueError):
    """A definite policy failure, never an infrastructure retry."""


def limits(config):
    if not isinstance(config, dict) or not isinstance(config.get('slow'), dict):
        raise BudgetFailure('Test budget requires seconds and a slow map')
    limits = [config.get('seconds'), *config['slow'].values()]
    if (any(isinstance(limit, bool) or not isinstance(limit, (int, float)) or
            not math.isfinite(limit) or limit <= 0 for limit in limits) or
            any(not isinstance(name, str) or not name for name in config['slow'])):
        raise BudgetFailure('Test budget limits must be finite positive seconds')
    return config


def timings(junit, files, config, require_all=True):
    observed, seconds = set(), {}
    for case in ET.parse(junit).iter('testcase'):
        filename, name = case.get('file', '').removeprefix('./'), case.get('name', '')
        name = test_name(name)
        classname = case.get('classname', '')
        if not classname and case.find('skipped') is not None and (name in files or filename in files):
            continue  # A module-level importorskip is not a test timing.
        if filename not in files or not name:
            raise ValueError('JUnit timing has missing or unexpected test identity')
        module = filename.removesuffix('.py').replace('/', '.')
        if classname != module and not classname.startswith(module + '.'):
            raise ValueError('JUnit timing cannot resolve test class: ' + classname)
        classes = classname[len(module):].lstrip('.').replace('.', '::')
        nodeid = filename + '::' + (classes + '::' if classes else '') + name
        try:
            elapsed = float(case.attrib['time'])
        except (KeyError, ValueError) as error:
            raise ValueError('JUnit timing missing or invalid for ' + nodeid) from error
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError('JUnit timing must be finite nonnegative seconds: ' + nodeid)
        observed.add(filename)
        seconds[nodeid] = seconds.get(nodeid, 0) + elapsed
    if require_all and observed != set(files):
        raise ValueError('JUnit timings do not cover every assigned file')
    return [{'nodeid': nodeid, 'seconds': elapsed,
             'budget_seconds': config['slow'].get(nodeid, config['seconds'])}
            for nodeid, elapsed in sorted(seconds.items())
            if elapsed > config['slow'].get(nodeid, config['seconds'])]
