"""Opt-in native transport fixture: fail after pytest publishes its population.

Loaded only by the qualification job's explicit ``-p tests.suite_late_failure``.
Ordinary test discovery does not register this hook.
"""


def pytest_unconfigure(config):
    raise RuntimeError("unit3 deliberate fatal exception after population publication")
