"""Development infrastructure that is not part of the installed package.

These modules exist for the repository's own tooling -- the merge-suite
deadline helper, the PrismaBuild source-identity reader, and the import-graph
analyser behind ``tools/impacted_tests.py``.  ``tools/`` imports them, which
is why they live under ``src/`` at all; a consumer of the wheel never calls
them and would not know what to do with a ``tools/`` they do not have.

They are kept out of the distribution by one line of packaging config,
``[tool.setuptools.packages.find] exclude``, and out of the built artifacts
by ``tools/check_wheel.py``, which reads that config and then reads the
wheel and the sdist.  Nothing here is a hand-kept roster of module names.
"""
