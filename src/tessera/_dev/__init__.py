"""Development infrastructure that is not part of the installed package.

These modules exist for the repository's own tooling -- the merge-suite
deadline helper, the PrismaBuild source-identity reader, the import-graph
analyser behind ``tools/impacted_tests.py``, and the publication line the
suite's conftest prints and ``tools/merge_suite.py`` reads back.  ``tools/``
imports them, which is why they live under ``src/`` at all; a consumer of the
wheel never calls them and would not know what to do with a ``tools/`` they do
not have.  A contract between ``tests/`` and ``tools/`` belongs here for the
same reason: it is the only place both can import from.

They are kept out of the distribution by one line of packaging config,
``[tool.setuptools.packages.find] exclude``, and out of the built artifacts
by ``tools/check_wheel.py``, which reads that config and then reads the
wheel and the sdist.  Nothing here is a hand-kept roster of module names.
"""
