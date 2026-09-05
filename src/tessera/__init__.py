"""Tessera wire format, codecs, kernels, allocation, and serving integration.

This lightweight initializer exposes the schema ID and package version without
loading the encoder or serving modules. See ``docs/ARCHITECTURE.md`` for the
current system map and ``docs/schema/prismaquant.tessera.v1.md`` for the wire.

WHERE THE VERSION COMES FROM.  ``pyproject.toml``'s ``[project] version`` is
the one declaration in this repository; nothing else states it.  Two readers,
in this order:

* **the checkout** -- a ``pyproject.toml`` beside this package's ``src/``
  whose ``[project] name`` is this distribution.  It is read FIRST because it
  describes the code actually being imported, which installed metadata need
  not: this repository runs as ``PYTHONPATH=src`` in venvs that also hold a
  ``tessera-quant`` dist-info pointing at a different checkout, and a wheel
  build leaves an ``src/tessera_quant.egg-info`` behind in this one.
* **the installed distribution** -- ``importlib.metadata.version`` of
  :data:`DISTRIBUTION`, which is what a wheel has and a checkout does not.

Neither found is a refusal, not a placeholder: the string is an input to the
vLLM compile-cache key (``serving/compile_identity.py``) and is published by
the route census, so a guessed version is a wrong receipt and a wrong cache
key.  ``tests/test_packaging.py`` holds every remaining copy -- the serving
package's re-export, the packaged runtime contract, the documentation URL --
to this one declaration.
"""

from pathlib import Path

from .manifest import SCHEMA_ID

__all__ = ["SCHEMA_ID", "DISTRIBUTION", "__version__"]

#: The distribution name on PyPI.  The import name is ``tessera``.
DISTRIBUTION = "tessera-quant"


def _checkout_version() -> str | None:
    """The version declared by the checkout this package is imported from."""
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # < 3.11; the installed metadata answers
        return None
    project = tomllib.loads(text).get("project", {})
    # A vendored copy of this package can sit beside a foreign pyproject.
    if project.get("name") != DISTRIBUTION:
        return None
    return project.get("version")


def _installed_version() -> str | None:
    """The version of the installed distribution, if this is one."""
    from importlib import metadata

    try:
        return metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        return None


def _resolve_version() -> str:
    for read in (_checkout_version, _installed_version):
        version = read()
        if version:
            return version
    raise RuntimeError(
        f"cannot read the version of {DISTRIBUTION}: no [project] table naming "
        f"it in a pyproject.toml beside {Path(__file__).resolve().parent.parent}, "
        "and no installed distribution metadata.  The version is a compile-cache "
        "key input and a receipt field, so it is refused rather than guessed"
    )


__version__ = _resolve_version()
