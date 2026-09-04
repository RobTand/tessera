"""Tessera wire format, codecs, kernels, allocation, and serving integration.

This lightweight initializer exposes the schema ID and package version without
loading the encoder or serving modules. See ``docs/ARCHITECTURE.md`` for the
current system map and ``docs/schema/prismaquant.tessera.v1.md`` for the wire.
"""

from .manifest import SCHEMA_ID

__all__ = ["SCHEMA_ID", "__version__"]
__version__ = "0.1.0"
