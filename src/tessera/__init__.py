"""Tessera -- ``prismaquant.tessera.v1`` wire schema, parser, and accountant.

This package is build items **1a** and **1b** of the Tessera design document,
plus item 11's pure calculator.  It is deliberately stdlib-only and contains
**no encoder, no decoder, no allocator, no menu, and no shipping-code wiring**:
the document gates all of those, and "no menu, pipeline, or shipping-code
wiring precedes 1b passing" (Codex round-6 P0-1).

See ``docs/schema/prismaquant.tessera.v1.md`` for the normative schema and
``README.md`` for what is owed before any of this lands in PrismaQuant.
"""

from .manifest import SCHEMA_ID

__all__ = ["SCHEMA_ID", "__version__"]
__version__ = "0.1.0"
