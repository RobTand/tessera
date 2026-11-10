"""Sorted recursive entries for dependency seals and model content proofs."""
from pathlib import Path


def directory_entries(root):
    """Retain relative names and file symlinks, without following directory links."""
    return sorted(Path(root).rglob('*'))
