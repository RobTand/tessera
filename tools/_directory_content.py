"""File content inventories shared by runtime attestation and model specs."""
import hashlib
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(root, *, include_cache=False):
    root = Path(root)
    return {str(p.relative_to(root)): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*'))
            if p.is_file() and (include_cache or '__pycache__' not in p.parts)}
