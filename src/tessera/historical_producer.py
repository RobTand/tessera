"""Bind an original cached-unit producer without replacing the current reader.

Only the historical identity factory and verifier are imported.  The full
producer package has one exact source seal; each imported file is checked
again against that seal, and its serving namespace is inaccessible.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import re
import sys
from types import SimpleNamespace


SOURCE_SUFFIXES = {".py", ".cu", ".cuh", ".cpp", ".h"}


class _SealedLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, digest):
        super().__init__(fullname, path)
        self.digest = digest

    def get_code(self, fullname):
        source = Path(self.path).read_bytes()
        if hashlib.sha256(source).hexdigest() != self.digest:
            raise ImportError(f"historical producer source changed: {self.path}")
        tree = ast.parse(source, filename=self.path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            elif isinstance(node, ast.Call) and node.args:
                function = node.func
                name = function.attr if isinstance(function, ast.Attribute) else getattr(function, "id", "")
                names = ([node.args[0].value] if name in {"import_module", "__import__", "files"}
                         and isinstance(node.args[0], ast.Constant) else [])
            else:
                names = []
            if any(isinstance(name, str) and
                   (name == "tessera" or name.startswith("tessera.")) for name in names):
                raise ImportError(f"{fullname} would escape into the current producer")
        return compile(tree, self.path, "exec", dont_inherit=True)


class _SealedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, namespace, files):
        self.namespace, self.files = namespace, files

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self.namespace + "."):
            return None
        relative = fullname[len(self.namespace) + 1:]
        if relative.split(".")[0] in {"serving", "stock", "kernel_window_gemv"}:
            raise ImportError("historical producer cannot import serving or stock")
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.origin is None:
            raise ImportError(f"historical producer module absent: {fullname}")
        origin = str(Path(spec.origin).resolve())
        if origin not in self.files or not origin.endswith(".py"):
            raise ImportError(f"historical producer module escaped source tree: {fullname}")
        spec.loader = _SealedLoader(fullname, origin, self.files[origin])
        return spec


def load_historical_producer(package: Path, source_sha256: str):
    """Return the producer's original receipt API, kept outside ``tessera``.

    This hashes the bounded producer *package*, not the model checkpoint.
    Every original source tensor, Hessian and numerical setting is still
    derived independently by the caller before verifying an existing receipt.
    """
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("historical producer requires an exact source SHA256")
    root = Path(package).resolve()
    if not (root / "__init__.py").is_file():
        raise ValueError("historical producer package lacks __init__.py")
    digest = hashlib.sha256()
    files = {}
    for path in sorted(p for p in root.rglob("*") if p.suffix in SOURCE_SUFFIXES):
        if path.is_symlink() or not path.is_file():
            raise ValueError("historical producer source must contain regular files")
        source = path.read_bytes()
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(source)
        digest.update(b"\0")
        files[str(path.resolve())] = hashlib.sha256(source).hexdigest()
    if digest.hexdigest() != source_sha256:
        raise ValueError("historical producer package source SHA256 differs")
    namespace = "tessera_cached_producer_" + source_sha256
    if namespace not in sys.modules:
        initial = str(root / "__init__.py")
        finder = _SealedFinder(namespace, files)
        spec = importlib.util.spec_from_file_location(
            namespace, initial, submodule_search_locations=[str(root)],
            loader=_SealedLoader(namespace, initial, files[initial]))
        module = importlib.util.module_from_spec(spec)
        sys.meta_path.insert(0, finder)
        sys.modules[namespace] = module
        try:
            spec.loader.exec_module(module)
            importlib.import_module(namespace + ".cached_unit")
        except BaseException:
            for name in tuple(sys.modules):
                if name == namespace or name.startswith(namespace + "."):
                    del sys.modules[name]
            sys.meta_path.remove(finder)
            raise
    elif Path(sys.modules[namespace].__file__).resolve().parent != root:
        raise ValueError("historical producer namespace is bound to another package")
    cached = importlib.import_module(namespace + ".cached_unit")
    control = importlib.import_module(namespace + ".control")
    if cached.encoder_source_sha256() != source_sha256:
        raise ValueError("historical producer disagrees with its package source seal")
    return SimpleNamespace(source_sha256=source_sha256, namespace=namespace,
                           input_identity=cached.unit_input_identity,
                           dense_identity=cached.encoding_input_identity,
                           verify=cached.verify_cached_unit,
                           grid_for_name=control.grid_for_name)
