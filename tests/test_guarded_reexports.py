"""A proved lazy export condition must not transfer unrelated uncertainty."""

from pathlib import Path

import pytest

from test_impacted_tests import _dynamic_repo, _files_an_import_actually_reaches, impacted


_HOOK = '''\
_NAMES = frozenset({'lazy'})
def safe(): return 1
def __getattr__(name: str):
    if name in _NAMES:
        from . import impl as _impl
        return getattr(_impl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
'''
_IMPL = "def lazy(path): return exec(path.read_text(), {})\n"


def _fixture(tmp_path, consumer="from support.lazy import safe\n", *, hook=_HOOK, extra=None):
    files = {
        "support/__init__.py": "",
        "support/lazy.py": hook,
        "support/impl.py": _IMPL,
        "tests/conftest.py": consumer,
    }
    files.update(extra or {})
    return _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", files)[0]


def _select(repo):
    return impacted.select(repo, ["tools/driver.py"])


@pytest.mark.parametrize("consumer", [
    "from support.lazy import safe\n",
    "from support.lazy import safe as renamed\n",
    "from support.facade import safe\n",
])
def test_direct_nonlazy_names_do_not_invoke_the_guarded_branch(tmp_path, consumer):
    repo = _fixture(tmp_path, consumer, extra={
        "support/facade.py": "from .lazy import safe\n",
        "tests/test_lazy.py": "from support.lazy import lazy\n",
    })
    result = _select(repo)
    assert result["verdict"] == "narrowed", result
    assert result["uncertainty_paths"] == []
    assert "support/impl.py" in result["unresolved_file_loaders"]
    assert "tests/test_lazy.py" in result["tests"]


@pytest.mark.parametrize("consumer", [
    "from support.lazy import lazy\n",
    "from support.lazy import lazy as renamed\n",
    "from support.impl import lazy\n",
    "import support.lazy\n",
    "import support.lazy as module\n",
    "from support import lazy as module\n",
    "from support.lazy import *\n",
    "from support.lazy import __getattr__ as lookup\n",
    "import support.lazy as module\nvalue = getattr(module, runtime_name)\n",
    "import support.lazy as module\na = module\nb = a\na = b\nvalue = a.lazy\n",
    "from support.facade import helper\nhelper(runtime_name)\n",
])
def test_lazy_unknown_and_escaping_demands_keep_the_union(tmp_path, consumer):
    repo = _fixture(tmp_path, consumer, extra={
        "support/facade.py": "from .lazy import __getattr__ as lookup\ndef helper(name): return lookup(name)\n",
    })
    result = _select(repo)
    assert result["verdict"] == "full", result
    assert "tests/conftest.py" in result["forces_full"]


@pytest.mark.parametrize("change", [
    lambda body: body.replace("{'lazy'}", "{'lazy', '__path__'}"),
    lambda body: body + "_NAMES = frozenset({'safe'})\n",
    lambda body: body + "_NAMES |= frozenset({'safe'})\n",
    lambda body: body + "_NAMES.add('safe')\n",
    lambda body: body + "saved = _NAMES\n",
    lambda body: body + "saved = __getattr__\n",
    lambda body: body + "saved = globals()\n",
    lambda body: body + "from builtins import globals as get_globals\nsaved = get_globals()\n",
    lambda body: body.replace("    if name in", "    record()\n    if name in"),
    lambda body: body.replace("        from . import", "        record()\n        from . import"),
    lambda body: body.replace("def __getattr__(name: str):", "@decorate\ndef __getattr__(name: str):"),
    lambda body: body.replace("def __getattr__(name: str):", "def __getattr__(name=compute()):"),
    lambda body: body.replace("frozenset({'lazy'})", "set({'lazy'})"),
    lambda body: body + "getattr = custom\n",
    lambda body: body + "from builtins import exec as run\nrun(source)\n",
    lambda body: body.replace("raise AttributeError(f\"module {__name__!r} has no attribute {name!r}\")", "raise AttributeError(compute())"),
], ids=["implicit-path", "rebound", "augmented", "mutation", "guard-escape", "hook-escape",
        "globals-escape", "globals-alias", "extra-hook-effect", "extra-branch-effect",
        "decorated-hook", "computed-default", "mutable-guard", "shadowed-builtin", "exec-alias",
        "computed-refusal"])
def test_unsupported_or_mutated_grammar_keeps_the_union(tmp_path, change):
    repo = _fixture(tmp_path, hook=change(_HOOK))
    assert _select(repo)["verdict"] == "full"


@pytest.mark.parametrize("effect", ["provider", "initializer"])
def test_ordinary_and_initialization_imports_stay_unconditional(tmp_path, effect):
    extra = {"support/__init__.py": "from .impl import lazy\n"} if effect == "initializer" else {}
    hook = "from .impl import lazy as eager\n" + _HOOK if effect == "provider" else _HOOK
    assert _select(_fixture(tmp_path, hook=hook, extra=extra))["verdict"] == "full"


def test_ambiguous_module_candidates_do_not_choose_a_nonlazy_winner(tmp_path):
    repo = _fixture(tmp_path, extra={
        "src/support/lazy.py": _HOOK.replace("{'lazy'}", "{'safe'}"),
    })
    assert _select(repo)["verdict"] == "full"


def test_an_explicit_path_load_keeps_all_guarded_dependencies(tmp_path):
    repo = _fixture(tmp_path, consumer='''\
import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parents[1] / 'support' / 'lazy.py'
spec = importlib.util.spec_from_file_location('loaded', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
''')
    assert _select(repo)["verdict"] == "full"


@pytest.mark.parametrize("failure", ["read", "parse"])
def test_failed_provider_source_cannot_publish_a_guarded_summary(tmp_path, monkeypatch, failure):
    repo = _fixture(tmp_path)
    failed = repo / "support/lazy.py"
    original = Path.read_text
    calls = []

    def refuse(path, *args, **kwargs):
        if path == failed:
            calls.append(path)
            if failure == "read":
                raise OSError("provider source unavailable")
            return "def (\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)
    result = _select(repo)
    assert result["verdict"] == "full"
    assert "support/lazy.py" in result["unreadable_sources"]
    assert len(calls) == 1


def test_literal_directory_enumeration_does_not_execute_a_lazy_export(tmp_path):
    hook = _HOOK + "__all__ = ['safe', 'lazy']\ndef __dir__(): return sorted(set(__all__) | _NAMES)\n"
    assert _select(_fixture(tmp_path, hook=hook))["verdict"] == "narrowed"


@pytest.mark.parametrize("extra", [
    "__all__.append('other')\n",
    "saved = __all__\n",
    "def __dir__():\n    record()\n    return sorted(set(__all__) | _NAMES)\n",
])
def test_unsupported_directory_or_exports_keep_the_union(tmp_path, extra):
    hook = _HOOK + "__all__ = ['safe', 'lazy']\ndef __dir__(): return sorted(set(__all__) | _NAMES)\n" + extra
    assert _select(_fixture(tmp_path, hook=hook))["verdict"] == "full"


def test_literal_from_import_matches_actual_initialization_reach(tmp_path):
    repo = _fixture(tmp_path, extra={
        "tests/test_safe.py": "from support.lazy import safe\ndef test_safe(): assert safe() == 1\n",
    })
    actual = _files_an_import_actually_reaches(repo, "tests/test_safe.py")
    assert "support/lazy.py" in actual
    assert "support/impl.py" not in actual
    result = _select(repo)
    assert result["verdict"] == "narrowed", result
    assert "tests/test_safe.py" not in result["tests"], result
