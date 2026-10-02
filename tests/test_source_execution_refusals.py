"""#808 independent-review findings: refusal consistency and finite helper facts."""

import ast
import sys
import textwrap
from pathlib import Path

import pytest

from test_impacted_tests import _dynamic_repo, impacted


@pytest.mark.parametrize("target", ["reader", "compiler"])
@pytest.mark.parametrize("failure", ["read", "parse"])
def test_first_source_failure_is_authoritative_without_retry(tmp_path, monkeypatch, target, failure):
    repo, _ = _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", {
        "support/reader.py": "from support.compiler import execute\ndef consume(path): return execute(path.read_text())\n",
        "support/compiler.py": "def execute(text): exec(text, {})\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    failed = repo / f"support/{target}.py"
    original = Path.read_text
    attempts = []

    def read_once(path, *args, **kwargs):
        if path == failed:
            attempts.append(path)
            if len(attempts) == 1:
                if failure == "read":
                    raise OSError("transient source refusal")
                return "def (\n"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_once)
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == "full", result
    diagnostic = result["unreadable_sources"][str(failed.relative_to(repo))]
    assert diagnostic.startswith("OSError:" if failure == "read" else "SyntaxError:")
    assert len(attempts) == 1, "retrying edges without rebuilding helper facts erases uncertainty"


@pytest.mark.parametrize("executor", [False, True], ids=["ordinary-call", "helper-call"])
def test_deep_acyclic_callable_aliases_do_not_consume_the_python_stack(tmp_path, executor):
    depth = sys.getrecursionlimit() + 1
    source = "from support.compiler import execute\na0 = " + ("execute" if executor else "unknown") + "\n"
    source += "".join(f"a{i} = a{i - 1}\n" for i in range(1, depth + 1))
    source += f"def consume(path): return a{depth}(path.read_text())\n"
    repo, _ = _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", {
        "support/reader.py": source,
        "support/compiler.py": "def execute(text): exec(text, {})\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == ("full" if executor else "narrowed"), result
    assert result["unresolved_file_loaders"] == (["support/reader.py"] if executor else [])


_EXECUTOR_ALIASES = [
    pytest.param("from runpy import run_module as run", "run(text)", id="runpy-module-alias"),
    pytest.param("import runpy as runner\nrun = runner.run_module", "run(text)", id="assigned-runpy-alias"),
    pytest.param("from builtins import exec as run", "run(text)", id="builtin-exec-alias"),
    pytest.param("from builtins import compile as build", "build(text, 'input', 'exec')", id="builtin-compile-alias"),
    pytest.param("import codeop as compiler\nbuild = compiler.compile_command", "build(text)", id="compile-command-alias"),
    pytest.param("from ast import parse as parse_source", "parse_source(text)", id="qualified-parser-control"),
]


@pytest.mark.parametrize("binding, call", _EXECUTOR_ALIASES)
def test_module_scoped_executor_aliases_seed_imported_helpers(tmp_path, binding, call):
    repo, _ = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/reader.py": "from support.compiler import execute\ndef consume(path): return execute(path.read_text())\n",
        "support/compiler.py": binding + "\ndef execute(text): return " + call + "\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == "full", result
    assert "support/reader.py" in result["unresolved_file_loaders"]


@pytest.mark.parametrize("binding, call", _EXECUTOR_ALIASES)
def test_module_and_helper_recognition_share_the_source_call_rule(tmp_path, binding, call):
    from tessera._dev.source_dependencies import file_imports

    source = binding + "\ndef consume(path):\n    text = path.read_text()\n    return " + call + "\n"
    repo, _ = _dynamic_repo(tmp_path, source)
    _, unknown, _ = file_imports(ast.parse(source), repo / "tests/test_dynamic.py", repo)
    assert unknown, "the same recognized call must classify direct readers as well as helpers"


@pytest.mark.parametrize("execute", [False, True], ids=["data-control", "executor"])
def test_reexported_helpers_retain_the_exported_callable_capability(tmp_path, execute):
    repo, _ = _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", {
        "support/reader.py": "from support.bridge import rebuild\ndef consume(path): return rebuild(path.read_text())\n",
        "support/bridge.py": "from support.compiler import execute as rebuild\n",
        "support/compiler.py": "def execute(text): " + ("exec(text, {})" if execute else "return text") + "\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == ("full" if execute else "narrowed"), result
    assert result["unresolved_file_loaders"] == (["support/reader.py"] if execute else [])


@pytest.mark.parametrize("execute", [False, True], ids=["all-data", "mixed-candidates"])
def test_ambiguous_helper_spelling_keeps_every_candidate(tmp_path, execute):
    repo, _ = _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", {
        "support/reader.py": "from compiler import execute\ndef consume(path): return execute(path.read_text())\n",
        "compiler.py": "def execute(text): return text\n",
        "tests/compiler.py": "def execute(text): " + ("exec(text, {})" if execute else "return text") + "\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == ("full" if execute else "narrowed"), result
    assert result["unresolved_file_loaders"] == (["support/reader.py"] if execute else [])


@pytest.mark.parametrize("seeded", [False, True], ids=["unseeded", "executor-seeded"])
def test_callable_cycles_propagate_only_established_executor_seeds(tmp_path, seeded):
    seed = "\nfrom support.compiler import execute\nb = execute" if seeded else ""
    repo, _ = _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", {
        "support/reader.py": "a = b\nb = a" + seed + "\ndef consume(path): return a(path.read_text())\n",
        "support/compiler.py": "def execute(text): exec(text, {})\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == ("full" if seeded else "narrowed"), result


def test_attribute_binding_cycle_terminates_without_manufacturing_an_executor(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", {
        "support/reader.py": "alias = alias.member\ndef consume(path): return alias(path.read_text())\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == "narrowed", result
    assert result["unresolved_file_loaders"] == []


def test_uncertainty_predecessors_name_resolved_files_from_seed_to_conftest(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/reader.py": "def consume(path): exec(path.read_text(), {})\n",
        "support/bridge.py": "from support.reader import consume\n",
        "tests/conftest.py": "from support.bridge import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["uncertainty_paths"] == [{
        "seed": "support/reader.py", "seed_kind": "unresolved_file_loader",
        "source_failure": None, "conftest": "tests/conftest.py",
        "path": ["support/reader.py", "support/bridge.py", "tests/conftest.py"],
    }]
    assert result["uncertainty_collection_probes_skipped"] == []


@pytest.mark.parametrize("probe", [False, True], ids=["ordinary-import", "collection-probe"])
def test_uncertainty_diagnostic_uses_the_same_collection_probe_exclusion(tmp_path, probe):
    conftest = "from tests.test_dynamic import consume\n"
    if probe:
        conftest = textwrap.dedent('''
            from pathlib import Path
            from importlib.util import spec_from_file_location
            for path in Path(__file__).parent.glob("test_*.py"):
                spec_from_file_location("probe", path)
        ''')
    repo, _ = _dynamic_repo(tmp_path, "import ast\ndef consume(path): return ast.parse(path.read_text())\n", {
        "tests/conftest.py": conftest,
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == ("narrowed" if probe else "full"), result
    if probe:
        assert result["uncertainty_paths"] == []
        assert result["uncertainty_collection_probes_skipped"] == [{
            "seed": "tests/test_dynamic.py", "from": "tests/test_dynamic.py",
            "to": "tests/conftest.py",
        }]
    else:
        assert result["uncertainty_paths"][0]["path"] == ["tests/test_dynamic.py", "tests/conftest.py"]
        assert result["uncertainty_collection_probes_skipped"] == []
