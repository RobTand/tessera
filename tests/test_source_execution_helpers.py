"""#808 helper execution composes with unresolved read provenance, not imports."""

import textwrap

import pytest

from test_impacted_tests import _dynamic_repo, impacted


@pytest.mark.parametrize("binding, call", [
    ("from support.compiler import execute", "execute(text)"),
    ("from support.compiler import execute as rebuild", "rebuild(text)"),
    ("import support.compiler as compiler", "compiler.execute(text)"),
    ("from support.compiler import execute\nrun = execute", "run(text)"),
    ("from .compiler import execute", "execute(text)"),
])
@pytest.mark.parametrize("replacement", ["", "execute = replacement", "run = replacement"])
def test_helper_alias_uncertainty_cannot_erase_source_execution(tmp_path, binding, call, replacement):
    repo, _ = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/__init__.py": "",
        "support/reader.py": binding + "\n" + textwrap.dedent(f"""
            def consume(path, replacement):
                {replacement or 'pass'}
                text = path.read_text()
                return {call}
        """),
        "support/compiler.py": "def execute(text): exec(text, {})\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == "full", result
    assert result["unresolved_file_loaders"] == ["support/reader.py"]


def test_helper_chain_reaches_its_fixed_point_without_reading_external_files(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/reader.py": "from support.bridge import rebuild\ndef consume(path): return rebuild(path.read_text())\n",
        "support/bridge.py": "from support.compiler import execute\ndef rebuild(text): return execute(text)\n",
        "support/compiler.py": "def execute(text): exec(text, {})\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == "full", result
    assert result["unresolved_file_loaders"] == ["support/reader.py"]


def test_importing_but_not_calling_an_executor_does_not_promote_a_data_reader(tmp_path):
    repo, _ = _dynamic_repo(tmp_path, "from tools.driver import VALUE\n", {
        "support/reader.py": "from support.compiler import execute\ndef consume(path): return path.read_text()\n",
        "support/compiler.py": "def execute(text): exec(text, {})\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == "narrowed", result
    assert result["unresolved_file_loaders"] == []


@pytest.mark.parametrize("binding", [
    "import omitted_local_source as stock",
    "import foreign_runtime as stock",
    "from support import local_source as stock",
])
def test_absent_import_origin_is_not_proof_of_external_source(tmp_path, binding):
    repo, _ = _dynamic_repo(tmp_path, "def test_unrelated(): pass\n", {
        "support/reader.py": f"{binding}\nfrom pathlib import Path\ndef consume(): exec(Path(stock.__file__).read_text(), {{}})\n",
        "tests/conftest.py": "from support.reader import consume\n",
    })
    result = impacted.select(repo, ["tools/driver.py"])
    assert result["verdict"] == "full", result
    assert result["unresolved_file_loaders"] == ["support/reader.py"]
