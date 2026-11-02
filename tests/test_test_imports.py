"""Test-to-test imports may only shrink as shared helpers move out of tests."""
import ast
from pathlib import Path


def test_test_to_test_imports_only_shrink():
    tests = Path(__file__).resolve().parent
    modules = {path.stem: path for path in tests.glob("test_*.py")}
    actual = set()
    for importer, path in sorted(modules.items()):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
                if node.module in (None, "tests"):
                    names.extend(alias.name for alias in node.names)
            elif (isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and isinstance(node.func.value, ast.Name)
                  and node.func.value.id == "importlib"
                  and node.func.attr == "import_module"
                  and node.args
                  and isinstance(node.args[0], ast.Constant)
                  and isinstance(node.args[0].value, str)):
                names = [node.args[0].value]
            for name in names:
                imported = name.removeprefix("tests.")
                if imported in modules:
                    actual.add(f"tests/{importer}.py -> tests/{imported}.py")

    recorded = set((tests / "fixtures/test_to_test_imports.txt").read_text().splitlines())
    assert actual <= recorded, "Unrecorded test imports:\n" + "\n".join(sorted(actual - recorded))
    assert recorded <= actual, "Remove vanished test imports:\n" + "\n".join(sorted(recorded - actual))
