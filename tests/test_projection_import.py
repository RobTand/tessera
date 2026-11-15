"""Import the public decoder without runtime dependencies."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_projection_cold_import_has_no_runtime_dependencies():
    script = '''
import importlib.abc, sys
sys.path.insert(0, sys.argv[1] + "/src")
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in
               ("tessera.serving", "torch", "transformers", "triton", "vllm")):
            raise ImportError("projection crossed dependency boundary: " + fullname)
sys.meta_path.insert(0, Block())
from tessera.projection import direct_consumer_weight
assert callable(direct_consumer_weight)
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(ROOT)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


