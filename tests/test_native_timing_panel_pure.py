"""Scientific-independent receipt/parser controls."""
import subprocess
import sys
from pathlib import Path
import pytest
from tessera.serving import timing_panel as tp

def test_nonfinite_samples_and_duplicate_json_refuse():
    for value in (float("nan"), float("inf"), -1, True):
        with pytest.raises(ValueError): tp.timing_summary([1, 2, value])
    with pytest.raises(ValueError, match="duplicate"): tp.json_bytes(b'{"x":1,"x":2}')


def test_validator_and_canonical_frame_are_torch_free():
    root = Path(__file__).resolve().parents[1]
    script = '''import importlib.abc,sys
sys.path.insert(0,sys.argv[1]+"/src")
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split(".")[0] in {"torch","vllm","prismaquant","prismabuild"}: raise ImportError(fullname)
sys.meta_path.insert(0,Block())
from tessera.serving.timing_panel import timing_summary
from tessera.fused_frame import pack_fused,parse_fused
assert timing_summary([1,2,3,4])["median_ms"]==2.5
assert parse_fused(pack_fused([("role",1,b"x")]))[0].blob==b"x"
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(root)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
