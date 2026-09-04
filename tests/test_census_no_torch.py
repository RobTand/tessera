"""Receipt comparison must remain usable in the dependency-free CI population."""
from pathlib import Path
import subprocess
import sys


def test_census_agreement_does_not_import_torch():
    root = Path(__file__).resolve().parents[1]
    script = r'''
import importlib.abc
import sys
sys.path[:0] = [sys.argv[1] + "/src", sys.argv[1]]
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ImportError("torch deliberately unavailable in pure census regression")
sys.meta_path.insert(0, NoTorch())
from tessera.serving.contract import CENSUS_PHASE_REGIMES, PAYLOAD_FAMILY_BY_ROUTE
from tessera.serving.scheme import TESSERA_FP8, launch_pairs
from tools.tessera_route_census import all_structure_agreement
symbol, decoder = next(iter(launch_pairs(
    TESSERA_FP8, structure="routed_moe", regime="decode", mode="resident")))
record = {"kind": "moe", "policy": TESSERA_FP8 + ":resident",
          "symbol": symbol + ":runtime_backend", "decoder": decoder, "state": "served"}
block, problems = all_structure_agreement(
    {"decode": {"experts.child": record}}, cells=[], phase_regimes=CENSUS_PHASE_REGIMES,
    platform="sm_121", declared_rungs={"experts": 1024},
    record_owners={"decode": {"experts.child": "experts"}},
    families_by_route=PAYLOAD_FAMILY_BY_ROUTE,
    runtime_image="example/runtime@sha256:" + "1" * 64, execution_mode="eager")
assert not problems
assert block["agrees"] is None
assert "torch" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-I", "-c", script, str(root)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
