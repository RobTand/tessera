"""Scientific-independent receipt/parser controls."""
import copy
import subprocess
import sys
from pathlib import Path
import pytest
from tessera.serving import contract as serving_contract, scheme, timing_panel as tp

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


def test_cell_join_reads_the_launch_scope_at_the_panel_rung():
    """Lane schema v12: a launch scoped to other census rungs admits no pair at this rung."""
    doc = copy.deepcopy(serving_contract.load_serving_contract())
    family = serving_contract.PAYLOAD_FAMILY_BY_ROUTE[scheme.TESSERA_FP8]
    fmt = serving_contract.format_entry(family, doc)
    cell = next(c for c in doc["lane_eligibility"]["cells"] if c["platform"] == "sm_121"
                and c["family"] == family and c["structure"] == "dense" and c["regime"] == "batch"
                and "resident" in serving_contract.cell_residency_modes(c)
                and "eager" in serving_contract.cell_runtime_scope(c)[1]
                and serving_contract.cell_is_device_backed(c))
    cell["runtime"].update(tessera_commit="1" * 40, serving_source_sha256="2" * 64)
    runtime = {"image": cell["runtime"]["image"], "tessera_commit": "1" * 40,
               "serving_source_sha256": "2" * 64, "platform": "sm_121",
               "torch": cell["runtime"]["torch"], "vllm": cell["runtime"]["vllm"],
               "serve_flags": {"TESSERA_SERVE_MODE": "resident"}}
    q256 = cell["rungs_q256"][0]
    scope = {"route": scheme.TESSERA_FP8, "regime": "batch", "q256": q256}
    launch = next(x for x in scheme.route_launches(scheme.TESSERA_FP8, structure="dense",
                                                    regime="batch", mode="resident")
                  if x["lane"] and {"symbol": x["symbol"], "decoder": x["decoder"]} in cell["executes"])
    pair = (launch["symbol"], launch["decoder"])
    assert tp.admitted_cell(doc, scope, runtime, pair, [])[0]["id"] == cell["id"]
    # The launch keeps a census rung whose run table is not the panel rung's.
    other = next(r for r in range(256, 2049) if serving_contract.rung_allowable(fmt, r)
                 and serving_contract.rung_rates(fmt, r) != serving_contract.rung_rates(fmt, q256))
    next(e for e in cell["executes"] if (e["symbol"], e["decoder"]) == pair)["rungs_q256"] = [other]
    assert pair not in serving_contract.cell_executes(cell, q256=q256, entry=fmt)
    with pytest.raises(ValueError, match="positively matching"):
        tp.admitted_cell(doc, scope, runtime, pair, [])
