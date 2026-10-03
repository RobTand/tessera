"""Changed-source dual-B numeric protocol controls; no CUDA qualification."""
import pytest
from test_piece_major_comparison_protocol import pp, protocol, options, environment

SCHEMA = "tessera.routed_mma8_dual_b_numeric.v1"


def dual(tmp_path, choice=0, width=64):
    doc = protocol(tmp_path)
    doc.update(schema=SCHEMA, power_s=0, order=["legacy", "piece_major"],
               dual_b={"compile_choice":choice,"superblock_rows":width})
    return doc


@pytest.mark.parametrize("choice",[0,1])
@pytest.mark.parametrize("width",[64,128])
def test_new_numeric_admits_only_bound_compile_choice_and_width(tmp_path,choice,width):
    doc=dual(tmp_path,choice,width)
    assert pp.validate(doc) is doc


@pytest.mark.parametrize("field,value",[("compile_choice",True),("compile_choice",2),
    ("compile_choice","1"),("superblock_rows",32),("superblock_rows",True)])
def test_new_numeric_refuses_unknown_or_coerced_geometry(tmp_path,field,value):
    doc=dual(tmp_path);doc["dual_b"][field]=value
    with pytest.raises(ValueError):pp.validate(doc)


@pytest.mark.parametrize("choice,width",[(0,64),(1,128)])
def test_actual_new_protocol_options_bind_environment_before_cuda(tmp_path,monkeypatch,choice,width):
    doc=dual(tmp_path,choice,width)
    environment(monkeypatch,doc)
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_WIDE","0" if width==64 else "1")
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH",str(choice))
    args=options(doc);args.power_s=0
    pp.require_options(args,doc,stubbed=False)
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH",str(1-choice))
    with pytest.raises(ValueError):pp.require_options(args,doc,stubbed=False)


@pytest.mark.parametrize("phase",["timing","repeatability","ncu"])
def test_new_numeric_cannot_borrow_performance_or_old_proof_phase(tmp_path,monkeypatch,phase):
    doc=dual(tmp_path);environment(monkeypatch,doc)
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_WIDE","0")
    monkeypatch.setenv("TESSERA_ROUTED_FUSED_MMA8_GATE_UP_B_PREFETCH","0")
    args=options(doc);args.power_s=0;args.comparison_phase=phase
    with pytest.raises(ValueError):pp.require_options(args,doc,stubbed=False)


def test_new_numeric_refuses_historical_numeric_protocol_as_changed_source_proof(tmp_path):
    doc=dual(tmp_path)
    doc["numeric_protocol"]={"path":str(tmp_path/"old.json"),"sha256":"a"*64}
    with pytest.raises(ValueError):pp.validate(doc)
