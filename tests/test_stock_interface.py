"""CPU facts and refusal controls for stock source/signature inspection."""
from types import SimpleNamespace
import pytest
from tessera.serving import stock_interface as si


def test_complete_interface_tuples_cannot_mix_inspected_revisions(tmp_path):
    modules=[]
    for name,raw in [('a','version-a'),('b','version-b')]:
        path=tmp_path/name;path.write_text(raw);modules.append(SimpleNamespace(__file__=str(path)))
    a,b=[si.module_digest(module) for module in modules]
    interfaces=(si.InspectedInterface('first',(a,'0'*64)),si.InspectedInterface('second',('1'*64,b)))
    assert si.match_modules(modules,('a','b'),interfaces)[0] is None
    exact=si.InspectedInterface('exact',(a,b))
    assert si.match_modules(modules,('a','b'),(exact,))[0] is exact


@pytest.mark.parametrize('fault',['missing','directory','invalid_path','getter'])
def test_unreadable_source_is_a_declining_fact(tmp_path,fault):
    path=tmp_path/'source.py';path.write_text('source')
    module=SimpleNamespace(__file__=str(path))
    if fault=='missing':path.unlink()
    elif fault=='directory':module.__file__=str(tmp_path)
    elif fault=='invalid_path':module.__file__=object()
    else:
        class BadSource:
            @property
            def __file__(self):raise ValueError('unreadable source attribute')
        module=BadSource()
    assert si.source_digest(module) is None
    assert si.module_digest(module) is None


@pytest.mark.parametrize('fault',['missing','not_callable','bad_signature'])
def test_uninspectable_parameters_are_a_declining_fact(fault):
    owner=SimpleNamespace()
    if fault=='not_callable':owner.method=object()
    elif fault=='bad_signature':
        class BadSignature:
            def __call__(self):pass
            @property
            def __signature__(self):raise ValueError('uninspectable signature')
        owner.method=BadSignature()
    assert si.signature_parameters(owner,'method') is None
