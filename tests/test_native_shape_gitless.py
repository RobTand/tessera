"""Pure controls for independent host producer and installed runtime provenance."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from tools import tessera_shape_time_panel as app, tessera_shape_time_worker as worker


def test_installed_runtime_metadata_survives_gitless_image(tmp_path,monkeypatch):
    path=tmp_path/'tessera'/'__init__.py';path.parent.mkdir();path.write_text('')
    package=SimpleNamespace(__file__=str(path))
    def no_git(*a,**kw):raise FileNotFoundError('git')
    monkeypatch.setattr(worker.subprocess,'run',no_git)
    distribution=SimpleNamespace(files=[Path('tessera/__init__.py')],locate_file=lambda p:tmp_path/p,
        read_text=lambda *a:json.dumps({'vcs_info':{'vcs':'git','commit_id':'b'*40}}))
    monkeypatch.setattr(worker.importlib.metadata,'distribution',lambda *a:distribution)
    assert worker.observed_commit(package)=='b'*40


def proof(tmp_path):
    value={'schema':app.PRODUCER_SCHEMA,'commit':'a'*40,'commit_source':'sealed_checkout',**app.producer_source_identity()}
    bound=app.publish_json(value,tmp_path/'producer.json')
    return value,bound


def test_producer_recomputes_sealed_source_without_git(tmp_path,monkeypatch):
    value,bound=proof(tmp_path)
    monkeypatch.setattr(app.subprocess,'run',lambda *a,**kw:pytest.fail('container tried to observe host Git'))
    assert app.producer_identity(bound)==value


@pytest.mark.parametrize('fault',['source','tool','label'])
def test_sealed_producer_refuses_unmatched_source(tmp_path,fault):
    value,_=proof(tmp_path)
    if fault=='source':value['source_tree_sha256']='0'*64
    elif fault=='tool':value['tool_source_sha256']='0'*64
    else:value['commit_source']='inferred_from_runtime'
    bound=app.publish_json(value,tmp_path/'changed.json')
    with pytest.raises(ValueError):app.producer_identity(bound)


def test_host_seal_requires_clean_source(tmp_path,monkeypatch):
    def run(argv,**kw):return SimpleNamespace(stdout='a'*40 if 'rev-parse' in argv else ' M src/tessera/example.py')
    monkeypatch.setattr(app.subprocess,'run',run)
    with pytest.raises(ValueError,match='dirty'):app.seal_producer(tmp_path/'never-published.json')
    assert not (tmp_path/'never-published.json').exists()
