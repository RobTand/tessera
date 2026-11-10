"""Declared expert inputs bind one explicit complete module roster."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('layer',[10,11,42])
def test_explicit_module_roster_binds_without_foreign_layer_alias(layer):
    spec=importlib.util.spec_from_file_location('declared_inputs',ROOT/'experiments/t8r_speed/declared_inputs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    reader=module.DeclaredInputs.__new__(module.DeclaredInputs)
    name=f'model.language_model.layers.{layer}.mlp.experts'
    roles=[{'expert':e,'role':r,'tensor':f'{name}.{e}.{r}.weight'}
           for e in range(288) for r in ('gate_proj','up_proj','down_proj')]
    reader.bind_roles('/unused',roles,module=name)
    assert len(reader.roles)==864
    roles[-1]=dict(roles[-1],tensor=f'{name}.0.gate_proj.weight')
    with pytest.raises(ValueError,match='roster'):
        reader.bind_roles('/unused',roles,module=name)
