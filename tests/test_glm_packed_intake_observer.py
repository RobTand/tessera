"""CPU observation guards; allocation/trace/collective claims require native TP2."""
from types import SimpleNamespace

import pytest
import torch

from experiments.glm_packed_intake_observer import IntakeObservation


def test_intake_observer_checks_zero_wire_anchors_and_complete_projection_count(tmp_path):
    observer = IntakeObservation(tmp_path, {'expected_mode':'rank_local', 'profile_initial_callbacks':3})
    layer = SimpleNamespace(w13_wire=torch.empty(0), w2_wire=torch.empty(0))
    prepared = SimpleNamespace(wire_bytes_resident=lambda:12, rows=2)
    method = SimpleNamespace(_rank_local_intake=SimpleNamespace(prepared={'w13':[[prepared, prepared]],
                                                                          'w2':[[prepared]]}))
    observer.after_create(layer, method)
    observer.count = 3
    observer.after_load(method, 3)
    assert observer.record['after_complete_load'] == {
        'completed_loader_callbacks':3, 'prepared_local_projections':3, 'prepared_local_bytes':60}
    with pytest.raises(AssertionError):
        observer.after_load(method, 4)
    method._rank_local_intake.prepared['w2'][0][0] = None
    with pytest.raises(AssertionError):
        observer.after_load(method, 3)
    layer.w13_wire = torch.zeros(8)
    with pytest.raises(AssertionError):
        observer.after_create(layer, method)


def test_intake_observer_padded_arm_requires_wire_bank_and_no_incremental_owners(tmp_path):
    observer = IntakeObservation(tmp_path, {'expected_mode':'padded', 'profile_initial_callbacks':3})
    layer = SimpleNamespace(w13_wire=torch.zeros(8), w2_wire=torch.zeros(4))
    method = SimpleNamespace()
    observer.after_create(layer, method)
    observer.count = 3
    observer.after_load(method, 3)
    assert observer.record['after_complete_load']['prepared_local_bytes'] == 0
    layer.w13_wire = layer.w2_wire = torch.empty(0)
    with pytest.raises(AssertionError):
        observer.after_create(layer, method)


@pytest.mark.parametrize('options', [{}, {'expected_mode':'maybe', 'profile_initial_callbacks':3},
                                   {'expected_mode':'rank_local', 'profile_initial_callbacks':864}])
def test_intake_observer_refuses_unbounded_or_ambiguous_profile_requests(tmp_path, options):
    with pytest.raises(ValueError, match='exactly 3 callbacks'):
        IntakeObservation(tmp_path, options)
