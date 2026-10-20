"""Read-only semantic regressions against the configured publication's helpers."""
from test_stageprev_793_prepare import _run_published_sdk


def test_delegated_semantics_refuse_broken_predicate_and_denial_paths():
    _run_published_sdk('''
        import inspect
        import stageprev_793_claim_contract as contract
        from prismabuild.pool import candidate_policy

        decision = inspect.getsource(client.PoolQueue._claim_pass)
        preparation = inspect.getsource(client.PoolQueue._prepare_host_claim)
        predicate = inspect.getsource(candidate_policy.never_fits)
        assert contract.delegated_positive_kind_refusal(decision, preparation, predicate)
        mutations = [
            (2, '< need', '> need'),
            (2, 'total.get(kind, 0)', 'available.get(kind, 0)'),
            (2, 'total.get(kind, 0)', 'total.get(kind, 1)'),
            (2, 'reservation_demand.items()', 'demand.items()'),
            (2, 'reservation_demand.items()', 'reservation_demand.items() if need'),
            (1, 'candidate.demand.reservation', 'candidate.demand.host'),
            (1, 'never_fits(reservation_demand, total)', 'never_fits(reservation_demand, available)'),
            (1, '"never_fits_capacity"', '"other"'),
            (1, '"capacity_total": total', '"capacity_total": available'),
            (0, 'total=scan_pass.total', 'total=scan_pass.available'),
            (0, 'prepared.reason', '"other"'),
            (0, 'prepared.evidence)', 'None)'),
            (0, 'order = prepared.order_before', 'order = None'),
            (0, 'continue', 'pass'),
        ]
        for index, old, new in mutations:
            sources = [decision, preparation, predicate]
            assert old in sources[index], (index, old)
            sources[index] = sources[index].replace(old, new)
            assert not contract.delegated_positive_kind_refusal(*sources), (index, old, new)
    ''')


def test_delegated_helpers_require_manifest_identity_and_remain_observation_only():
    _run_published_sdk('''
        import copy, json
        import pytest
        import stageprev_793_claim_contract as contract
        from prismabuild.pool import candidate_policy

        manifest = json.loads((Path(sys.argv[1]) / 'RUNTIME_VERSION.json').read_text())
        observed = contract.observe_current_claim_contract(manifest)
        assert observed['verified'], observed
        key = 'src/prismabuild/candidate_policy.py'
        assert observed['helper_source_sha256'][key] == manifest['files'][key]
        for absent in (False, True):
            changed = copy.deepcopy(manifest)
            if absent:
                del changed['files'][key]
            else:
                changed['files'][key] = '0' * 64
            refused = contract.observe_current_claim_contract(changed)
            assert not refused['identity_verified'], refused
            assert not refused['verified'], refused
            assert refused['claim_invoked'] is False
            assert refused['denial_synthesized'] is False

        def forbidden(*args, **kwargs):
            raise AssertionError('observer executed a scheduling helper')
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(candidate_policy, 'never_fits', forbidden)
            refused = contract.observe_current_claim_contract(manifest)
            assert not refused['verified'], refused
            assert not refused['identity_verified'], refused
            assert refused['claim_invoked'] is False
            assert refused['denial_synthesized'] is False
    ''')
