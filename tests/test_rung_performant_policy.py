"""Scoped performance admission is not inherited numerical or serving evidence."""
import copy
import unittest

from test_rung_allowability import fixture_v2
from tessera.rung_allowability import (
    PERFORMANT_POLICY, admit_rung, geometry_class_identity, measured_geometry_classes,
    performant_rungs, publication_scope, rung_quality, rung_speed, scope_cell_ids, validate_table,
)


def v3_fixture():
    table = fixture_v2()
    table['schema'] = 'fleet.rung_allowability.v3'
    table['performant_policy'] = dict(PERFORMANT_POLICY)
    table['scope']['shapes'] = [{'kernel_kind': 'dense', 'shape_id': 'o', 'rows': 512, 'columns': 256, 'mode': 2}]
    for row in table['rungs']:
        measurement = row['measurements'][0]
        measurement['evidence'].update(rows=512, columns=256, mode=2, routing='none')
        rates = [3] if row['rung'] == 768 else [3, 4]
        measurement['geometry']['decode_width']['run_widths'] = rates
        measurement['geometry']['alignment'].update(lane_bits=[8 * r for r in rates], half_bytes=[8 * r for r in rates],
            lane_ends_on_word=[8 * r % 32 == 0 for r in rates], half_copy=['copy' for _r in rates])
    table['geometry_classes'] = measured_geometry_classes(table)
    return table


def decide(table, rung=768, **kwargs):
    return admit_rung(table, format=table['format'], kernel_build_id=table['kernel_build']['id'], rung=rung, **kwargs)


def approved_t4_table():
    # Ground truth read from the approved table v0003 (SHA-256 35e1f829...dad3c6).
    # The source fix must record this same scope in E2M1_K2_PERFORMANT_MENU.
    dims = {'gate_up': (1024, 4096, 0, 'SwiGLU clipped at 10'),
            'down': (4096, 1024, 2, 'route-weighted BF16 down'),
            'o_proj': (4096, 4096, 2, 'BF16 linear output'),
            'q_b': (8192, 1536, 2, 'BF16 linear output')}
    from tessera.rung_allowability import E2M1_K2_PERFORMANT_MENU
    table = fixture_v2('tcq', 'native_tcq')
    table['schema'] = 'fleet.rung_allowability.v3'
    table['performant_policy'] = dict(PERFORMANT_POLICY)
    menu = E2M1_K2_PERFORMANT_MENU
    table['scope']['required_cells'] = [
        {'cell_id': cell_id, 'kernel_kind': 'routed' if cell_id.startswith('routed') else 'dense',
         'shape_id': cell_id.split(':')[1], 'M': int(cell_id.split(':')[2][1:])}
        if not cell_id.endswith(':recorded') else
        {'cell_id': cell_id, 'kernel_kind': 'routed', 'shape_id': cell_id.split(':')[1],
         'M': int(cell_id.split(':')[2][1:]), 'routing': 'recorded'}
        for cell_id in menu['cell_ids']]
    table['scope']['shapes'] = [{'kernel_kind': 'routed' if name in ('gate_up', 'down') else 'dense',
                                 'shape_id': name, 'rows': rows, 'columns': columns, 'mode': mode}
                                for name, (rows, columns, mode, _epilogue) in dims.items()]
    table['kernel_build']['id'] = menu['kernel_build_id']
    table['kernel_build']['activation_contract'] = menu['activation_contract']
    paths = {'routed': 'tessera.kernel_a4.a4_span2_grouped_gemm', 'dense': 'tessera.kernel_a4.a4_span2_gemm'}
    base = table['rungs'][0]['measurements'][0]
    row = table['rungs'][0]
    row['measurements'] = []
    for cell in table['scope']['required_cells']:
        measurement = copy.deepcopy(base)
        rows, columns, mode, epilogue = dims[cell['shape_id']]
        routing = 'recorded' if cell['cell_id'].endswith(':recorded') else ('balanced' if cell['kernel_kind'] == 'routed' else 'none')
        measurement.update(cell_id=cell['cell_id'], kernel_kind=cell['kernel_kind'],
                           shape_id=cell['shape_id'], M=cell['M'],
                           measurement_build_id=menu['kernel_build_id'],
                           kernel_path=paths[cell['kernel_kind']])
        measurement['evidence'].update(rows=rows, columns=columns, mode=mode, routing=routing,
                                       epilogue=epilogue, input_distribution=menu['activation_contract'])
        row['measurements'].append(measurement)
    table['geometry_classes'] = measured_geometry_classes(table)
    return table


class PerformantPolicy(unittest.TestCase):
    def test_menu_is_structure_specific(self):
        self.assertEqual(performant_rungs('TESSERA_E4M3_K1', 'dense'), (768, 896, 1024))
        self.assertEqual(performant_rungs('TESSERA_BF16_K1', 'routed'), tuple(sorted((*range(256, 2049, 256), 896))))
        self.assertEqual(performant_rungs('TESSERA_BF16_K1', 'dense'), tuple(sorted((*range(256, 3585, 256), 896))))
        self.assertEqual(performant_rungs('TESSERA_E2M1_K2', 'dense'), (896,))
        self.assertEqual(performant_rungs('TESSERA_E2M1_K2', 'routed'), (896,))

    def test_diagnostic_measurement_is_not_admission(self):
        table = v3_fixture()
        self.assertEqual(decide(table, 769)['status'], 'excluded')
        speed = rung_speed(table, rung=769, cell_id='dense:o:M1')
        self.assertEqual(speed['status'], 'measured')
        self.assertFalse(speed['menu_admitted'])
        self.assertFalse(speed['numerical_qualification_inherited'])
        self.assertFalse(speed['serving_qualification_inherited'])

    def test_actual_scope_can_allow_without_fake_aggregate_measured(self):
        table = v3_fixture()
        table['scope']['required_cells'].append({'cell_id': 'dense:o:M16', 'kernel_kind': 'dense', 'shape_id': 'o', 'M': 16})
        table['table_status'] = 'partial'
        for row in table['rungs']:
            row['measurement_status'] = 'pending'
        self.assertEqual(decide(table)['status'], 'wait')
        self.assertEqual(decide(table, cell_ids=['dense:o:M1'])['status'], 'allow')
        self.assertEqual(table['rungs'][0]['measurement_status'], 'pending')
        self.assertEqual(scope_cell_ids(table, kernel_kind='dense', rows=512, columns=256, M=1), ('dense:o:M1',))
        self.assertEqual(scope_cell_ids(table, kernel_kind='routed', rows=512, columns=256, M=1), ())

    def test_missing_shape_activation_recipe_and_M_wait(self):
        table = v3_fixture()
        for kwargs in ({'cell_ids': []}, {'cell_ids': ['missing']}, {'activation_contract': 'different'}, {'recipe': {'body': 'tcq'}}):
            self.assertEqual(decide(table, **kwargs)['status'], 'wait')
        with self.assertRaises(ValueError):
            scope_cell_ids(table, kernel_kind='dense', rows=512, columns=256, M=True)

    def test_quality_sample_reversal_does_not_admit_or_exclude(self):
        table = v3_fixture()
        table['rungs'][0]['quality']['samples'][0]['relative_sse'] = 999
        table['rungs'][1]['quality']['samples'][0]['relative_sse'] = 0.00001
        self.assertEqual(decide(table)['status'], 'allow')
        self.assertEqual(decide(table, 769)['status'], 'excluded')
        derived = rung_quality(896, lower_rung=768, upper_rung=1024, lower_value=0.1, upper_value=0.05)
        self.assertEqual(derived['status'], 'derived')
        self.assertAlmostEqual(derived['value'], 0.075)

    def test_class_and_actual_shape_mismatch_refuse(self):
        for mutation in ('class', 'shape', 'payload'):
            table = v3_fixture()
            if mutation == 'class':
                table['geometry_classes'][0]['identity']['shape'] = [1024, 256]
            elif mutation == 'shape':
                table['scope']['shapes'][0]['rows'] = 1024
            else:
                table['rungs'][0]['measurements'][0]['geometry']['decode_width']['value_bits'] = 16
            with self.assertRaises(ValueError):
                validate_table(table)

    def test_pair_arity_q896_is_pure_seven(self):
        table = fixture_v2('tcq', 'native_tcq')
        measurement = table['rungs'][0]['measurements'][0]
        measurement['evidence'].update(rows=512, columns=256)
        identity = geometry_class_identity(table, 896, measurement)
        self.assertEqual((identity['kind'], identity['arity'], identity['run_widths']), ('pure', 2, [7]))

    def test_legacy_semantics_and_new_query_do_not_mix(self):
        table = fixture_v2()
        self.assertEqual(decide(table)['status'], 'allow')
        self.assertEqual(decide(table, cell_ids=['dense:o:M1'])['reason'], 'scoped_policy_requires_v3_table')


    def test_class_inheritance_does_not_set_measurement_or_admission(self):
        table = v3_fixture()
        last = copy.deepcopy(table['rungs'][1]); last['rung'] = 771
        last['quality']['scope']['rung'] = 771
        last['measurements'][0].update(kernel_time_us=8, pass_times_us=[8, 8])
        missing = copy.deepcopy(last); missing.update(rung=770, measurement_status='pending', supported=None, measurements=[], quality={})
        table['rungs'] += [missing, last]
        table['scope']['rung_max'] = 771; table['table_status'] = 'partial'
        table['geometry_classes'] = measured_geometry_classes(table)
        identity = geometry_class_identity(table, 769, table['rungs'][1]['measurements'][0])
        result = rung_speed(table, rung=770, class_identity=identity)
        self.assertEqual(result['status'], 'inherited')
        self.assertFalse(result['menu_admitted'])
        self.assertEqual(missing['measurement_status'], 'pending')
        self.assertEqual(decide(table, 770)['status'], 'excluded')
        self.assertEqual(rung_speed(table, rung=1200, class_identity=identity)['status'], 'wait')
        self.assertEqual(rung_speed(table, rung=770, cell_id='other', class_identity=identity)['status'], 'wait')

    def test_present_partial_cell_waits_before_class_reconstruction(self):
        for state in ('pending', 'failed', 'unsupported'):
            table = v3_fixture()
            row = table['rungs'][1]
            cell = row['measurements'][0]
            identity = geometry_class_identity(table, row['rung'], cell)
            row.update(measurement_status=state, supported=None)
            cell.update(measurement_status=state, kernel_time_us=None, kernel_path=None,
                        evidence={}, pass_times_us=[])
            cell['geometry']['decode_width'].pop('run_widths')
            table['table_status'] = 'partial'
            table['geometry_classes'] = measured_geometry_classes(table)
            result = rung_speed(table, rung=row['rung'], class_identity=identity)
            self.assertEqual(result['status'], 'wait' if state == 'pending' else state)
            self.assertNotIn('kernel_time_us', result)

    def test_class_donors_never_supply_held_or_refused_times(self):
        for state in ('hold', 'failed', 'unsupported', 'unsupported_support'):
            table = v3_fixture()
            donor = table['rungs'][1]
            last = copy.deepcopy(donor)
            last['rung'] = 771
            last['quality']['scope']['rung'] = 771
            last['measurements'][0].update(kernel_time_us=8, pass_times_us=[8, 8])
            missing = copy.deepcopy(last)
            missing.update(rung=770, measurement_status='pending', supported=None, measurements=[], quality={})
            table['rungs'] += [missing, last]
            table['scope']['rung_max'] = 771
            table['table_status'] = 'partial'
            if state == 'hold':
                donor['anomaly_flags'] = donor['quality']['anomaly_flags'] = ['reader_correctness']
            elif state == 'unsupported_support':
                donor.update(measurement_status='pending', supported=False)
            else:
                donor.update(measurement_status=state, supported=False)
            table['geometry_classes'] = measured_geometry_classes(table)
            identity = geometry_class_identity(table, donor['rung'], donor['measurements'][0])
            result = rung_speed(table, rung=770, class_identity=identity)
            self.assertIn(result['status'], ('hold', 'wait', 'failed', 'unsupported'))
            self.assertNotIn('kernel_time_us', result)
            self.assertNotIn('observed_range_us', result)

    def test_measured_r896_uses_its_actual_scoped_cost(self):
        table = v3_fixture()
        row = table['rungs'][1]
        table['rungs'] = [row]
        table['scope'].update(rung_min=896, rung_max=896)
        row['rung'] = row['quality']['scope']['rung'] = 896
        cell = row['measurements'][0]
        cell.update(kernel_time_us=12.5, pass_times_us=[12.5, 12.5])
        for family, value_bits in (('TESSERA_E4M3_K1', 8), ('TESSERA_BF16_K1', 16)):
            table['format'] = family
            cell['geometry']['decode_width']['value_bits'] = value_bits
            table['geometry_classes'] = measured_geometry_classes(table)
            self.assertEqual(decide(table, 896, cell_ids=[cell['cell_id']])['status'], 'allow')
            speed = rung_speed(table, rung=896, cell_id=cell['cell_id'])
            self.assertEqual(speed['status'], 'measured')
            self.assertEqual(speed['measurement']['kernel_time_us'], 12.5)
            self.assertTrue(speed['menu_admitted'])
            self.assertFalse(speed['numerical_qualification_inherited'])
            self.assertFalse(speed['serving_qualification_inherited'])
            self.assertEqual(decide(table, 896, cell_ids=['missing'])['status'], 'wait')
            self.assertEqual(decide(table, 896, recipe={'body': 'tcq'})['status'], 'wait')
        row.update(measurement_status='pending', supported=None, measurements=[], quality={})
        table['table_status'] = 'partial'
        table['geometry_classes'] = []
        self.assertEqual(decide(table, 896)['status'], 'wait')
        self.assertEqual(rung_speed(table, rung=896, cell_id=cell['cell_id'])['status'], 'wait')


class TimingPublication(unittest.TestCase):
    def test_retained_samples_publish_descriptive_uncertainty(self):
        table = v3_fixture()
        cell = table['rungs'][0]['measurements'][0]
        cell['evidence'].update(action_key='original-action', paired_seed_contract='paired seed', timer='graph')
        cell['evidence']['F'] = {'samples_ms': [0.009, 0.01, 0.011], 'median_ms': 0.01}
        cell['evidence']['R'] = {'samples_ms': [0.009, 0.01, 0.012], 'median_ms': 0.01}
        cell.update(kernel_time_us=10, pass_times_us=[10, 10])
        result = rung_speed(table, rung=768, cell_id=cell['cell_id'])
        self.assertEqual(result['value_kind'], 'measured')
        self.assertEqual(result['timing']['passes']['F']['sample_count'], 3)
        self.assertEqual(result['timing']['passes']['F']['sample_range_us'], [9, 11])
        self.assertEqual(result['timing']['confidence_interval']['status'], 'unavailable')
        self.assertEqual(result['timing']['source']['action_key'], 'original-action')

    def test_missing_samples_are_not_fake_repetitions(self):
        table = v3_fixture()
        result = rung_speed(table, rung=768, cell_id='dense:o:M1')
        self.assertEqual(result['timing']['passes']['F']['status'], 'unavailable')
        self.assertEqual(result['timing']['passes']['F']['sample_count'], 0)
        self.assertEqual(result['timing']['confidence_interval']['status'], 'unavailable')

    def test_measured_samples_must_agree_with_their_pass_median(self):
        for samples in ([1, 2, 3], [1e308]):
            table = v3_fixture()
            cell = table['rungs'][0]['measurements'][0]
            cell['evidence']['F'] = {'samples_ms': samples}
            with self.assertRaisesRegex(ValueError, 'sample median'):
                rung_speed(table, rung=768, cell_id=cell['cell_id'])

    def test_publication_preserves_noise_observations_and_hard_refusals(self):
        from tessera.rung_allowability import publication_scope
        table = v3_fixture()
        table['rungs'][0]['quality']['adjacent_higher_raw_error_ratios'] = [1.2]
        table['rungs'][1]['anomaly_flags'] = table['rungs'][1]['quality']['anomaly_flags'] = ['reader_bounds']
        scope = publication_scope(table)
        self.assertEqual(scope['cells'][0]['rates'][0]['admission']['status'], 'allow')
        self.assertEqual(scope['cells'][0]['rates'][0]['timing']['status'], 'measured')
        self.assertEqual(scope['quality_observations'][0]['blocking'], False)
        self.assertEqual(scope['correctness_holds'][0]['flags'], ['reader_bounds'])
        self.assertEqual(scope['unavailable_rates'][0]['rung'], 640)
        self.assertEqual(scope['pricing_anchors'][0]['rung'], 1280)
        self.assertFalse(scope['pricing_anchors'][0]['performance_admitted'])

    def test_class_publication_binds_all_observed_times(self):
        from tessera.rung_allowability import TIMING_PUBLICATION
        table = v3_fixture()
        table['timing_publication'] = dict(TIMING_PUBLICATION)
        table['geometry_classes'] = measured_geometry_classes(table)
        timing = table['geometry_classes'][0]['timings'][0]
        self.assertEqual(timing['value_kind'], 'measured')
        table['geometry_classes'][0]['timings'][0]['kernel_time_us'] *= 2
        with self.assertRaisesRegex(ValueError, 'class identity'):
            validate_table(table)

    def test_pair_publication_binds_only_the_approved_build(self):
        from tessera.rung_allowability import publication_scope
        table = fixture_v2('tcq', 'native_tcq')
        table['schema'] = 'fleet.rung_allowability.v3'
        table['performant_policy'] = dict(PERFORMANT_POLICY)
        table['scope']['shapes'] = [{'kernel_kind': 'dense', 'shape_id': 'o', 'rows': 512, 'columns': 256, 'mode': 2}]
        for row in table['rungs']:
            row['measurements'][0]['evidence'].update(rows=512, columns=256)
        table['geometry_classes'] = measured_geometry_classes(table)
        scope = publication_scope(table)
        self.assertEqual(scope['qualified_menu'], {'dense': [], 'routed': []})
        self.assertEqual(scope['rate_semantics']['scalar_weights_per_code'], 2)
        self.assertEqual(scope['rate_semantics']['code_bits_per_symbol_denominator'], 128)
        self.assertEqual(scope['rate_semantics']['body_bits_per_scalar_denominator'], 256)
        self.assertFalse(scope['rate_semantics']['metadata_fees_included'])

    def test_approved_scope_admits_exact_cells_and_waits_elsewhere(self):
        table = approved_t4_table()
        scope = publication_scope(table)
        self.assertEqual(scope['qualified_menu'], {'dense': [896], 'routed': [896]})
        allowed = decide(table, 896)
        self.assertEqual(allowed['status'], 'allow')
        self.assertTrue(all(cell['status'] == 'allow' for cell in allowed['cells']))
        self.assertEqual(len(allowed['cells']), 20)
        foreign = copy.deepcopy(table)
        foreign['kernel_build']['id'] = 'other-build'
        for measurement in foreign['rungs'][0]['measurements']:
            measurement['measurement_build_id'] = 'other-build'
        foreign['geometry_classes'] = measured_geometry_classes(foreign)
        refused = decide(foreign, 896)
        self.assertEqual(refused['status'], 'wait')
        self.assertTrue(all(cell['reason'] == 'performance_admission_not_established' for cell in refused['cells']))
        self.assertEqual(decide(table, 768)['cells'][0]['reason'], 'performance_admission_not_established')
        held = copy.deepcopy(table)
        held['rungs'][0]['anomaly_flags'] = held['rungs'][0]['quality']['anomaly_flags'] = ['reader_bounds']
        self.assertEqual(decide(held, 896)['status'], 'hold')
        failed = copy.deepcopy(table)
        failed['rungs'][0].update(measurement_status='failed', supported=None, measurements=[], quality={})
        failed['geometry_classes'] = measured_geometry_classes(failed)
        self.assertEqual(decide(failed, 896)['status'], 'wait')

    def test_approved_semantic_scope_binds_every_field(self):
        from tessera.rung_allowability import E2M1_K2_PERFORMANT_MENU
        menu = E2M1_K2_PERFORMANT_MENU
        table = approved_t4_table()
        self.assertEqual(decide(table, 896)['status'], 'allow')
        self.assertEqual(publication_scope(table)['qualified_menu'], {'dense': [896], 'routed': [896]})
        cell_id = 'dense:o_proj:M1'
        cases = [
            ('routing', {'routing': 'shuffled'}, 'unmeasured_shape_or_M_scope'),
            ('mode', {'mode': 99}, 'unmeasured_execution_scope'),
            ('epilogue', {'epilogue': 'other epilogue'}, 'unmeasured_execution_scope'),
            ('kernel_path', {'kernel_path': 'tessera.kernel_a4.other'}, 'unmeasured_execution_scope'),
            ('input_distribution', {'input_distribution': 'other contract'}, 'unmeasured_activation_scope'),
            ('recipe', {'recipe': dict(menu['recipe'], body='window')}, 'unmeasured_recipe_scope'),
        ]
        for name, mutation, reason in cases:
            with self.subTest(field=name):
                mutated = copy.deepcopy(table)
                target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
                for key, value in mutation.items():
                    if key == 'kernel_path':
                        target[key] = value
                    elif key == 'recipe':
                        target['geometry']['recipe'] = value
                    else:
                        target['evidence'][key] = value
                mutated['geometry_classes'] = measured_geometry_classes(mutated)
                result = decide(mutated, 896, cell_ids=[cell_id])
                self.assertEqual(result['cells'][0]['status'], 'wait', name)
                self.assertEqual(result['cells'][0]['reason'], reason, name)
        with self.subTest(field='dims'):
            mutated = copy.deepcopy(table)
            target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            shape = next(s for s in mutated['scope']['shapes'] if s['shape_id'] == target['shape_id'])
            shape.update(rows=1, columns=1)
            for measurement in mutated['rungs'][0]['measurements']:
                if measurement['shape_id'] == target['shape_id']:
                    measurement['evidence'].update(rows=1, columns=1)
            mutated['geometry_classes'] = measured_geometry_classes(mutated)
            result = decide(mutated, 896)
            self.assertEqual(result['status'], 'wait')
            self.assertTrue(all(cell['reason'] == 'performance_admission_not_established'
                                for cell in result['cells']))
        with self.subTest(field='decoder'):
            mutated = copy.deepcopy(table)
            target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            target['geometry']['decoder_kind'] = 'fused_window'
            with self.assertRaises(ValueError):
                validate_table(mutated)
        with self.subTest(field='extra_M'):
            mutated = copy.deepcopy(table)
            template = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            extra = copy.deepcopy(template)
            extra.update(cell_id='dense:o_proj:M7', M=7, measurement_build_id=menu['kernel_build_id'])
            mutated['scope']['required_cells'].append(
                {'cell_id': 'dense:o_proj:M7', 'kernel_kind': 'dense', 'shape_id': 'o_proj', 'M': 7})
            mutated['rungs'][0]['measurements'].append(extra)
            mutated['geometry_classes'] = measured_geometry_classes(mutated)
            result = decide(mutated, 896)
            self.assertEqual(result['status'], 'wait')
            self.assertTrue(all(cell['reason'] == 'performance_admission_not_established'
                                for cell in result['cells']))
        with self.subTest(field='shape_dims'):
            mutated = copy.deepcopy(table)
            shape = next(s for s in mutated['scope']['shapes'] if s['shape_id'] == 'o_proj')
            shape.update(rows=1, columns=1)
            for measurement in mutated['rungs'][0]['measurements']:
                if measurement['shape_id'] == 'o_proj':
                    measurement['evidence'].update(rows=1, columns=1)
            mutated['geometry_classes'] = measured_geometry_classes(mutated)
            result = decide(mutated, 896)
            self.assertEqual(result['status'], 'wait')
            self.assertTrue(all(cell['reason'] == 'performance_admission_not_established'
                                for cell in result['cells']))
        with self.subTest(field='absent_cell'):
            absent = copy.deepcopy(table)
            absent['scope']['required_cells'] = [cell for cell in absent['scope']['required_cells']
                                                 if cell['cell_id'] != cell_id]
            absent['rungs'][0]['measurements'] = [m for m in absent['rungs'][0]['measurements']
                                                  if m['cell_id'] != cell_id]
            absent['geometry_classes'] = measured_geometry_classes(absent)
            self.assertEqual(publication_scope(absent)['qualified_menu'], {'dense': [], 'routed': []})
            missing = decide(absent, 896, cell_ids=[cell_id])
            self.assertEqual(missing['reason'], 'unmeasured_shape_or_M_scope')
    def test_approved_cell_records_bind_M_shape_and_kind(self):
        from tessera.rung_allowability import E2M1_K2_PERFORMANT_MENU
        menu = E2M1_K2_PERFORMANT_MENU
        table = approved_t4_table()
        cell_id = 'dense:o_proj:M1'
        with self.subTest(field='substituted_M'):
            mutated = copy.deepcopy(table)
            declared = next(c for c in mutated['scope']['required_cells'] if c['cell_id'] == cell_id)
            declared['M'] = 16
            target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            target['M'] = 16
            mutated['geometry_classes'] = measured_geometry_classes(mutated)
            result = decide(mutated, 896, cell_ids=[cell_id])
            self.assertEqual(result['cells'][0]['status'], 'wait')
            self.assertEqual(result['cells'][0]['reason'], 'performance_admission_not_established')
            self.assertEqual(publication_scope(mutated)['qualified_menu'], {'dense': [], 'routed': []})
        with self.subTest(field='substituted_shape'):
            mutated = copy.deepcopy(table)
            declared = next(c for c in mutated['scope']['required_cells'] if c['cell_id'] == cell_id)
            declared['shape_id'] = 'q_b'
            target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            target['shape_id'] = 'q_b'
            target['evidence'].update(rows=8192, columns=1536)
            mutated['geometry_classes'] = measured_geometry_classes(mutated)
            result = decide(mutated, 896, cell_ids=[cell_id])
            self.assertEqual(result['cells'][0]['status'], 'wait')
            self.assertEqual(result['cells'][0]['reason'], 'performance_admission_not_established')
            self.assertEqual(publication_scope(mutated)['qualified_menu'], {'dense': [], 'routed': []})
        with self.subTest(field='substituted_kind'):
            mutated = copy.deepcopy(table)
            mutated['scope']['shapes'].append(
                {'kernel_kind': 'routed', 'shape_id': 'o_proj', 'rows': 4096, 'columns': 4096, 'mode': 2})
            declared = next(c for c in mutated['scope']['required_cells'] if c['cell_id'] == cell_id)
            declared['kernel_kind'] = 'routed'
            target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            target['kernel_kind'] = 'routed'
            target['kernel_path'] = 'tessera.kernel_a4.a4_span2_grouped_gemm'
            target['evidence']['routing'] = 'balanced'
            mutated['geometry_classes'] = measured_geometry_classes(mutated)
            result = decide(mutated, 896, cell_ids=[cell_id])
            self.assertEqual(result['cells'][0]['status'], 'wait')
            self.assertEqual(result['cells'][0]['reason'], 'performance_admission_not_established')
            self.assertEqual(publication_scope(mutated)['qualified_menu'], {'dense': [], 'routed': []})
        with self.subTest(field='substituted_measurement_M'):
            mutated = copy.deepcopy(table)
            target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            target['M'] = 16
            with self.assertRaisesRegex(ValueError, 'unknown/duplicate measurement cell'):
                validate_table(mutated)
        with self.subTest(field='substituted_measurement_shape'):
            mutated = copy.deepcopy(table)
            target = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == cell_id)
            target['shape_id'] = 'q_b'
            with self.assertRaisesRegex(ValueError, 'unknown/duplicate measurement cell'):
                validate_table(mutated)
        with self.subTest(field='swapped_measurement_identity'):
            mutated = copy.deepcopy(table)
            first = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == 'dense:o_proj:M1')
            second = next(m for m in mutated['rungs'][0]['measurements'] if m['cell_id'] == 'dense:o_proj:M16')
            first['M'], second['M'] = second['M'], first['M']
            with self.assertRaisesRegex(ValueError, 'unknown/duplicate measurement cell'):
                validate_table(mutated)
        with self.subTest(field='approved_record_admits'):
            result = decide(table, 896, cell_ids=[cell_id])
            self.assertEqual(result['cells'][0]['status'], 'allow')
            self.assertEqual(result['cells'][0]['reason'], 'measured_performant_scope')


    def test_release_units_link_exact_rank_shapes_without_qualification(self):
        from tessera.rung_allowability import publication_scope
        units = {'release': 'retained-release', 'config': {}, 'units': [
            {'name': 'layer.0.mlp.gate', 'category': 'dense_mlp', 'role': 'gate', 'shape': [512, 512],
             'tensor_parallel_shapes': {'1': [512, 512], '2': [512, 256]}}]}
        scope = publication_scope(v3_fixture(), unit_inventory=units)
        coverage = scope['release_unit_coverage'][0]['coverage']
        self.assertEqual(coverage[0]['status'], 'wait')
        self.assertEqual(coverage[1]['cell_ids'], ['dense:o:M1'])
        self.assertFalse(scope['native_qualification_inherited'])
        units['units'][0]['tensor_parallel_shapes'] = {}
        missing = publication_scope(v3_fixture(), unit_inventory=units)['release_unit_coverage'][0]
        self.assertEqual(missing['geometry_status'], 'wait')
        self.assertEqual(missing['coverage'], [])

    def test_all_class_spots_bound_the_derived_value(self):
        table = v3_fixture()
        donor = table['rungs'][1]
        for q, time in ((770, 30), (771, 10), (772, 11), (773, 12)):
            row = copy.deepcopy(donor)
            row['rung'] = row['quality']['scope']['rung'] = q
            row['measurements'][0].update(kernel_time_us=time, pass_times_us=[time, time])
            table['rungs'].append(row)
        missing = copy.deepcopy(donor)
        missing.update(rung=774, measurement_status='pending', supported=None, measurements=[], quality={})
        table['rungs'].append(missing)
        table['scope']['rung_max'] = 774
        table['table_status'] = 'partial'
        table['geometry_classes'] = measured_geometry_classes(table)
        identity = geometry_class_identity(table, 769, donor['measurements'][0])
        result = rung_speed(table, rung=774, class_identity=identity)
        self.assertEqual(result['kernel_time_us'], 30)
        self.assertEqual(result['value_kind'], 'derived')
        self.assertEqual(result['anchors'], [769, 770, 771, 772, 773])
        self.assertFalse(result['menu_admitted'])

    def test_candidate_table_digest_is_a_byte_integrity_refusal(self):
        import tempfile
        import json
        import importlib.util
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location('publication_producer', root/'experiments/t8r_speed/rung_allowability_table.py')
        producer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(producer)
        table = v3_fixture()
        record = {'path': 'table.json', 'table_schema': table['schema'], 'table_status': table['table_status'], 'sha256': '0' * 64}
        index = {'schema': 'fleet.rung_allowability.index.v2', 'formats': {table['format']: {'kernel_builds': {
            table['kernel_build']['id']: {'current_version': 1, 'versions': {'1': record}}}}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            active = json.dumps(index)
            (path/'index.json').write_text(active)
            (path/'index.v3-candidate.json').write_text(active)
            (path/'table.json').write_text(json.dumps(table))
            with self.assertRaisesRegex(ValueError, 'bytes differ from their digest'):
                producer.activate_published_index(path, 'index.v3-candidate.json')
            self.assertEqual((path/'index.json').read_text(), active)

    def test_supplier_prose_is_not_a_new_identity_gate(self):
        import tempfile
        import json
        import subprocess
        import sys
        from pathlib import Path
        from test_rung_allowability import CHILD_ENV
        repo = Path(__file__).resolve().parents[1]
        # The native pair fixture already carries the physical two-scalar grammar.
        table = fixture_v2('tcq', 'native_tcq')
        table['evidence'] = {}
        table['scope']['shapes'] = [{'kernel_kind': 'dense', 'shape_id': 'o', 'rows': 512, 'columns': 256, 'mode': 2}]
        for row in table['rungs']:
            row['measurements'][0]['evidence'].update(rows=512, columns=256)
        packet = {'canonical_producer': 'paired-value-owner',
                  'paired_value_semantics': {'arity': 2, 'one_code': 'One code encodes two scalar weights.',
                                             'code_bits_per_symbol': 7, 'body_bits_per_scalar_weight': 3.5,
                                             'metadata_fees_included': False},
                  'qualified_class_menu': {'dense': [896], 'routed': [896]}, 'missing_evidence': {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'table.json').write_text(json.dumps(table))
            (root/'packet.json').write_text(json.dumps(packet))
            command = [sys.executable, str(repo/'experiments/t8r_speed/rung_allowability_table.py'),
                       '--input-table', str(root/'table.json'), '--paired-value-packet', str(root/'packet.json'),
                       '--root', str(root), '--out', str(root/'out'), '--version', '2',
                       '--schema', str(repo/'docs/schema/allowable-rung-table.v3.schema.json'),
                       '--index-schema', str(repo/'docs/schema/index.v2.schema.json')]
            result = subprocess.run(command, env=CHILD_ENV, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class SupplierRevision(unittest.TestCase):
    def test_metadata_increment_uses_pure_grid_owner(self):
        import json
        import subprocess
        import sys
        from test_rung_allowability import CHILD_ENV, SRC
        for version in ('fleet.rung_allowability.v2', 'fleet.rung_allowability.v3'):
            table = v3_fixture()
            table['schema'] = version
            table['evidence'] = {}
            program = (
                'import json,sys; '
                'sys.path.insert(0, "experiments/t8r_speed"); '
                'from rung_allowability_table import performance_increment; '
                'table=json.loads(sys.argv[1]); '
                'result=performance_increment(table, [], 2); '
                'assert result["schema"] == "fleet.rung_allowability.v3"; '
                'assert "torch" not in sys.modules; '
                'assert "tessera.control" not in sys.modules'
            )
            result = subprocess.run([sys.executable, '-c', program, json.dumps(table)],
                                    cwd=SRC.parent, env=CHILD_ENV, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_routed_release_unit_never_matches_dense_cell(self):
        from tessera.rung_allowability import publication_scope
        table = v3_fixture()
        unit = {'name': 'expert.gate', 'kernel_kind': 'routed', 'category': 'routed_moe',
                'tensor_parallel_shapes': {'2': [512, 256]}}
        inventory = {'units': [unit]}
        unmatched = publication_scope(table, unit_inventory=inventory)['release_unit_coverage'][0]['coverage'][0]
        self.assertEqual(unmatched['status'], 'wait')
        self.assertEqual(unmatched['cell_ids'], [])
        table['scope']['required_cells'].append({'cell_id': 'routed:o:M1', 'kernel_kind': 'routed', 'shape_id': 'o', 'M': 1})
        table['scope']['shapes'].append({'kernel_kind': 'routed', 'shape_id': 'o', 'rows': 512, 'columns': 256, 'mode': 2})
        for row in table['rungs']:
            cell = copy.deepcopy(row['measurements'][0])
            cell.update(cell_id='routed:o:M1', kernel_kind='routed')
            cell['evidence'].update(routing='balanced', epilogue='route-weighted BF16 down')
            row['measurements'].append(cell)
        table['geometry_classes'] = measured_geometry_classes(table)
        matched = publication_scope(table, unit_inventory=inventory)['release_unit_coverage'][0]['coverage'][0]
        self.assertEqual(matched['cell_ids'], ['routed:o:M1'])
        unit.pop('kernel_kind')
        unit['structure'] = 'routed_moe'
        mapped = publication_scope(table, unit_inventory=inventory)['release_unit_coverage'][0]['coverage'][0]
        self.assertEqual(mapped['cell_ids'], ['routed:o:M1'])

    def test_missing_rank_shape_field_returns_wait(self):
        from tessera.rung_allowability import publication_scope
        unit = {'name': 'shared.gate', 'kernel_kind': 'dense'}
        result = publication_scope(v3_fixture(), unit_inventory={'units': [unit]})['release_unit_coverage'][0]
        self.assertEqual(result['geometry_status'], 'wait')
        self.assertEqual(result['coverage'], [])
        self.assertEqual(result['unavailable_reason'], 'The retained inventory supplies no tensor-parallel shape.')

