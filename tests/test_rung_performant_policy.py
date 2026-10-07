"""Scoped performance admission is not inherited numerical or serving evidence."""
import copy
import unittest

from test_rung_allowability import fixture_v2
from tessera.rung_allowability import (
    PERFORMANT_POLICY, admit_rung, geometry_class_identity, measured_geometry_classes,
    performant_rungs, rung_quality, rung_speed, scope_cell_ids, validate_table,
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


class PerformantPolicy(unittest.TestCase):
    def test_menu_is_structure_specific(self):
        self.assertEqual(performant_rungs('TESSERA_E4M3_K1', 'dense'), (768, 1024))
        self.assertEqual(performant_rungs('TESSERA_BF16_K1', 'routed'), tuple(range(256, 2049, 256)))
        self.assertEqual(performant_rungs('TESSERA_BF16_K1', 'dense'), tuple(range(256, 3585, 256)))
        self.assertEqual(performant_rungs('TESSERA_E2M1_K2', 'routed'), ())

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
