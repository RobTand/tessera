"""The canonical admission home is metadata-only and fail closed on bad evidence."""
import copy
import json
import subprocess
import sys
import unittest
from tessera.rung_allowability import admit_rung, validate_index, validate_table


def fixture():
    cell={"cell_id":"dense:o:M1","kernel_kind":"dense","shape_id":"o","M":1}
    build={"id":"build","source_commit":"abc","library_variant":"e4m3mma","architecture":"sm_121","activation_contract":"fp8"}
    rows=[]
    for q in (768,769):
        evidence={"comparison_id":"paired","paired_seed_contract":"same","timing_statistic":"F/R","timer":"graph"}
        geom={"bits_per_256_weight_tile":{"numerator":q,"denominator":1},"alignment":{},"shared_memory":{"requested_bytes":2,"available_bytes":3,"fits":True},"register_pressure":{"REG":32,"STACK":0,"LOCAL":0,"SHARED":0},"decode_width":{}}
        m={**cell,"measurement_status":"measured","kernel_time_us":10-q%2,"kernel_path":"native","geometry":geom,"evidence":evidence,"pass_times_us":[10,10],"measurement_build_id":"build"}
        quality={"measurement_status":"measured","source_kind":"actual_sampled_expert_weights","device":"cpu","samples":[{"source_sha256":"actual","source_squared_norm":2.0,"relative_sse":.1,"exact_bytes":3}]}
        rows.append({"rung":q,"measurement_status":"measured","supported":True,"anomaly_flags":[],"observations":[],"excluded":False,"dominating_rung":None,"measurements":[m],"quality":quality,"dominance_evidence":[],"lineage":{}})
    return {"schema":"fleet.rung_allowability.v1","table_version":1,"table_status":"complete","format":"TESSERA_E4M3_K1","generated_at":"2026-10-06T03:00:00Z","kernel_build":build,"scope":{"rung_min":768,"rung_max":769,"grid_step_q256":1,"grid_owner":"owner","required_cells":[cell]},"rungs":rows}


class Admission(unittest.TestCase):
    def test_metadata_only_import(self):
        subprocess.run([sys.executable,"-c","import sys; import tessera.rung_allowability; assert 'torch' not in sys.modules; assert not any(k.startswith('tessera.serving') for k in sys.modules)"],check=True)

    def test_valid_and_allowed(self):
        t=fixture()
        self.assertIs(validate_table(t),t)
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],'allow')

    def test_missing_scope_and_build_wait(self):
        for table, kwargs in [(None,{}),(fixture(),{'kernel_build_id':'other'}),(fixture(),{'scope':{'grid_step_q256':2}}),(fixture(),{'rung':770})]:
            args=dict(format='TESSERA_E4M3_K1',kernel_build_id='build',rung=768); args.update(kwargs)
            self.assertEqual(admit_rung(table,**args)['status'],'wait')

    def test_pending_wait_not_numeric(self):
        t=fixture(); t['table_status']='partial'; r=t['rungs'][0]; r['measurement_status']='pending'; r['supported']=None; r['measurements']=[]; r['quality']={}
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],'wait')

    def test_anomaly_hold_separate(self):
        t=fixture(); t['rungs'][0]['anomaly_flags']=['quality_unexplained']
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],'hold')

    def test_observations_do_not_exclude(self):
        t=fixture(); t['rungs'][0]['observations']=[{'issue':689,'missing_census':True},{'issue':690,'slowdown_ratio':2}]
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],'allow')

    def test_supported_failed_vocab(self):
        for status in ('unsupported','failed'):
            t=fixture(); r=t['rungs'][0]; r['measurement_status']=status; r['supported']=False; r['measurements']=[]; r['quality']={}
            self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],status)

    def test_adjacent_higher_dominance(self):
        t=fixture(); r=t['rungs'][0]; r['excluded']=True; r['dominating_rung']=769
        r['dominance_evidence']=[{'cell_id':'dense:o:M1','lower_time_us':10,'higher_time_us':9,'comparison_id':'paired'}]
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],'excluded')
        for mutate in [lambda x:x['rungs'][0].update(dominating_rung=770),lambda x:x['rungs'][1]['measurements'][0]['evidence'].update(comparison_id='other'),lambda x:x['rungs'][1]['measurements'][0].update(kernel_time_us=11),lambda x:x['rungs'][1].update(anomaly_flags=['bad'])]:
            bad=copy.deepcopy(t); mutate(bad)
            with self.assertRaises(ValueError): validate_table(bad)

    def test_invalid_numeric_coverage_quality_geometry(self):
        mutations=[lambda t:t['rungs'].pop(),lambda t:t['rungs'].append(copy.deepcopy(t['rungs'][0])),lambda t:t['rungs'][0]['measurements'][0].update(kernel_time_us=float('nan')),lambda t:t['rungs'][0]['measurements'][0].update(kernel_time_us=0),lambda t:t['rungs'][0].update(measurements=[]),lambda t:t['rungs'][0]['quality'].update(device='cuda'),lambda t:t['rungs'][0]['measurements'][0]['geometry'].update(register_pressure=None),lambda t:t['rungs'][0]['measurements'][0].update(measurement_build_id='other')]
        for mutate in mutations:
            bad=fixture(); mutate(bad)
            with self.assertRaises(ValueError):validate_table(bad)

    def test_index_current_and_safe_paths(self):
        index={'schema':'fleet.rung_allowability.index.v1','formats':{'TESSERA_E4M3_K1':{'kernel_builds':{'build':{'current_version':1,'versions':{'1':{'path':'TESSERA_E4M3_K1/build/v0001.json','table_schema':'fleet.rung_allowability.v1','table_status':'partial'}}}}}}}
        self.assertIs(validate_index(index),index)
        for path in ('../escape','/absolute','a/../b','a\\b','a//b','a/./b'):
            bad=copy.deepcopy(index); bad['formats']['TESSERA_E4M3_K1']['kernel_builds']['build']['versions']['1']['path']=path
            with self.assertRaises(ValueError):validate_index(bad)
        bad=copy.deepcopy(index);bad['formats']['TESSERA_E4M3_K1']['kernel_builds']['build']['current_version']=2
        with self.assertRaises(ValueError):validate_index(bad)


if __name__=='__main__': unittest.main()
