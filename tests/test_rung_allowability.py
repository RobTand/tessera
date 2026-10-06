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
        geom={"bits_per_256_weight_tile":{"numerator":q,"denominator":1},"alignment":{"lane_bits":[24],"lane_ends_on_word":[False],"half_bytes":[24],"half_copy":["8B tail"],"slot_words":8},"shared_memory":{"requested_bytes":2,"available_bytes":3,"fits":True},"register_pressure":{"REG":32,"STACK":0,"LOCAL":0,"SHARED":0},"decode_width":{"window_bits":14,"value_bits":8,"run_widths":[3],"word_stages":3,"superblock_rows":64,"k_split":1}}
        m={**cell,"measurement_status":"measured","kernel_time_us":10-q%2,"kernel_path":"native","geometry":geom,"evidence":evidence,"pass_times_us":[10-q%2,10-q%2],"measurement_build_id":"build"}
        quality={"measurement_status":"measured","source_kind":"actual_sampled_expert_weights","device":"cpu","anomaly_flags":[],"samples":[{"source_sha256":"actual","source_squared_norm":2.0,"relative_sse":.1,"exact_bytes":3}]}
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
        t=fixture(); t["rungs"][0]["anomaly_flags"]=["quality_unexplained"]; t["rungs"][0]["quality"]["anomaly_flags"]=["quality_unexplained"]
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

    def test_generated_timestamp_required_and_valid(self):
        for value in (None,"not-a-time","2026-10-06","2026-02-30T03:00:00Z","2026-10-06T03:00:00"):
            t=fixture()
            if value is None:t.pop('generated_at')
            else:t['generated_at']=value
            with self.assertRaises(ValueError):validate_table(t)

    def test_dominating_key_required_even_when_not_excluded(self):
        t=fixture();t['rungs'][0].pop('dominating_rung')
        with self.assertRaises(ValueError):validate_table(t)

    def test_pending_cell_required_structural_fields(self):
        t=fixture();t['table_status']='partial';r=t['rungs'][0]
        r.update(measurement_status='pending',supported=None,quality={})
        m=r['measurements'][0]
        m.update(measurement_status='pending',kernel_time_us=None,kernel_path=None,geometry=None,evidence={})
        self.assertIs(validate_table(t),t)
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],'wait')
        for field in ('measurement_status','kernel_time_us','kernel_path','geometry','evidence'):
            bad=copy.deepcopy(t);bad['rungs'][0]['measurements'][0].pop(field)
            with self.assertRaises(ValueError):validate_table(bad)

    def test_reader_full_roster_pending_and_unlisted_wait(self):
        t=fixture();t['scope'].update(rung_min=896,rung_max=898);t['table_status']='partial'
        t['rungs'][0]['rung']=896;t['rungs'][1]['rung']=897
        pending=copy.deepcopy(t['rungs'][0]);pending.update(rung=898,measurement_status='pending',supported=None,measurements=[],quality={})
        t['rungs'].append(pending)
        validate_table(t)
        for q in (898,899):self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=q)['status'],'wait')


    def test_boundary_overlap_uses_real_paired_witnesses(self):
        t=fixture();low,high=t['rungs']
        low.update(excluded=True,dominating_rung=769)
        a=copy.deepcopy(low['measurements'][0]);b=copy.deepcopy(high['measurements'][0])
        high['measurements'][0]['evidence']['comparison_id']='next-quantum'
        low['dominance_evidence']=[{'cell_id':a['cell_id'],'lower_time_us':a['kernel_time_us'],'higher_time_us':b['kernel_time_us'],'comparison_id':'paired','lower_measurement':a,'higher_measurement':b}]
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=768)['status'],'excluded')
        for field,value in [('measurement_build_id','different'),('shape_id','wrong'),('kernel_time_us',float('inf'))]:
            bad=copy.deepcopy(t);bad['rungs'][0]['dominance_evidence'][0]['higher_measurement'][field]=value
            with self.assertRaises(ValueError):validate_table(bad)


    def test_empty_measured_geometry_is_missing_not_admitted(self):
        for field in ("alignment","decode_width"):
            t=fixture();t["rungs"][0]["measurements"][0]["geometry"][field]={}
            with self.assertRaises(ValueError):validate_table(t)

    def test_quality_flags_must_match_row_flags(self):
        t=fixture();t['rungs'][0]['quality']['anomaly_flags']=['unexplained_quality']
        with self.assertRaises(ValueError):validate_table(t)

    def test_aggregate_must_match_paired_passes(self):
        t=fixture();t['rungs'][0]['measurements'][0]['kernel_time_us']=9.0
        t['rungs'][0]['measurements'][0]['pass_times_us']=[10.0,10.0]
        with self.assertRaises(ValueError):validate_table(t)



class GeometryHarvest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'/'t8r_speed'))
        import bench_rates, rung_allowability_table
        cls.rates,cls.harvest=bench_rates,rung_allowability_table

    def test_bf16_true_step_and_wide_boundaries(self):
        for q in (256,257,2048,2049,3584,3585,3840,3841,4095,4096):
            rate,fraction=self.rates.parse_case(f'q{q}','value')
            self.assertEqual(self.rates.q256_of(rate,fraction),q)
        for q in (255,4097):
            with self.assertRaises(ValueError):self.rates.parse_case(f'q{q}','value')
        with self.assertRaises(ValueError):self.rates.parse_case('q2049','e4m3')

    def test_bf16_projection_uses_actual_recipe_width_and_finite_table(self):
        import torch
        from tessera import routed_fused
        for q,width in ((3584,14),(3585,15),(3840,15),(3841,16),(4096,16)):
            rate,fraction=self.rates.parse_case(f'q{q}','value')
            p=self.rates.build_projection(routed_fused,1,512,256,rate,
                round(256*(fraction or 0)),41,torch.device('cpu'),False,bf16_table=True)
            self.assertEqual(p['window_bits'],width)
            self.assertEqual(p['table'].numel(),1<<width)
            self.assertTrue(bool(torch.isfinite(p['table'].view(torch.bfloat16)).all()))
            self.assertTrue(bool(((p['init']>=0)&(p['init']<(1<<width))).all()))

    def test_index_merge_preserves_t8_and_immutable_versions(self):
        index={'schema':'fleet.rung_allowability.index.v1','formats':{
            'TESSERA_E4M3_K1':{'kernel_builds':{'existing':{
                'current_version':9,'versions':{'9':{'path':'TESSERA_E4M3_K1/existing/v0009.json',
                'table_schema':'fleet.rung_allowability.v1','table_status':'complete'}}}}}}}
        before=copy.deepcopy(index['formats']['TESSERA_E4M3_K1'])
        table={'schema':'fleet.rung_allowability.v1','table_status':'partial'}
        self.harvest.merge_index(index,'TESSERA_BF16_K1','value',1,'TESSERA_BF16_K1/value/v0001.json',table)
        self.assertEqual(index['formats']['TESSERA_E4M3_K1'],before)
        with self.assertRaises(ValueError):
            self.harvest.merge_index(index,'TESSERA_BF16_K1','value',1,'TESSERA_BF16_K1/value/v0001.json',table)

    def test_actual_adapter_normalization_does_not_require_fused_symbols(self):
        from tessera.alphabet import E2M1_GRID,tuple_grid
        m=fixture()['rungs'][0]['measurements'][0]
        head={'kind':'dense','shape':'o_proj','q256':896,'body_kind':'TCQ','window_bits':0}
        cell={'F':{'median_ms':1,'timer':'graph'},'R':{'median_ms':1,'timer':'graph'},
              'ms':1,'bm':64,'kernel_path':'actual_span2','normalized_geometry':m['geometry']}
        data={'meta':{'pb_action':'actual','statistic':'paired'}}
        out=self.harvest.measurement('actual.json',data,head,cell,
            {'cell_id':'dense:o_proj:M1','kernel_kind':'dense','shape_id':'o_proj','M':1},
            'build',4096,4096,tuple_grid(E2M1_GRID,2))
        self.assertEqual(out['measurement_status'],'measured')
        self.assertIs(out['geometry'],cell['normalized_geometry'])
        self.assertEqual(out['kernel_path'],'actual_span2')



if __name__=='__main__': unittest.main()
