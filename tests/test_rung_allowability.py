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
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            # bench_rates imports torch at module level; the hosted bytes-only run has none.
            raise unittest.SkipTest('torch is required by experiments/t8r_speed/bench_rates.py')
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'/'t8r_speed'))
        import bench_rates, rung_allowability_table
        cls.rates,cls.harvest=bench_rates,rung_allowability_table

    def test_increment_reader_findings_preserve_correctness_holds(self):
        import tempfile
        from pathlib import Path
        from test_rung_performant_policy import v3_fixture
        table = v3_fixture()
        table['evidence'] = {}
        table['kernel_build']['metadata'] = {'observed_signature': {'kernel_sha': 'reader-source'}}
        table['rungs'][1]['anomaly_flags'] = ['prior_hold']
        table['rungs'][1]['quality']['anomaly_flags'] = ['prior_hold']
        prior = {'kind': 'reader_correctness_finding', 'blocking': True, 'finding': {'anomaly_flag': 'prior_hold'}}
        table['rungs'][1]['observations'].append(prior)
        finding = {'format': table['format'], 'kernel_sha': 'reader-source', 'anomaly_flag': 'reader_wrong'}
        findings = [finding, dict(finding, kernel_sha='other-source', anomaly_flag='other_source'),
                    dict(finding, format='TESSERA_BF16_K1', anomaly_flag='other_family')]
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'input.json').write_text(json.dumps(table))
            (root / 'findings.json').write_text(json.dumps(findings))
            command = [sys.executable, str(repo / 'experiments/t8r_speed/rung_allowability_table.py'),
                       '--input-table', str(root / 'input.json'), '--reader-findings', str(root / 'findings.json'),
                       '--root', str(root), '--out', str(root / 'out'), '--version', '2',
                       '--schema', str(repo / 'docs/schema/allowable-rung-table.v3.schema.json'),
                       '--index-schema', str(repo / 'docs/schema/index.v2.schema.json')]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            actual = json.loads((root / 'out/table.json').read_text())
        decision = admit_rung(actual, format=table['format'], kernel_build_id='build', rung=768, cell_ids=['dense:o:M1'])
        self.assertEqual(decision['status'], 'hold')
        self.assertEqual(actual['rungs'][0]['anomaly_flags'], ['reader_wrong'])
        self.assertEqual(actual['rungs'][1]['anomaly_flags'], ['prior_hold', 'reader_wrong'])
        self.assertIn(prior, actual['rungs'][1]['observations'])
        for row in actual['rungs']:
            self.assertEqual(row['quality']['anomaly_flags'], row['anomaly_flags'])
            self.assertIn({'kind': 'reader_correctness_finding', 'blocking': True, 'exclusion_basis': False,
                           'finding': finding}, row['observations'])

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
            self.harvest.merge_index(index,'TESSERA_BF16_K1','value',1,'TESSERA_BF16_K1/value/changed.json',table)

    def test_select_identical_published_version_preserves_history(self):
        index={'schema':'fleet.rung_allowability.index.v1','formats':{'TESSERA_E4M3_K1':{'kernel_builds':{'old':{'current_version':9,'versions':{'9':{'path':'TESSERA_E4M3_K1/old/v0009.json','table_schema':'fleet.rung_allowability.v1','table_status':'complete'}}}}}}}
        prior=copy.deepcopy(index['formats']['TESSERA_E4M3_K1'])
        table={'schema':'fleet.rung_allowability.v2','table_status':'partial'}
        args=(index,'TESSERA_BF16_K1','value',2,'TESSERA_BF16_K1/value/v0002.json',table)
        self.harvest.merge_index(*args)
        self.harvest.merge_index(*args)
        self.assertEqual(index['formats']['TESSERA_E4M3_K1'],prior)
        with self.assertRaises(ValueError):
            self.harvest.merge_index(index,'TESSERA_BF16_K1','value',2,'TESSERA_BF16_K1/value/changed.json',table)


    def test_activation_uses_staged_candidate_and_keeps_immutable_history(self):
        import tempfile,hashlib
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);published=root/'tables';published.mkdir()
            versions={}
            hashes={}
            for version in (1,2,3):
                t=fixture();t['table_version']=version
                relative=f'TESSERA_E4M3_K1/build/v{version:04d}.json'
                path=published/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(t));hashes[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
                versions[str(version)]={'path':relative,'table_schema':t['schema'],'table_status':t['table_status']}
            current={'schema':'fleet.rung_allowability.index.v1','formats':{'TESSERA_E4M3_K1':{'kernel_builds':{'build':{'current_version':1,'versions':{k:versions[k] for k in ('1','3')}}}}}}
            candidate=copy.deepcopy(current);candidate['schema']='fleet.rung_allowability.index.v2'
            candidate['formats']['TESSERA_E4M3_K1']['kernel_builds']['build']={'current_version':2,'versions':{k:versions[k] for k in ('1','2')}}
            (published/'index.json').write_text(json.dumps(current));(published/'index.v2-candidate.json').write_text(json.dumps(candidate))
            script=Path(__file__).resolve().parents[1]/'experiments'/'t8r_speed'/'rung_allowability_table.py'
            command=[sys.executable,str(script),'--activate-index','--publish-root',str(published),
                     '--root',str(root/'must-not-reharvest'),'--schema',str(root/'unused-schema'),
                     '--index-schema',str(root/'unused-index-schema'),'--out',str(root/'report'),
                     '--version','2','--format','TESSERA_E4M3_K1']
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            selected=json.loads((published/'index.json').read_text())['formats']['TESSERA_E4M3_K1']['kernel_builds']['build']
            self.assertEqual(selected['current_version'],2)
            self.assertEqual(set(selected['versions']),{'1','2','3'})
            self.assertEqual(json.loads((published/'index.v1-history.json').read_text()),current)
            self.assertEqual(subprocess.run(command,capture_output=True,text=True).returncode,0)
            for relative,digest in hashes.items():self.assertEqual(hashlib.sha256((published/relative).read_bytes()).hexdigest(),digest)





def fixture_v2(body='window',decoder='fused_window'):
    t=fixture()
    t['schema']='fleet.rung_allowability.v2'
    for row in t['rungs']:
        g=row['measurements'][0]['geometry']
        g.update(body_kind=body,decoder_kind=decoder,decoder_owner='tessera.routed_fused',
                 execution_scope='raw_packed_window',word_ring={'kind':'staged','owner':'tessera.routed_fused'})
        g['shared_memory']['kind']='used'
        g['register_pressure']['compiler']='cuda_cuobjdump'
    if body=='tcq':
        t['format']='TESSERA_E2M1_K2';t['scope'].update(rung_min=896,rung_max=896)
        t['rungs']=t['rungs'][:1];t['rungs'][0]['rung']=896
        g=t['rungs'][0]['measurements'][0]['geometry']
        g.update(decoder_owner='tessera.kernel_a4',execution_scope='native_tcq_decode_gemm',
                 word_ring={'kind':'none','owner':'tessera.kernel_a4'})
        g['decode_width']={'window_bits':0,'word_stages':None,'value_bits':4,'arity':2,
            'run_widths':[7],'memory':6,'span':2,'history_lookup_bits':7,'label_lut_entries':128,
            'block_m':64,'block_n':64,'block_k':128,'mma_k':64,'scale_group':16}
        # Recorded native o_proj census from cff24a0e7d05; not a synthetic two-plane proxy.
        g['alignment']={"rate":7,"arity":2,"span":2,"plane_shapes":{"select":[528392],"label":[1048576],"point":[6291456],"nibbles":[524288],"lut_bytes":[16],"label_lut":[128],"code_nibbles":[256]},"plane_bytes":{"select":528392,"label":1048576,"point":6291456,"nibbles":524288,"lut_bytes":16,"label_lut":512,"code_nibbles":256},"kind":"tcq_planes","owner":"tessera.compact_prep.prepare_span2_compact","slot_words":None,"plane_element_bytes":{"select":1,"label":1,"point":1,"nibbles":1,"lut_bytes":1,"label_lut":4,"code_nibbles":1}}
        g['shared_memory']={'kind':'used','requested_bytes':12800,'available_bytes':101376,'fits':True}
        g['register_pressure']={'compiler':'triton_compiled_kernel','REG':196,'SPILLS':0,
            'STACK':None,'LOCAL':None,'SHARED':12800,'compiler_symbol':'_a4_span2_gemm_kernel'}
    for row in t['rungs']:
        cfg={'body':body,'span':2 if body=='tcq' else 1,'plane':'lut16' if body=='tcq' else 'channel',
             'window_bits':0 if body=='tcq' else 14,'seed':0,'sigma':None,'channel_sigma':None if body=='tcq' else 1.0}
        row['measurements'][0]['geometry']['recipe']=copy.deepcopy(cfg)
        row['quality']['scope']={'format':t['format'],'grid':'E2M1x2' if body=='tcq' else 'E4M3',
            'arity':2 if body=='tcq' else 1,'rung':row['rung'],'recipe':cfg,'kernel_kinds':['dense','routed'],
            'owner':'tessera.export.encode_linear'}

    return t


class BodyAwareGrammar(unittest.TestCase):
    def test_empty_required_native_input_planes_refuse(self):
        for name in ('select','label','nibbles'):
            t=fixture_v2('tcq','native_tcq');a=t['rungs'][0]['measurements'][0]['geometry']['alignment']
            a['plane_shapes'][name]=[0];a['plane_bytes'][name]=0
            with self.assertRaises(ValueError):validate_table(t)

    def test_nonempty_point_for_zero_width_field_refuses(self):
        t=fixture_v2('tcq','native_tcq');g=t['rungs'][0]['measurements'][0]['geometry']
        g['decode_width']['run_widths']=[1]
        g['alignment']['plane_shapes']['code_nibbles']=[4];g['alignment']['plane_bytes']['code_nibbles']=4
        with self.assertRaises(ValueError):validate_table(t)


    def test_quality_scope_missing_or_wrong_family_refuses(self):
        for change in ('missing','family','grid','arity','recipe'):
            t=fixture_v2('tcq','native_tcq');s=t['rungs'][0]['quality']['scope']
            if change=='missing':t['rungs'][0]['quality'].pop('scope')
            elif change=='family':s['format']='TESSERA_BF16_K1'
            elif change=='grid':s['grid']='BF16'
            elif change=='arity':s['arity']=1
            else:s['recipe']['body']='window'
            with self.assertRaises(ValueError):validate_table(t)


    def test_native_plane_census_missing_real_input_refuses(self):
        for name in ('select','label','point','nibbles','lut_bytes','label_lut','code_nibbles'):
            t=fixture_v2('tcq','native_tcq');a=t['rungs'][0]['measurements'][0]['geometry']['alignment']
            for field in ('plane_shapes','plane_bytes','plane_element_bytes'):a[field].pop(name)
            with self.assertRaises(ValueError):validate_table(t)

    def test_native_plane_byte_count_matches_actual_tensor_width(self):
        t=fixture_v2('tcq','native_tcq')
        t['rungs'][0]['measurements'][0]['geometry']['alignment']['plane_bytes']['label_lut']+=1
        with self.assertRaises(ValueError):validate_table(t)


    def test_actual_zero_width_point_has_zero_bytes_not_a_placeholder(self):
        t=fixture_v2('tcq','native_tcq');row=t['rungs'][0];row['rung']=128
        t['scope'].update(rung_min=128,rung_max=128)
        cell=row['measurements'][0];key=t['scope']['required_cells'][0]
        cell.update(cell_id='routed:gate_up:M1',kernel_kind='routed',shape_id='gate_up')
        key.update(cell_id='routed:gate_up:M1',kernel_kind='routed',shape_id='gate_up')
        g=cell['geometry'];g['decode_width']['run_widths']=[1]
        g['alignment']['plane_shapes']['point']=[0];g['alignment']['plane_bytes']['point']=0
        g['alignment']['plane_shapes']['code_nibbles']=[4];g['alignment']['plane_bytes']['code_nibbles']=4
        row['quality']['scope'].update(rung=128,kernel_kinds=['routed'])
        validate_table(t)
        g['alignment']['plane_bytes']['point']=1
        with self.assertRaises(ValueError):validate_table(t)


    def test_valid_native_tcq_and_explicit_v1_history(self):
        t=fixture_v2('tcq','native_tcq')
        self.assertIs(validate_table(t),t)
        self.assertEqual(admit_rung(t,format=t['format'],kernel_build_id='build',rung=896)['status'],'allow')
        legacy=fixture();self.assertIs(validate_table(legacy),legacy)

    def test_window_zero_remains_invalid_in_both_versions(self):
        for t in (fixture(),fixture_v2()):
            t['rungs'][0]['measurements'][0]['geometry']['decode_width']['window_bits']=0
            with self.assertRaises(ValueError):validate_table(t)

    def test_unknown_missing_body_or_owner_refuses(self):
        for field,value in (('body_kind','unknown'),('body_kind',None),('decoder_owner',None),('word_ring',{})):
            t=fixture_v2('tcq','native_tcq');g=t['rungs'][0]['measurements'][0]['geometry']
            if value is None:g.pop(field)
            else:g[field]=value
            with self.assertRaises(ValueError):validate_table(t)

    def test_native_tcq_scope_and_history_are_real_facts(self):
        for mutate in (lambda g:g.update(execution_scope='raw_packed_window'),
                       lambda g:g['decode_width'].update(window_bits=14),
                       lambda g:g['decode_width'].update(history_lookup_bits=6),
                       lambda g:g['decode_width'].update(label_lut_entries=64),
                       lambda g:g['alignment'].update(slot_words=8)):
            t=fixture_v2('tcq','native_tcq');mutate(t['rungs'][0]['measurements'][0]['geometry'])
            with self.assertRaises(ValueError):validate_table(t)

    def test_v2_witness_preserves_scope_and_finite_paired_mean(self):
        t=fixture_v2();low,high=t['rungs'];low.update(excluded=True,dominating_rung=769)
        a=copy.deepcopy(low['measurements'][0]);b=copy.deepcopy(high['measurements'][0])
        low['dominance_evidence']=[{'cell_id':a['cell_id'],'lower_time_us':a['kernel_time_us'],
            'higher_time_us':b['kernel_time_us'],'comparison_id':'paired','lower_measurement':a,'higher_measurement':b}]
        validate_table(t)
        for mutate in (lambda w:w['geometry'].update(body_kind='tcq'),
                       lambda w:w['geometry'].update(execution_scope='native_tcq_decode_gemm'),
                       lambda w:w.update(kernel_time_us=float('nan')),
                       lambda w:w.update(pass_times_us=[8,8])):
            bad=copy.deepcopy(t);mutate(bad['rungs'][0]['dominance_evidence'][0]['higher_measurement'])
            with self.assertRaises(ValueError):validate_table(bad)



if __name__=='__main__': unittest.main()
