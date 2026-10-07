"""Harvest the existing geometry/quality harness into versioned D41 tables.

Run through PrismaBuild on CPU. Unfinished jobs and missing quality stay pending.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re

"""This module stays free of torch at import time. Each tessera import
below runs inside the function that needs it, so unit tests load this
file with stdlib only."""

MS=(1,16,2048,4096)
SHAPES=(("routed","gate_up",1024,4096,0),("routed","down",4096,1024,2),("dense","o_proj",4096,4096,2),("dense","q_b",8192,1536,2))
FORMAT="TESSERA_E4M3_K1"
STRUCTURE_SPEC_SCHEMA="tessera.d41_structure_spec.v1"
DEFAULT_SHAPE_OWNER="bench_rates TP2 shapes; actual GLM config hidden4096, routed inter2048/2, experts288, topk8"
RECORDED_ROUTING_MIN_M=2048


def parse_structure_spec(path):
    """Read a D41 structure spec file and return its sweep geometry."""
    doc=json.loads(Path(path).read_text())
    if doc.get("schema")!=STRUCTURE_SPEC_SCHEMA:
        raise ValueError(f"structure spec schema must be {STRUCTURE_SPEC_SCHEMA}")
    raw_shapes=doc.get("shapes")
    if not isinstance(raw_shapes,list) or not raw_shapes:
        raise ValueError("structure spec needs a non-empty shapes list")
    shapes=[]
    for entry in raw_shapes:
        if not isinstance(entry,dict):
            raise ValueError("each shape must be an object")
        kind=entry.get("kernel_kind")
        if kind not in ("routed","dense"):
            raise ValueError(f"shape kernel_kind must be routed or dense, got {kind!r}")
        name=entry.get("shape_id")
        if not isinstance(name,str) or not name:
            raise ValueError("each shape needs a non-empty shape_id string")
        rows=entry.get("rows")
        columns=entry.get("columns")
        if not isinstance(rows,int) or rows<=0 or not isinstance(columns,int) or columns<=0:
            raise ValueError(f"shape {name!r} needs positive integer rows and columns")
        mode=entry.get("mode")
        if mode not in (0,2):
            raise ValueError(f"shape {name!r} mode must be 0 (gate/up) or 2 (down/dense)")
        shapes.append((kind,name,rows,columns,mode))
    raw_ms=doc.get("ms")
    if not isinstance(raw_ms,list) or not raw_ms or any(not isinstance(m,int) or m<=0 for m in raw_ms):
        raise ValueError("structure spec needs a non-empty ms list of positive integers")
    ms=tuple(raw_ms)
    if len(set(ms))!=len(ms):
        raise ValueError("structure spec ms must not repeat a value")
    meta={}
    for key in ("experts","top_k","hidden","inter"):
        value=doc.get(key)
        if not isinstance(value,int) or value<=0:
            raise ValueError(f"structure spec needs a positive integer {key}")
        meta[key]=value
    return {"shapes":tuple(shapes),"ms":ms,"meta":meta,"spec_id":doc.get("spec_id")}


def resolve_sweep_geometry(structure_spec):
    """Return shapes, ms, owner text and scope record for one harvest."""
    if structure_spec is None:
        return SHAPES,MS,DEFAULT_SHAPE_OWNER,None
    parsed=parse_structure_spec(structure_spec)
    meta=parsed["meta"]
    owner=(f"structure-spec {parsed['spec_id'] or Path(structure_spec).name}; "
             f"experts{meta['experts']}, top_k{meta['top_k']}, "
             f"hidden{meta['hidden']}, inter{meta['inter']}")
    digest=hashlib.sha256(Path(structure_spec).read_bytes()).hexdigest()
    record={"spec_id":parsed["spec_id"],"sha256":digest,
              "file":Path(structure_spec).name,
              "experts":meta["experts"],"top_k":meta["top_k"],
              "hidden":meta["hidden"],"inter":meta["inter"]}
    return parsed["shapes"],parsed["ms"],owner,record


def roster(shapes=SHAPES,ms=MS):
    """List one cell per shape and M, plus recorded-routing cells."""
    cells=[{"cell_id":f"{kind}:{name}:M{m}","kernel_kind":kind,"shape_id":name,"M":m} for kind,name,_r,_c,_mode in shapes for m in ms]
    seen=[]
    for kind,name,_r,_c,_mode in shapes:
        if kind=="routed" and name not in seen:
            seen.append(name)
    recorded=[m for m in ms if m>=RECORDED_ROUTING_MIN_M]
    cells += [{"cell_id":f"routed:{name}:M{m}:recorded","kernel_kind":"routed","shape_id":name,"M":m,"routing":"recorded"} for name in seen for m in recorded]
    return cells


def resources(usage,head,cell,*,fp8=True):
    want=f"routed_fused_kernel<{'true' if fp8 else 'false'}, {head.get('mode',2)}, {'true' if head['kind']=='dense' else 'false'}, {'true' if cell.get('k_split',1)>1 else 'false'}, {head['r_lo']}, {'true' if head['n_hi'] else 'false'}, {cell['bm']}, false, false>"
    observed=[p for p in cell.get("profile",{}).get("top",{}) if "routed_fused_kernel" in p]
    matches=[(name,res) for name,res in usage.get("kernels",{}).items() if want in name and any(name.startswith(p) for p in observed)]
    return matches[0] if len(matches)==1 else (None,None)


def body_geometry(head,cell,geometry,meta,grid):
    """Normalize actual decoder facts; canonical validation owns their semantics."""
    import copy
    g=copy.deepcopy(geometry)
    recipe=head.get('recipe')
    body=head.get('body_kind',recipe.get('body') if isinstance(recipe,dict) else None)
    if recipe is None and grid.name in ('BF16','E4M3'):
        from tessera.export import wire_recipe
        recipe=wire_recipe(grid,head['q256']).to_config()
        if body is None:body=recipe['body']
    body=str(body).lower()
    compact=bool(cell.get('compiler_resources'))
    if meta.get('library')=='native_span2':
        kind,owner,scope='native_tcq','tessera.kernel_a4','native_tcq_decode_gemm'
    elif compact:
        kind='compact_window'
        owner='tessera.window_gemm_grouped' if head['kind']=='routed' else 'tessera.window_gemm'
        scope='compact_packed_window'
    else:
        kind,owner,scope='fused_window','tessera.routed_fused','raw_packed_window'
    g.update(body_kind=body,decoder_kind=kind,decoder_owner=owner,execution_scope=scope,
             word_ring={'kind':'staged' if kind=='fused_window' else 'none','owner':owner})
    if recipe is not None:g['recipe']=recipe
    d,a,reg=g['decode_width'],g['alignment'],g.get('register_pressure')
    request=g['shared_memory'].get('requested_bytes')
    g['shared_memory']['kind']='used' if isinstance(request,int) and request>0 else ('none' if request==0 else 'unmeasured')
    if kind=='fused_window':
        if isinstance(reg,dict):reg['compiler']='cuda_cuobjdump'
        return g
    d['word_stages']=None
    a['slot_words']=None
    if kind=='native_tcq':
        a.update(kind='tcq_planes',owner='tessera.compact_prep.prepare_span2_compact')
        d['history_lookup_bits']=d['memory']+1
        d['label_lut_entries']=a['plane_shapes']['label_lut'][-1]
        # Element widths are observed bytes/numel; zero POINT keeps its actual
        # byte-plane dtype from A4Unit.from_prepared, never a positive plane size.
        a['plane_element_bytes']={}
        for name,shape in a['plane_shapes'].items():
            count=__import__('math').prod(shape)
            if count:a['plane_element_bytes'][name]=a['plane_bytes'][name]//count
            elif name=='point':
                import torch
                a['plane_element_bytes'][name]=torch.empty(0,dtype=torch.uint8).element_size()
            else:a['plane_element_bytes'][name]=None
    else:
        resources=cell.get('compiler_resources',{})
        observed=cell.get('profile',{}).get('top',{})
        selected=[(name,value) for name,value in resources.items() if any(name in p or p in name for p in observed)]
        if len(selected)==1:
            name,res=selected[0]
            reg={'REG':res.get('REG'),'SPILLS':res.get('spills'),'SHARED':res.get('SHARED'),'compiler_symbol':name}
            g['register_pressure']=reg
            for key in ('block_m','block_n','block_k'):
                d[key]=res.get('launch',{}).get({'block_m':'BM','block_n':'BN','block_k':'BK'}[key])
        g['alignment']={'kind':'column_chunk_words','owner':'tessera.kernel_window_gemv.Repacked',
                        'slot_words':None,'word_alignment_bytes':4,'tile_rows':512,
                        'tile_words':head['tile_words'],'column_chunk_words':[16*r for r in d['run_widths']]}
    if isinstance(reg,dict):
        reg.update(compiler='triton_compiled_kernel',STACK=None,LOCAL=None)
    return g



def measurement(path,data,head,cell,key,build_id,cols,rows,grid):
    meta=data['meta']
    from tessera import routed_fused as rf
    from tessera.control import unit_wire_bits
    f,r=cell.get('F',{}),cell.get('R',{})
    timer=f.get('timer') if f.get('timer')==r.get('timer') else None
    fp8 = grid.name == 'E4M3'
    normalized=cell.get('normalized_geometry',head.get('normalized_geometry'))
    name,res=resources(meta.get('resource_usage',{}),head,cell,fp8=fp8) if normalized is None and 'bm' in cell and 'r_lo' in head else (None,None)
    mode=head.get('mode',2)
    geo=head.get('geometry',{})
    slot=geo.get('slot_words')
    requested=rf.launch_smem_bytes(mode,slot,mma8=bool(meta.get('library')=='e4m3mma'),bm=cell['bm']) if slot is not None and 'bm' in cell else None
    compiler=cell.get('compiler_resources',{})
    if compiler:
        observed=cell.get('profile',{}).get('top',{})
        selected={n:v for n,v in compiler.items() if any(n in p or p in n for p in observed)}
        if selected:
            name='; '.join(sorted(selected)); res=selected
            shared=[v.get('SHARED') for v in selected.values()]
            requested=max(shared) if all(type(v) is int for v in shared) else None
    available=meta.get('shared_memory_available')
    bits=Fraction(unit_wire_bits(grid,head['q256'],rows,cols))*256/(rows*cols)
    error=f.get('error') or r.get('error') or cell.get('error')
    from tessera.export import wire_recipe
    recipe=wire_recipe(grid,head['q256'])
    geometry={'bits_per_256_weight_tile':{'numerator':bits.numerator,'denominator':bits.denominator},
              'alignment':{k:v for k,v in geo.items() if k in ('lane_bits','lane_ends_on_word','half_bytes','half_copy','slot_words')},
              'shared_memory':{'requested_bytes':requested,'available_bytes':available,'fits':bool(requested is not None and available is not None and requested<=available)},
              'register_pressure':res,'decode_width':{'window_bits':head.get('window_bits',recipe.window_bits),'value_bits':grid.payload_bits,'arity':grid.arity,'body_kind':head.get('body_kind',recipe.body.name),'run_widths':geo.get('rates'),'word_stages':None if compiler else geo.get('word_stages'),'superblock_rows':cell.get('bm'),'k_split':cell.get('k_split',1)},
              'raw':{'owner':'actual fused launch or Triton CompiledKernel; control.unit_wire_bits','packed_tile_words':head.get('tile_words'),'payload_bits_per_256_weights':{'numerator':head.get('tile_words',0)*16,'denominator':cols}}}
    if normalized is not None:
        geometry=normalized
        name=cell.get('kernel_path')
        res=geometry.get('register_pressure')
        requested=geometry.get('shared_memory',{}).get('requested_bytes')
        available=geometry.get('shared_memory',{}).get('available_bytes')
    good=bool(not error and cell.get('ms',0)>0 and timer and name and res and requested is not None and available is not None)
    evidence={'geometry_file':str(path),'action_key':meta.get('pb_action'),'host':meta.get('host'),
              'comparison_id':meta.get('pb_action'),'paired_seed_contract':meta.get('paired_seed_contract'),
              'timing_statistic':meta.get('statistic'),'timer':timer,'F':f,'R':r,'profile':cell.get('profile'),
              'power':cell.get('power'),'loaded_library_sha256':meta.get('library_sha256'),'kernel_source_sha256':meta.get('kernel_sha'),
              'quantum_window_unix':[meta.get('start_unix'),meta.get('end_unix')],
              'rows':rows,'columns':cols,'mode':mode,'routing':key.get('routing','balanced') if key['kernel_kind']=='routed' else 'none',
              'recorded_routing':meta.get('recorded',{}).get(str(key['M'])),
              'routing_weights':'uniform 1/top_k (1/8); recorded IDs only' if key['kernel_kind']=='routed' else 'none',
              'input_distribution':meta.get('activation_contract'),'epilogue':'SwiGLU clipped at 10' if mode==0 else ('route-weighted BF16 down' if key['kernel_kind']=='routed' else 'BF16 linear output'),
              'path_scope':cell.get('path_scope',head.get('path','raw packed fused')),
              'owner_refusal':head.get('owner_refusal'), 'decode_sources':meta.get('decode_sources')}
    if head.get('kind')=='dense' and head.get('path')=='compact_dense_folded' and not meta.get('dense_seed_without_routing_suffix'):
        evidence['paired_seed_contract']=str(evidence['paired_seed_contract'])+'; legacy compact dense seed has :None suffix'
    geometry=body_geometry(head,cell,geometry,meta,grid)
    if error:evidence['reason']=error
    elif not good:evidence['reason']='missing paired timer, actual compiler resource or launch geometry evidence'
    return {**key,'measurement_status':'measured' if good else ('failed' if error else 'pending'),'kernel_time_us':cell['ms']*1000 if good else None,
            'kernel_path':name,'geometry':geometry,'evidence':evidence,'pass_times_us':[f['median_ms']*1000,r['median_ms']*1000] if good else [],'measurement_build_id':build_id}



def merge_index(index, format_name, build_id, version, relative, table):
    """Preserve every immutable version and every other family's entry."""
    from tessera.rung_allowability import validate_index
    validate_index(index)
    index['schema']='fleet.rung_allowability.index.v2'
    builds=index['formats'].setdefault(format_name, {'kernel_builds':{}})['kernel_builds']
    entry=builds.setdefault(build_id, {'current_version':version, 'versions':{}})
    record={'path':relative,'table_schema':table['schema'],'table_status':table['table_status']}
    if str(version) in entry['versions'] and entry['versions'][str(version)]!=record:
        raise ValueError('conflicting immutable table version')
    entry['versions'][str(version)]=record
    entry['current_version']=version
    return validate_index(index)


def activate_published_index(publication, candidate_name="index.v2-candidate.json"):
    """Select the staged immutable versions; never rerun their harvest."""
    import copy,fcntl,os
    from tessera.rung_allowability import validate_index, validate_table
    publication=Path(publication)
    with (publication/'.publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        selected=publication/"index.json";candidate_path=publication/candidate_name
        current=json.loads(selected.read_text())
        candidate=json.loads(candidate_path.read_text())
        validate_index(current);validate_index(candidate)
        merged=copy.deepcopy(current);merged['schema']='fleet.rung_allowability.index.v2'
        for format_name,format_entry in candidate['formats'].items():
            builds=merged['formats'].setdefault(format_name,{'kernel_builds':{}})['kernel_builds']
            for build_id,entry in format_entry['kernel_builds'].items():
                dest=builds.setdefault(build_id,{'current_version':entry['current_version'],'versions':{}})
                for version,record in entry['versions'].items():
                    if version in dest['versions'] and dest['versions'][version]!=record:
                        raise ValueError('conflicting immutable index history')
                    dest['versions'][version]=record
                version=str(entry['current_version']);record=entry['versions'][version]
                table=json.loads((publication/record['path']).read_text())
                validate_table(table)
                if (table['schema']!=record['table_schema'] or table['table_status']!=record['table_status']
                        or table['format']!=format_name or table['kernel_build']['id']!=build_id
                        or table['table_version']!=entry['current_version']):
                    raise ValueError('selected table/index bytes disagree')
                dest['current_version']=entry['current_version']
        validate_index(merged)
        if current['schema']=='fleet.rung_allowability.index.v1':
            history=publication/'index.v1-history.json'
            if history.exists() and json.loads(history.read_text())!=current:
                raise ValueError('immutable original index history differs')
            if not history.exists():
                with history.open('x') as stream:json.dump(current,stream,indent=2,allow_nan=False)
        temporary=selected.with_suffix('.tmp')
        with temporary.open('w') as stream:
            json.dump(merged,stream,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        temporary.replace(selected)
        return {'status':'staged_candidate_selected','schema':merged['schema'],'formats':list(merged['formats'])}





def apply_reader_findings(row, findings, format_name, kernel_shas):
    """Keep source-specific correctness findings on both flag sets and observations."""
    for finding in findings:
        if finding['format'] != format_name or finding['kernel_sha'] not in kernel_shas:
            continue
        flags = sorted(set(row['anomaly_flags']) | {finding['anomaly_flag']})
        row['anomaly_flags'] = flags
        row['quality']['anomaly_flags'] = flags
        observation = {'kind': 'reader_correctness_finding', 'blocking': True, 'exclusion_basis': False, 'finding': finding}
        if observation not in row['observations']:
            row['observations'].append(observation)


def performance_increment(table, raw_inputs, version, findings=()):
    """Upgrade a newly loaded table, preserve history, append only actual cells."""
    from tessera.rung_allowability import PERFORMANT_POLICY, measured_geometry_classes, validate_table
    validate_table(table)
    original_schema = table['schema']
    base, arity_text = table['format'].removeprefix('TESSERA_').rsplit('_K', 1)
    from tessera.alphabet import tuple_grid
    from tessera.control import grid_for_name
    grid = grid_for_name(base)
    if int(arity_text) != 1:
        grid = tuple_grid(grid, int(arity_text))
    by_q = {row['rung']: row for row in table['rungs']}
    for row in table['rungs']:
        if row['excluded']:
            row['observations'].append({'kind': 'historical_exclusion', 'dominating_rung': row['dominating_rung'], 'evidence': row['dominance_evidence']})
        row.update(excluded=False, dominating_rung=None, dominance_evidence=[])
        if original_schema == 'fleet.rung_allowability.v1':
            for cell in row['measurements']:
                if cell['measurement_status'] == 'measured':
                    head = {'q256': row['rung'], 'kind': cell['kernel_kind'], 'body_kind': 'WINDOW'}
                    cell['geometry'] = body_geometry(head, {}, cell['geometry'], {'library': table['kernel_build']['library_variant']}, grid)
    required = {cell['cell_id']: cell for cell in table['scope']['required_cells']}
    shapes = {(shape['kernel_kind'], shape['shape_id']): shape for shape in table['scope']['shapes']}
    imported, ignored = 0, []
    for input_path in raw_inputs:
        data = json.loads(Path(input_path).read_text())
        meta = data['meta']
        if meta.get("format", FORMAT) != table["format"]:
            ignored.append(str(input_path))
            continue
        if not meta.get('end_unix') or not meta.get('pb_action'):
            raise ValueError('Incomplete actual geometry input')
        signature = {key: meta.get(key) for key in ('kernel_sha', 'architecture', 'library', 'activation_contract', 'image', 'torch', 'decode_sources')}
        actual_build = meta['library'] + '-' + meta['architecture'] + '-' + hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        if actual_build != table['kernel_build']['id']:
            from tessera.dev_mode import seal_check
            seal_check('historical geometry build label', table['kernel_build']['id'], actual_build, where=str(input_path))
        if meta['architecture'] != table['kernel_build']['architecture'] or meta['library'] != table['kernel_build']['library_variant'] or meta['activation_contract'] != table['kernel_build']['activation_contract']:
            raise ValueError('Actual architecture, decoder variant or activation scope differs; publish a separate scoped table')
        for head in data['groups'].values():
            if head.get('kind') != 'dense':
                continue
            q = head['q256']
            if q not in by_q:
                raise ValueError('Actual measured rung is outside this table census')
            name, rows, columns = head['shape'], head['rows'], head['cols']
            shape = {'shape_id': name, 'kernel_kind': 'dense', 'rows': rows, 'columns': columns, 'mode': 2}
            shape_key = ('dense', name)
            if shape_key in shapes and shapes[shape_key] != shape:
                raise ValueError('Same shape identifier has different current dimensions')
            shapes[shape_key] = shape
            row = by_q[q]
            cells = {cell['cell_id']: cell for cell in row['measurements']}
            for M_text, observed in head['cells'].items():
                key = {'cell_id': 'dense:' + name + ':M' + M_text, 'kernel_kind': 'dense', 'shape_id': name, 'M': int(M_text)}
                required[key['cell_id']] = key
                current = measurement(Path(input_path), data, head, observed, key, table['kernel_build']['id'], columns, rows, grid)
                current['evidence']['observed_measurement_build_id'] = actual_build
                # Failed/unsupported cells remain negative evidence, not borrowed timing.
                if key['cell_id'] in cells:
                    if cells[key['cell_id']] != current:
                        row['observations'].append({'kind': 'previous_actual_measurement', 'measurement': cells[key['cell_id']]})
                cells[key['cell_id']] = current
                imported += 1
            row['measurements'] = list(cells.values())
    table['scope']['required_cells'] = list(required.values())
    table['scope']['shapes'] = list(shapes.values())
    for row in table['rungs']:
        observed = {cell['cell_id']: cell for cell in row['measurements']}
        complete = set(observed) == set(required) and all(cell['measurement_status'] == 'measured' for cell in observed.values())
        if complete and row["measurement_status"] not in ("failed", "unsupported") and row["supported"] is not False:
            row.update(measurement_status='measured', supported=True)
        elif row['measurement_status'] == 'measured':
            row['measurement_status'] = 'pending'
    source_sha = table['kernel_build'].get('metadata', {}).get('observed_signature', {}).get('kernel_sha')
    for row in table['rungs']:
        kernel_shas = {cell['evidence'].get('kernel_source_sha256') for cell in row['measurements']}
        kernel_shas.add(source_sha)
        kernel_shas.discard(None)
        apply_reader_findings(row, findings, table['format'], kernel_shas)
    table.update(schema='fleet.rung_allowability.v3', table_version=version, generated_at=datetime.now(timezone.utc).isoformat(), performant_policy=dict(PERFORMANT_POLICY))
    table['table_status'] = 'complete' if all(row['measurement_status'] != 'pending' for row in table['rungs']) else 'partial'
    table['geometry_classes'] = measured_geometry_classes(table)
    table['evidence']['performance_increment'] = {'input_schema': original_schema, 'actual_cells_imported': imported, 'ignored_other_format_inputs': ignored,
        'scope': 'Actual per-cell performance evidence, not original-weight numerical, assembled module, tensor-parallel collective or serving qualification. Quality samples remain historical telemetry.'}
    table['evidence']['summary'] = dict(Counter(row['measurement_status'] for row in table['rungs']))
    return validate_table(table)



def publish_table(table, args):
    """One immutable writer for fresh harvests and class-only metadata increments."""
    from tessera.rung_allowability import validate_index, validate_table
    import jsonschema
    validate_table(table)
    jsonschema.Draft202012Validator(json.loads(Path(args.schema).read_text()), format_checker=jsonschema.FormatChecker()).validate(table)
    format_name, build_id = table['format'], table['kernel_build']['id']
    rows = table['rungs']
    relative=f'{format_name}/{build_id}/v{args.version:04d}.json'
    index=json.loads(Path(args.index).read_text()) if args.index else {'schema':'fleet.rung_allowability.index.v1','formats':{}}
    merge_index(index,format_name,build_id,args.version,relative,table)
    validate_index(index)
    jsonschema.Draft202012Validator(json.loads(Path(args.index_schema).read_text())).validate(index)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    with (out/'table.json').open('x') as h:json.dump(table,h,indent=1,allow_nan=False)
    with (out/'index.json').open('x') as h:json.dump(index,h,indent=2,allow_nan=False)
    if args.publish_root:
        import fcntl
        publication=Path(args.publish_root)
        publication.mkdir(parents=True,exist_ok=True)
        with (publication/'.publication.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            selected_path=publication/'index.json'
            current_path=publication/args.candidate_index_name
            source_path=current_path if current_path.exists() else selected_path
            current=json.loads(source_path.read_text()) if source_path.exists() else {'schema':'fleet.rung_allowability.index.v2','formats':{}}
            merge_index(current,format_name,build_id,args.version,relative,table)
            destination=publication/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            with destination.open('x') as stream:json.dump(table,stream,indent=1,allow_nan=False)
            temporary=current_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(current,indent=2,allow_nan=False))
            temporary.replace(current_path)
    report={'status':'schema_and_semantic_validation_passed','table_path':relative,'table_status':table['table_status'],'kernel_build_id':build_id,'summary':dict(Counter(row['measurement_status'] for row in rows)),'excluded':[row['rung'] for row in rows if row['excluded']],'action_key':__import__('os').environ.get('PRISMABUILD_ACTION_KEY')}
    report['cell_summary']=dict(Counter(m['measurement_status'] for row in rows for m in row['measurements']))
    report['quality_rungs_measured']=sum(row['quality'].get('measurement_status')=='measured' for row in rows)
    report['geometry_classes']=len(table.get('geometry_classes', []))
    (out/'validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',required=True)
    ap.add_argument('--schema',required=True)
    ap.add_argument('--index-schema',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--version',required=True,type=int)
    ap.add_argument('--format',default=FORMAT)
    ap.add_argument('--index',help='existing shared index to preserve')
    ap.add_argument('--publish-root',help='publish immutable table and atomically advance merged index')
    ap.add_argument('--activate-index',action='store_true',help='advance current selection after the consumer supports this explicit schema')
    ap.add_argument('--catalog',help='family owner catalog with exact recipes and concrete path refusals')
    ap.add_argument('--reader-findings',help='explicit source-specific correctness findings; existing anomaly holds remain canonical')
    ap.add_argument('--quality-root',help='actual producer-scoped quality outputs, without retagging unscoped history')
    ap.add_argument("--input-table", help="existing immutable table for a scoped performance-policy increment")
    ap.add_argument("--measurement-input", action="append", default=[], help="completed raw geometry JSON, normalized by this existing producer")
    ap.add_argument("--candidate-index-name", choices=("index.v2-candidate.json", "index.v3-candidate.json"), default="index.v2-candidate.json")
    ap.add_argument('--structure-spec',help='D41 structure spec JSON with sweep shapes, experts, top_k, hidden, inter and Ms; without it the tool keeps its compiled GLM defaults')
    args=ap.parse_args()
    if args.activate_index:
        if not args.publish_root:raise ValueError('index activation needs publication root')
        report=activate_published_index(args.publish_root, args.candidate_index_name)
        print(json.dumps(report),flush=True)
        return
    findings=json.loads(Path(args.reader_findings).read_text()) if args.reader_findings else []
    if args.input_table:
        table = performance_increment(json.loads(Path(args.input_table).read_text()), args.measurement_input, args.version, findings)
        publish_table(table, args)
        return
    root=Path(args.root)
    format_name=args.format
    match=re.fullmatch(r'TESSERA_([A-Z0-9]+)_K(\d+)',format_name)
    if not match:raise ValueError('expected family format name, without a rung suffix')
    base,arity=match[1],int(match[2])
    from tessera.alphabet import tuple_grid
    from tessera.control import grid_for_name
    scalar=grid_for_name(base)
    grid=scalar if arity==1 else tuple_grid(scalar,arity)
    completed=[]
    geometry_paths=sorted(root.glob("gpu*/**/bench_geometry*.json"))
    for path in geometry_paths:
        try:data=json.loads(path.read_text())
        except (json.JSONDecodeError,OSError):continue
        if data.get('meta',{}).get('end_unix') and data['meta'].get('format',FORMAT)==format_name: completed.append((path,data))
    if not completed: raise ValueError('No completed geometry quantum; no observed build to publish')
    meta=completed[0][1]['meta']
    from tessera.export import wire_recipe
    from tessera.manifest import body_rate_cap
    catalog=json.loads(Path(args.catalog).read_text())['families'][format_name] if args.catalog else {}
    lower=catalog.get('rung_min',meta.get('rung_min',768 if format_name==FORMAT else 256//arity))
    upper=catalog.get('rung_max',meta.get('rung_max',1152 if format_name==FORMAT else 256*body_rate_cap(wire_recipe(grid).body,grid)//arity))
    # Identity derives from observed code/architecture/variant, not a refusal seal.
    signature={k:meta.get(k) for k in ('kernel_sha','architecture','library','activation_contract','image','torch','decode_sources')}
    build_id=meta['library']+'-'+meta['architecture']+'-'+hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest()[:16]
    build={'id':build_id,'source_commit':meta['tessera_head'],'library_variant':meta['library'],'architecture':meta['architecture'],'activation_contract':meta['activation_contract'],
           'metadata':{'observed_signature':signature,'loaded_libraries':sorted({d['meta']['library_sha256'] for _,d in completed}),'actual_source_snapshots':sorted({d['meta']['tessera_head'] for _,d in completed}),'flags':[]}}
    for path,_ in completed:
        for ninja in path.parent.glob('home/torch_extensions/*/build.ninja'):
            build['metadata']['flags'].append({'file':str(ninja),'cuda_cflags':[l for l in ninja.read_text().splitlines() if l.startswith('cuda_cflags =')]})
    shapes,ms,shape_owner,spec_record=resolve_sweep_geometry(args.structure_spec)
    required=roster(shapes,ms)
    by_rung={q:{'rung':q,'measurement_status':'pending','supported':None,'anomaly_flags':[],'observations':[], 'excluded':False,'dominating_rung':None,'measurements':[],'quality':{},'dominance_evidence':[],'lineage':{}} for q in range(lower,upper+1)}
    for facts in catalog.get('rungs',[]):
        by_rung[facts['q256']]['observations'].append({'kind':'producer_recipe_and_reader_scope','facts':facts,'blocking':False,'exclusion_basis':False})
    quality={'rungs':{}}
    quality_root=Path(args.quality_root) if args.quality_root else root/'quality'
    quality_paths=([quality_root] if quality_root.is_file() else sorted(quality_root.glob('*.json')))
    if not args.quality_root and (root/'quality.json').exists():quality_paths.insert(0,root/'quality.json')
    for quality_path in quality_paths:
        document=json.loads(quality_path.read_text())
        if document.get('format')!=format_name:continue
        for rung,value in document.get('rungs',{}).items():
            if not isinstance(value.get('scope'),dict):continue
            if rung in quality['rungs'] and quality['rungs'][rung]!=value:raise ValueError(f'conflicting quality evidence for rung {rung}')
            quality['rungs'][rung]=value
    # Prefer the quantum where the rung and its next neighbor share one F/R run.
    candidates={}
    alternates={}
    for path,data in completed:
        if any(data['meta'].get(k)!=v for k,v in signature.items()):
            continue  # Different actual build belongs in a different table, not mixed numerics.
        cases=data['meta']['cases']
        qs=[int(str(v).removeprefix('q')) for v in (cases.split(',') if isinstance(cases,str) else cases)]
        for group in data['groups'].values():
            q=group.get('q256')
            if q not in by_rung or group.get('kind') not in ('routed','dense'):continue
            kind=group["kind"]; name=group.get("shape") if kind=="dense" else ("gate_up" if group["mode"]==0 else "down")
            spec=next((s for s in shapes if s[:2]==(kind,name)),None)
            if not spec:continue
            for key in required:
                if key["kernel_kind"]!=kind or key["shape_id"]!=name:continue
                m=key["M"]
                how=key.get("routing","balanced")
                cell=group.get("cells",{}).get(str(m)+(":"+how if kind=="routed" else ""),{})
                if not cell:continue
                candidate=measurement(path,data,group,cell,key,build_id,spec[3],spec[2],grid)
                preference=(int(q+1 in qs),int(candidate['measurement_status']=='measured'))
                ckey=(q,key['cell_id'])
                if ckey not in candidates or preference>candidates[ckey][0]:candidates[ckey]=(preference,candidate)
                alternates.setdefault(ckey,[]).append(candidate)
    for q,row in by_rung.items():
        row['measurements']=[candidates[(q,k['cell_id'])][1] for k in required if (q,k['cell_id']) in candidates]
        row['quality']=quality.get('rungs',{}).get(str(q),{'measurement_status':'pending'})
        row['anomaly_flags']=row['quality'].get('anomaly_flags',[])
        apply_reader_findings(row, findings, format_name, {meta['kernel_sha']})
        row['lineage']={'quality_files':[str(p) for p in quality_paths],'geometry_files':sorted({m['evidence']['geometry_file'] for m in row['measurements']})}
        if len(row['measurements'])==len(required) and all(m['measurement_status']=='measured' for m in row['measurements']) and row['quality'].get('measurement_status')=='measured':
            row['measurement_status']='measured';row['supported']=True
        elif any(m['measurement_status']=='failed' for m in row['measurements']) or row['quality'].get('measurement_status')=='failed':
            row['measurement_status']='failed'; row['supported']=None
        if format_name==FORMAT and q in (880,912):row['observations'].append({'kind':'missing_census','issue':689,'url':'https://github.com/RobTand/tessera/issues/689','blocking':False,'exclusion_basis':False,'scope':'historical routed E4M3 census gap; geometry does not mint served cells'})
    for q,row in by_rung.items():
        high=by_rung.get(q+1)
        # Keep every logical adjacent comparison, including quantum boundaries,
        # on the exact common receipt, rather than crossing clock windows.
        paired_cells=[]
        for key in required:
            low_options=alternates.get((q,key['cell_id']),[])
            high_options=alternates.get((q+1,key['cell_id']),[])
            pair=next(((a,b) for a in low_options for b in high_options if a['measurement_status']=='measured' and b['measurement_status']=='measured' and all(a['evidence'].get(k) and a['evidence'].get(k)==b['evidence'].get(k) for k in ('comparison_id','paired_seed_contract','timing_statistic','timer'))),None)
            if pair:
                a,b=pair
                paired_cells.append({'cell_id':key['cell_id'],'lower_time_us':a['kernel_time_us'],'higher_time_us':b['kernel_time_us'],'comparison_id':a['evidence']['comparison_id'],'lower_measurement':a,'higher_measurement':b})
        if high:
            row['observations'].append({'kind':'adjacent_higher_comparison','higher_rung':q+1,'paired_cells_completed':len(paired_cells),'required_cells':len(required),'all_cell_at_least_as_fast':len(paired_cells)==len(required) and all(e['higher_time_us']<=e['lower_time_us'] for e in paired_cells),'evidence':paired_cells})
        if row['measurement_status']!='measured' or not high or high['measurement_status']!='measured' or high['supported'] is not True or high['anomaly_flags']:continue
        if len(paired_cells)==len(required) and all(e['higher_time_us']<=e['lower_time_us'] for e in paired_cells):
            row["observations"].append({"kind":"proposed_adjacent_higher_exclusion","higher_rung":q+1,"status":"pending parent and independent review","blocking":False,"applied":False,"evidence":paired_cells})

    baseline=by_rung.get(1024)
    for q,row in by_rung.items():
        if format_name==FORMAT and q>1024 and baseline:
            ratios=[]
            for m in row['measurements']:
                base=next((x for x in baseline['measurements'] if x['cell_id']==m['cell_id'] and x['measurement_status']=='measured'),None)
                if base and m['measurement_status']=='measured':ratios.append({'cell_id':m['cell_id'],'ratio_to_1024':m['kernel_time_us']/base['kernel_time_us'],'receipt':m['evidence']['action_key'],'baseline_receipt':base['evidence']['action_key'],'paired':m['evidence']['comparison_id']==base['evidence']['comparison_id']})
            row['observations'].append({'kind':'beyond_1024_slow_lane','issue':690,'url':'https://github.com/RobTand/tessera/issues/690','blocking':False,'exclusion_basis':False,'ratios':ratios,'missing_baseline':not bool(ratios)})
    rows=list(by_rung.values())
    table={"schema":"fleet.rung_allowability.v2","table_version":args.version,"table_status":"complete" if all(r["measurement_status"]!="pending" for r in rows) else "partial","format":format_name,"kernel_build":build,"generated_at":datetime.now(timezone.utc).isoformat(),
           "scope":{"rung_min":lower,"rung_max":upper,"grid_step_q256":1,"grid_owner":meta.get('grid_owner','prismaquant.tessera_formats.realisable_rungs(step_q256=1)'),"required_cells":required,"shapes":[{"shape_id":n,"kernel_kind":k,"rows":r,"columns":c,"mode":mode} for k,n,r,c,mode in shapes],"timing_statistic":meta["statistic"],"shape_owner":shape_owner},"rungs":rows,"evidence":{"summary":dict(Counter(r["measurement_status"] for r in rows)),"completed_quanta":len(completed),"quality_scope":"fixed actual expert 0 layer3 gate/up/down 32x256 sample; unweighted weight-space SSE, not served KL","exclusion_review_status":"pending independent review; no defaults promoted"}}
    if spec_record is not None:
        table["scope"]["structure_spec"]=spec_record
    table['scope']['kernel_execution_scope']=meta.get('kernel_execution_scope',meta.get('execution_scope','rank-local packed fused and actual public compact projections at TP2 dimensions; synthetic packed wires; each cell names its path; serving intake refusals and gates remain independent'))
    table["evidence"]["quality_summary_file"]=str(root/"quality-summary.json")
    table["evidence"]["exclusion_review_status"]="Candidate proofs are non-blocking proposals only; no exclusions applied before parent and independent review."
    table['evidence']['compiler_resource_lookup_lineage']={'resource_source':'Actual cuobjdump matched to torch.profiler kernel prefixes, or actual Triton CompiledKernel returned by the measured launch','actual_measured_sources':build['metadata']['actual_source_snapshots'],'no_cross_family_inheritance':True}
    table["evidence"]["source_cohort_scope"]="Source/variant/architecture/activation/image/compiler cohort; exact loaded binary SHA-256 retained in every cell. Different snapshot or binary identities are lineage, not a new seal."
    table["evidence"]["power_series_scope"]="Per-cell NVML samples plus both-Spark Netdata power series; utilization percentages are not used to diagnose saturation."
    publish_table(table, args)


if __name__=='__main__':main()
