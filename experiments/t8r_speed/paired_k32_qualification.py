"""Closed-world paired-K32 numeric evidence inside the existing T8R harness.

The ordinary Store, rank-local intake, native callback, stage implementation,
materializing reader and independent routed oracle remain the owners. This
module selects a finite population and observes those owners; it is not a
second decoder, runtime or benchmark scheduler.
"""
from pathlib import Path
import hashlib
import json

REPLAY_ARTIFACT = '/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported'
GROUP = 'experts.R1024.L10'
MODULE = 'model.language_model.layers.10.mlp.experts'
NUMERIC_RECEIPT = '/mnt/shared/astra-resume-20261002/t8_performance/paired-k32-direct-numeric-v3/RESULT.json'
NUMERIC_RECEIPT_SHA = 'a5a8e479a8ab6e9ad26944d554aa7c88e2dda41423b4882e7eaeac9ae129c9d9'


def require_options(args, *, stubbed):
    timing = getattr(args,'paired_k32_timing',False)
    expected = (10,30,3.0) if timing else (0,0,0)
    if (args.artifact != REPLAY_ARTIFACT or args.groups != GROUP or args.ms != '1,512,2048'
            or not args.no_graph or args.ncu
            or (args.warmup,args.iters,args.power_s) != expected or not args.input_manifest
            or args.routing is not None or args.single_routing_file is not None
            or not args.profile_native_file or not args.paired_k32_source_sha256
            or len(args.paired_k32_source_sha256) != 64
            or any(c not in '0123456789abcdef' for c in args.paired_k32_source_sha256)):
        raise ValueError('paired-K32 numerics requires exact A8SE L10 TP2rank0 M1,512,2048, '
                         'pinned inputs/native, no graph/routing/timing/profiling overrides')
    if stubbed:
        raise ValueError('paired-K32 numerics refuses stubbed vLLM')


def numeric_certificate():
    raw=Path(NUMERIC_RECEIPT).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=NUMERIC_RECEIPT_SHA:
        raise ValueError('accepted numeric receipt differs')
    receipt=json.loads(raw)
    if not receipt['exact_words_equal'] or len(receipt['compared'])!=9 or len(receipt['synthetic_compared'])!=6:
        raise ValueError('numeric qualification is incomplete')
    return receipt


def _words(tensor):
    import torch
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _tensor_record(tensor):
    raw = _words(tensor)
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw),
            'shape': list(tensor.shape), 'dtype': str(tensor.dtype)}


def _reference_prefix(fn, store, x, ids, weights, actual, reference_holder):
    """Independent materializer/fp64 oracle for token0's eight real TP-cut experts.

    All full-shape output words are checked separately across binaries. This
    bounded reference qualifies only the stated first-token output population.
    """
    import torch
    from types import SimpleNamespace
    import importlib.util
    oracle_path = Path(__file__).resolve().parent.parent/'routed_pair_oracle.py'
    spec = importlib.util.spec_from_file_location('_paired_owned_fp64_oracle', oracle_path)
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    from tessera.serving.moe_route import _packed_group_shard_plan
    from tessera.serving.scheme import parse_tessera_expert_blob
    from tessera.serving.sharding import shard_parsed_roles
    from tessera.stock import materialize_stock

    if ids[0].tolist() != list(range(8)):
        raise ValueError('bounded real-wire reference requires balanced token0 experts0..7')
    ref = reference_holder.get('weights')
    if ref is None:
        scheme = store.schemes[MODULE]
        declared, roles = oracle.parse_roles(scheme, MODULE)
        ref = {p: {'W': [], 'm': []} for p in oracle.PROJ}
        for expert in range(8):
            for projection in oracle.PROJ:
                group = 'w2' if projection == 'down_proj' else 'w13'
                wire = store.get(f'{MODULE}.{expert}.{projection}.wire')
                blob = _words(wire)
                parsed = parse_tessera_expert_blob(blob, roles[projection],
                    f'{MODULE} {projection} reference expert{expert}', device=x.device)
                plan = _packed_group_shard_plan(declared, group, MODULE, 0, 2)
                local = shard_parsed_roles(parsed, plan)
                if len(local) != 1:
                    raise ValueError('materializing reference must have one TP-local role')
                pu = local[0][1]
                tiles = materialize_stock(pu.unit, pu.forests, pu.code)
                ref[projection]['W'].append(tiles['weight'].to(x.device).float().contiguous())
                ref[projection]['m'].append(tiles['weight_scale'].to(x.device).reshape(-1).float())
                del parsed, local, pu, tiles, wire, blob
        reference_holder['weights'] = ref
    native = fn.native_adapter
    original_inter, original_hidden = oracle.INTER, oracle.HIDDEN
    oracle.INTER, oracle.HIDDEN = int(native.down.cols), int(native.down.rows)
    # The oracle's teacher-forced stages use their public native interfaces;
    # apply observes the ACTUAL full paired batch's first-token result.
    def observed_apply(layer, prefix_x, prefix_weights, prefix_ids, *unused):
        return actual[:1].clone()
    method = SimpleNamespace(_native=native, apply=observed_apply)
    recorder = SimpleNamespace(take=lambda: [])
    try:
        result, _reference_tensors = oracle.oracle_case('e4m3', oracle.FAMILIES['e4m3'], None, method,
            ref, x[:1], ids[:1], weights[:1], 10.0, 288, recorder)
    finally:
        oracle.INTER, oracle.HIDDEN = original_inter, original_hidden
    # No serving method emit_route call occurs in bench_t8r's direct adapter.
    # Kernel dispatch is independently read from its production-call profile.
    result['route_telemetry']['scope'] = 'not applicable: direct operator harness; profile binds actual launch'
    stages_ok = all(stage['pass'] for stage in result['stages_teacher_forced'].values())
    prefix_exact = result['apply_vs_staged_composition_max_abs_diff'] == 0
    if not stages_ok or not prefix_exact:
        raise ValueError('independent teacher-forced bound or paired prefix/staged bits differ')
    result['bounded_numeric_pass'] = bool(stages_ok and prefix_exact)
    result['reference_scope'] = 'token0, experts0..7, real parsed and independently TP-sharded A8SE wires; fp64 derived bound'
    return result


def numeric_cell(fn, store, x, ids, weights, output_dir, *, kernel_profile, independent_reference=True, reference_holder=None):
    import torch
    from tessera import routed_fused as rf

    native = fn.native_adapter
    if type(native) is not rf.FusedRoutedWindowMoE or native.library != 'e4m3mma':
        raise ValueError('paired-K32 numerics requires the actual fused E4M3 MMA adapter')
    owner = type(native)
    original = owner._launch
    captured = {}
    def observe(instance, mode, *args, **kwargs):
        original(instance, mode, *args, **kwargs)
        if instance is not native:return
        if mode not in (0, 2) or mode in captured:
            raise ValueError('numeric forward must launch each routed role once')
        captured[mode] = kwargs['out'].detach().clone()
    # The production adapter is frozen. This one-process diagnostic observes
    # its class seam with an exact instance guard, never changes its fields,
    # and restores the real method before profiling or reference execution.
    owner._launch = observe
    try:
        first = fn(x, ids, weights)
        first_stages = captured.copy()
        captured.clear()
        second = fn(x, ids, weights)
        second_stages = captured.copy()
        torch.cuda.synchronize()
        if set(first_stages) != {0, 2} or set(second_stages) != {0, 2}:
            raise ValueError('numeric forward did not expose both native roles')
        if not torch.equal(first.view(torch.int16), second.view(torch.int16)):
            raise ValueError('repeated forward bits differ')
        for mode in (0, 2):
            if not torch.equal(first_stages[mode].view(torch.int16), second_stages[mode].view(torch.int16)):
                raise ValueError('repeated intermediate bits differ')
    finally:
        owner._launch = original
    routes = first_stages[2].reshape(x.shape[0], ids.shape[1], native.down.rows).float()
    reduced = torch.zeros_like(routes[:, 0])
    for route in range(ids.shape[1]):
        reduced = reduced + routes[:, route]
    if not torch.equal(reduced.bfloat16().view(torch.int16), first.view(torch.int16)):
        raise ValueError('native reduction differs from independent fixed-order FP32 sum')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, tensor in [('gate_up', first_stages[0]), ('down_routes', first_stages[2]), ('out', first)]:
        raw = _words(tensor)
        path = output_dir/(name+'.bin')
        with path.open('xb') as stream:
            stream.write(raw)
        records[name] = {**_tensor_record(tensor), 'path': str(path)}
    lib = rf._ext('e4m3mma')
    profile = kernel_profile(lambda: fn(x, ids, weights), reps=1, full_names=True)
    geometry = profile_geometry(native,x,ids,lib,profile)
    result = {'outputs': records, 'repeat_bits_equal': True, 'independent_token_sum_equal': True,
              'input_hashes': {name: _tensor_record(t) for name,t in [('x',x),('ids',ids),('weights',weights)]},
              'profile': profile, 'geometry': geometry, 'native_paired_build': bool(lib.PAIRED_K32_BUILD)}
    if independent_reference:
        result['independent_reference'] = _reference_prefix(fn, store, x, ids, weights, first,
            reference_holder if reference_holder is not None else {})
    return result


def profile_geometry(native,x,ids,lib,profile):
    from tessera import routed_fused as rf
    geometry={}
    for mode, bundle, slot in ((0, native.gate, native.slot_words_gate_up),
                              (2, native.down, native.slot_words_down)):
        bm = rf.superblock_rows('e4m3mma', mode, x.shape[0])
        rows = x.shape[0] if mode == 0 else x.shape[0]*ids.shape[1]
        paired = bool(lib.paired_k32_scope(True, True, mode, False, False, 4, False,
                                          bm, bundle.cols, rows, slot))
        symbol = f'routed_fused_kernel<true, {mode}, false, false, 4, false, {bm}, false, {str(paired).lower()}>'
        hits = [record for name,record in profile['top'].items() if symbol in name]
        if len(hits) != 1 or hits[0]['count_per_call'] != 1:
            raise ValueError('actual profiled native role differs from paired dispatch scope')
        geometry[str(mode)] = {'K': int(bundle.cols), 'N': int(bundle.rows), 'rows_x': int(rows),
                              'bm': bm, 'paired': paired, 'slot_words': slot,
                              'dynamic_shared_bytes': int(lib.launch_smem_bytes(mode,slot,bm,paired)),
                              'live_max_dynamic_shared_bytes': int(lib.max_dynamic_smem_bytes(x.device.index))}
    return geometry


def timing_cell(fn,x,ids,weights,*,certificate,time_events,summarize,kernel_profile,power):
    """Accepted numerics reused; raw guard outside the existing event owner."""
    import torch
    import time
    from tessera import routed_fused as rf
    native=fn.native_adapter
    if type(native) is not rf.FusedRoutedWindowMoE or native.library!='e4m3mma':
        raise ValueError('timing requires the qualified actual adapter')
    call=lambda:fn(x,ids,weights)
    raw=_words(call())
    expected=next(v for v in certificate['compared'] if v['M']==x.shape[0] and v['role']=='out')
    stored=(Path(NUMERIC_RECEIPT).parent/'baseline/numeric-words'/str(x.shape[0])/'out.bin').read_bytes()
    if (len(raw)!=expected['bytes'] or hashlib.sha256(raw).hexdigest()!=expected['sha256']
            or hashlib.sha256(stored).hexdigest()!=expected['sha256'] or raw!=stored):
        raise ValueError('timing output differs from accepted stored numeric words')
    started=time.time()
    samples=time_events(call,10,30)
    wall_window=[started,time.time()]
    # Profiling and power runs are separate from all unprofiled CUDA events.
    profile=kernel_profile(call,reps=3,full_names=True)
    lib=rf._ext('e4m3mma')
    geometry=profile_geometry(native,x,ids,lib,profile)
    sampled=power.sample_during(call,3.0,capture_series=True)
    sampled['calls_per_j']=None
    sampled['energy_status']='HOLD: numeric operator timing; clock/sample attribution not accepted'
    return {'wall':summarize(samples),'raw_events_ms':samples,'wall_window_unix':wall_window,
            'profile':profile,'geometry':geometry,'power':sampled,
            'out_sha256':expected['sha256'],'numeric_receipt_sha256':NUMERIC_RECEIPT_SHA,
            'native_paired_build':bool(lib.PAIRED_K32_BUILD)}


def compare_reports(baseline, candidate):
    """Compare the exact retained raw words, not worker-declared status strings."""
    for report in (baseline, candidate):
        direct = report['meta'].get('direct_input_bindings', {})
        if (direct.get('transport') != 'direct-vllm-held-original-fds'
                or not direct.get('before') or direct.get('before') != direct.get('after')
                or direct.get('manifest_sha256') != report['meta']['paired_k32']['input_manifest_sha256']):
            raise ValueError('direct input ownership is not stable through the numeric arm')
        owner = report['meta'].get('native_code_artifact', {})
        if not owner.get('before_load_sha256') or owner.get('before_load_sha256') != owner.get('after_load_sha256') or owner.get('before_load_sha256') != owner.get('after_profile_sha256'):
            raise ValueError('native code identity is not stable through the numeric arm')
    if baseline['meta']['paired_k32']['input_manifest_sha256'] != candidate['meta']['paired_k32']['input_manifest_sha256']:
        raise ValueError('paired numeric arms have different pinned input manifests')
    if baseline['meta']['direct_input_bindings']['before'] != candidate['meta']['direct_input_bindings']['before']:
        raise ValueError('paired numeric arms consumed different original file identities')
    a = baseline['results']; b = candidate['results']
    if len(a) != 1 or len(b) != 1 or not a[0]['ok'] or not b[0]['ok'] or set(a[0]['cells']) != {'1','512','2048'} or set(b[0]['cells']) != {'1','512','2048'}:
        raise ValueError('paired numeric output population differs')
    compared = []
    for key in ('1','512','2048'):
        left, right = a[0]['cells'][key], b[0]['cells'][key]
        for cell, expected_build in ((left,False),(right,True)):
            if cell.get('native_paired_build') is not expected_build or not cell.get('repeat_bits_equal') or not cell.get('independent_token_sum_equal') or not cell.get('independent_reference', {}).get('bounded_numeric_pass'):
                raise ValueError('numeric arm lacks build/repeat/reduction/independent-reference proof')
        for mode in ('0','2'):
            if left['geometry'][mode]['paired'] is not False or right['geometry'][mode]['paired'] is not (key != '1') or left['geometry'][mode]['bm'] != 128 or right['geometry'][mode]['bm'] != 128:
                raise ValueError('actual paired/original dispatch population differs')
        if left['input_hashes'] != right['input_hashes']:
            raise ValueError('paired numeric inputs differ')
        for role in ('gate_up','down_routes','out'):
            x, y = left['outputs'][role], right['outputs'][role]
            if (x['shape'],x['dtype'],x['bytes'],x['sha256']) != (y['shape'],y['dtype'],y['bytes'],y['sha256']):
                raise ValueError(f'paired numeric output differs: M{key}/{role}')
            raw_x, raw_y = Path(x['path']).read_bytes(), Path(y['path']).read_bytes()
            if len(raw_x) != x['bytes'] or len(raw_y) != y['bytes'] or hashlib.sha256(raw_x).hexdigest()!=x['sha256'] or hashlib.sha256(raw_y).hexdigest()!=y['sha256'] or raw_x != raw_y:
                raise ValueError('retained output words do not bind numeric result')
            compared.append({'M':int(key),'role':role,'bytes':len(raw_x),'sha256':x['sha256']})
    synthetic = []
    for report in (baseline,candidate):
        if set(report['results'][0].get('synthetic_controls',{})) != {'128','192'}:
            raise ValueError('synthetic K128/K192 control population differs')
    for K in ('128','192'):
        left=a[0]['synthetic_controls'][K];right=b[0]['synthetic_controls'][K]
        if left['input_hashes'] != right['input_hashes'] or left.get('diagnostic_grid')!=1 or right.get('diagnostic_grid')!=1:
            raise ValueError('synthetic inputs/grid differ')
        for mode in ('0','2'):
            if left['geometry'][mode]['paired'] is not False or right['geometry'][mode]['paired'] is not (mode=='0' or K=='192'):
                raise ValueError('synthetic fallback/paired dispatch differs')
        for role in ('gate_up','down_routes','out'):
            x=left['outputs'][role];y=right['outputs'][role]
            raw_x=Path(x['path']).read_bytes();raw_y=Path(y['path']).read_bytes()
            if (x['shape'],x['dtype'],x['bytes'],x['sha256']) != (y['shape'],y['dtype'],y['bytes'],y['sha256']) or raw_x!=raw_y or len(raw_x)!=x['bytes'] or hashlib.sha256(raw_x).hexdigest()!=x['sha256']:
                raise ValueError('synthetic retained output bits differ')
            synthetic.append({'K_down':int(K),'role':role,'bytes':len(raw_x),'sha256':x['sha256']})
    return {'exact_words_equal': True, 'compared': compared, 'synthetic_compared':synthetic,
            'scope':'balanced real A8SE L10 TP2rank0 plus separately declared synthetic one-CTA controls; numerical operator only'}


def synthetic_controls(output_dir, *, kernel_profile):
    """Small existing encoded-wire fixtures, distinct from real A8SE evidence."""
    import torch
    import test_routed_fused_window as fixture
    from tessera import routed_fused as rf
    records = {}
    original_sm_count = rf._sm_count
    # Diagnostic only: one persistent CTA forces item-slot reuse at K192.
    rf._sm_count = lambda index: 1
    try:
        for inter in (128,192):
            stacks = fixture._stacks('e4m3',hidden=256,inter=inter,experts=5,seed=740+inter,cut=True)
            bundles = fixture._bundles('e4m3',stacks)
            native = fixture._fused(bundles)
            ids,weights = fixture._routes(512,3,1100+inter,experts=5)
            generator = torch.Generator(device='cpu').manual_seed(1600+inter)
            x = torch.randn((512,256),generator=generator).bfloat16().cuda()
            def call(a,i,w):
                return native(a,i,w,swiglu_limit=10.0,apply_router_weight_on_input=False)
            call.native_adapter = native
            record = numeric_cell(call,None,x,ids,weights,Path(output_dir)/('downK'+str(inter)),
                kernel_profile=kernel_profile,independent_reference=False)
            record['diagnostic_grid'] = 1
            record['scope'] = 'existing synthetic encoded TP-history fixtures; not real A8SE or served shape promotion'
            records[str(inter)] = record
            del call,native,bundles,stacks,x,ids,weights
    finally:
        rf._sm_count = original_sm_count
    return records
