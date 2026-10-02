"""One resident stock/L0 paired cell; CUDA-event ABBA, ambient CPU explicitly recorded."""
from __future__ import annotations
import argparse,json,os,statistics,time
import math
from pathlib import Path
import torch
from d1_bench import make_cache,make_indices,HEADS,D_LATENT,WIDTH,SM_SCALE,PowerSampler
from library import Library

def bit_equal(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
def build_case(library, stock, T, E, mode, reference_library=None):
    gen=torch.Generator(device='cuda').manual_seed(1000+E+100000*T)
    kv=make_cache(E,gen,'cuda');idx=make_indices(E,T,gen,'cuda',mode)
    q=(torch.randn(T,HEADS,D_LATENT,generator=gen,device='cuda')*2).to(torch.bfloat16)
    ws=torch.zeros(128*1024*1024,dtype=torch.uint8,device='cuda')
    outputs={arm:(torch.empty_like(q),torch.empty(T,HEADS,dtype=torch.float32,device='cuda')) for arm in ['stock','l0','copy']}
    def invoke(arm):
        out,lse=outputs[arm]
        if arm!='stock':return library.call(0 if arm=='copy' else 1,q,kv,idx,out,lse)
        return stock(query=q.unsqueeze(1),kv_cache=kv.unsqueeze(1),workspace_buffer=ws,
            qk_nope_head_dim=256,kv_lora_rank=D_LATENT,qk_rope_head_dim=0,block_tables=idx.unsqueeze(1),
            seq_lens=None,max_seq_len=WIDTH,out=out.unsqueeze(1),bmm1_scale=SM_SCALE,bmm2_scale=1.,
            sparse_mla_top_k=WIDTH,kv_scale_format='arbitrary_fp32',lse=lse,return_lse=True,return_lse_base='base2')
    for arm in outputs:invoke(arm)
    torch.cuda.synchronize()
    for arm in ['copy','l0']:
        if not all(bit_equal(a,b) for a,b in zip(outputs['stock'],outputs[arm])):raise RuntimeError('bitwise failure '+arm)
    if reference_library is not None:
        reference_library.call(1,q,kv,idx,*outputs['copy']);torch.cuda.synchronize()
        if not all(bit_equal(a,b) for a,b in zip(outputs['stock'],outputs['copy'])):raise RuntimeError('bitwise failure retained reference')
    def paired(arm):
        if reference_library is not None and arm=='stock':
            return reference_library.call(1,q,kv,idx,*outputs['copy'])
        return invoke(arm)
    return paired, idx

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--shape',required=True);p.add_argument('--index-mode',default='pools',choices=['pools','random'])
    p.add_argument('--iters',type=int,default=100);p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--min-arm-s',type=float,default=0)
    p.add_argument('--build-dir');p.add_argument('--build-manifest-sha256');p.add_argument('--p0-buffers',action='store_true')
    p.add_argument('--reference-build-dir');p.add_argument('--reference-manifest-sha256')
    p.add_argument('--netdata-host',action='append',default=[],metavar='NAME=ADDRESS');args=p.parse_args()
    if bool(args.build_dir)!=bool(args.build_manifest_sha256) or bool(args.reference_build_dir)!=bool(args.reference_manifest_sha256):p.error('retained builds require directory AND manifest SHA-256')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as stock
    library=Library(args.build_dir or root/'build',p0_buffers=args.p0_buffers,
                    require_retained=bool(args.build_dir),retained_manifest_sha256=args.build_manifest_sha256)
    reference=Library(args.reference_build_dir,require_retained=True,
                      retained_manifest_sha256=args.reference_manifest_sha256) if args.reference_build_dir else None
    shapes=[tuple(map(int,shape.split('@'))) for shape in args.shape.split(',')]
    cells=[build_case(library,stock,T,E,args.index_mode,reference) for T,E in shapes]
    T=sum(shape[0] for shape in shapes);E=max(shape[1] for shape in shapes)
    def invoke(arm):
        for call, _ in cells:call(arm)
    graphs={}
    for arm in ['stock','l0']:
        for _ in range(4):invoke(arm)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):invoke(arm)
        graphs[arm]=graph
    torch.cuda.synchronize()
    iterations={}
    for arm in graphs:
        # Startup/JIT and settling are outside the steady arm windows.
        until=time.monotonic()+max(0,min(2,args.min_arm_s))
        while time.monotonic()<until:graphs[arm].replay();torch.cuda.synchronize()
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(10):graphs[arm].replay()
        b.record();b.synchronize()
        iterations[arm]=max(args.iters,int(args.min_arm_s*1000/(a.elapsed_time(b)/10))+1)
    legs=[]
    for cycle in range(args.cycles):
      for pos,arm in enumerate(['stock','l0','l0','stock']):
        start=time.time();evt0=torch.cuda.Event(enable_timing=True);evt1=torch.cuda.Event(enable_timing=True)
        with PowerSampler() as power:
            evt0.record()
            for _ in range(iterations[arm]):graphs[arm].replay()
            evt1.record();evt1.synchronize()
        end=time.time();mean_ms=evt0.elapsed_time(evt1)/iterations[arm]
        samples=power.samples;mean_w=sum(s[1] for s in samples)/len(samples) if samples else None
        joules=mean_ms/1000*mean_w if mean_w is not None else None
        legs.append({'cycle':cycle,'position':pos,'arm':arm,'start_unix':start,'end_unix':end,
            'cuda_ms_per_call':mean_ms,'iterations':iterations[arm],'power':power.summary(),'power_samples':samples,
            'event_energy_j_estimate_per_call':joules,'queries_per_joule_estimate':T/joules if joules else None})
    # Both profiles use the same resident workload, after timing. They are not speed samples.
    profiles={}
    for arm in ['stock','l0']:
        path=root/f'{arm}.pt.trace.json'
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            for _ in range(3):invoke(arm)
            torch.cuda.synchronize()
        prof.export_chrome_trace(str(path));kernels={}
        for event in prof.events():
            if getattr(event.device_type,'name','')=='CUDA':kernels.setdefault(event.name,[]).append(event.time_range.elapsed_us())
        profiles[arm]={'trace':str(path),'kernels':{k:{'count':len(v),'mean_us':statistics.mean(v)} for k,v in kernels.items()}}
    medians={arm:statistics.median(l['cuda_ms_per_call'] for l in legs if l['arm']==arm) for arm in ['stock','l0']}
    energies={arm:None for arm in ['stock','l0']}  # instrument agreement is not qualified
    report={'schema':'tessera.mla_mask_abba.v1','T':T,'E':E,'index_mode':args.index_mode,
        'shapes':shapes,'masked_tile_fractions':[float((idx.view(shape[0],WIDTH//64,64)<0).all(-1).float().mean()) for shape,(_,idx) in zip(shapes,cells)],
        'placement':os.environ.get('MLA_PLACEMENT','unqualified'),
        'measurement_scope':'paired resident CUDA kernels; eager serving gate remains separate',
        'host':os.environ.get('HOST_NAME'),'started_unix':legs[0]['start_unix'],'finished_unix':time.time(),
        'bitwise_output_and_lse':True,'build_manifest':library.build_manifest,'native_identity':library.identity,
        'arm_implementations':{'stock':'retained_l0' if reference else 'stock_flashinfer','l0':'pass_buffers' if args.p0_buffers else 'l0'},
        'reference_build_manifest':reference.build_manifest if reference else None,
        'energy_status':'HOLD: event/power estimates are descriptive, no qualified work/J',
        'legs':legs,'median_cuda_ms':medians,'l0_over_stock':medians['l0']/medians['stock'],
        'median_event_energy_j_estimate':energies,'profiles':profiles}
    (root/'abba.json').write_text(json.dumps(report,indent=2)+'\n')
    if args.netdata_host:
        from box_power_window import collect
        after=math.floor(report['started_unix']);before=math.ceil(report['finished_unix'])
        boxes={}
        for spec in args.netdata_host:
            name,addr=spec.split('=',1)
            if name in boxes:raise ValueError('duplicate Netdata host '+name)
            boxes[name]=collect(addr,after,before,1024)
        (root/'both-host-netdata.json').write_text(json.dumps({'interval':[after,before],'boxes':boxes},indent=2)+'\n')
        errors=[f'{host}/{name}: {value["error"]}' for host,series in boxes.items() for name,value in series.items() if 'error' in value]
        if errors:raise RuntimeError('required Netdata collection incomplete: '+ '; '.join(errors))
    print(json.dumps({k:report[k] for k in ['T','E','index_mode','median_cuda_ms','l0_over_stock','arm_implementations']}))
if __name__=='__main__':main()
