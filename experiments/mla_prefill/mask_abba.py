"""One resident stock/L0 paired cell; CUDA-event ABBA, ambient CPU explicitly recorded."""
from __future__ import annotations
import argparse,json,os,statistics,time
from pathlib import Path
import torch
from d1_bench import make_cache,make_indices,HEADS,D_LATENT,WIDTH,SM_SCALE,PowerSampler
from library import Library

def bit_equal(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--shape',required=True);p.add_argument('--index-mode',default='pools',choices=['pools','random'])
    p.add_argument('--iters',type=int,default=100);p.add_argument('--cycles',type=int,default=3);args=p.parse_args()
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True);T,E=map(int,args.shape.split('@'))
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as stock
    library=Library(root/'build');gen=torch.Generator(device='cuda').manual_seed(1000+E+100000*T)
    kv=make_cache(E,gen,'cuda');idx=make_indices(E,T,gen,'cuda',args.index_mode)
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
    graphs={}
    for arm in ['stock','l0']:
        for _ in range(4):invoke(arm)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):invoke(arm)
        graphs[arm]=graph
    torch.cuda.synchronize()
    legs=[]
    for cycle in range(args.cycles):
      for pos,arm in enumerate(['stock','l0','l0','stock']):
        start=time.time();evt0=torch.cuda.Event(enable_timing=True);evt1=torch.cuda.Event(enable_timing=True)
        with PowerSampler() as power:
            evt0.record()
            for _ in range(args.iters):graphs[arm].replay()
            evt1.record();evt1.synchronize()
        end=time.time();mean_ms=evt0.elapsed_time(evt1)/args.iters
        samples=power.samples;mean_w=sum(s[1] for s in samples)/len(samples) if samples else None
        joules=mean_ms/1000*mean_w if mean_w is not None else None
        legs.append({'cycle':cycle,'position':pos,'arm':arm,'start_unix':start,'end_unix':end,
            'cuda_ms_per_call':mean_ms,'power':power.summary(),'power_samples':samples,
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
    energies={arm:statistics.median([l['event_energy_j_estimate_per_call'] for l in legs if l['arm']==arm and l['event_energy_j_estimate_per_call'] is not None]) for arm in ['stock','l0']}
    report={'schema':'tessera.mla_mask_abba.v1','T':T,'E':E,'index_mode':args.index_mode,
        'masked_tile_fraction':float((idx.view(T,WIDTH//64,64)<0).all(-1).float().mean()),
        'placement':'exclusive GPU, ambient CPU; ordinary PB action (no CPU isolation claim)',
        'host':os.environ.get('HOST_NAME'),'started_unix':legs[0]['start_unix'],'finished_unix':time.time(),
        'bitwise_output_and_lse':True,'build_manifest':library.build_manifest,'native_identity':library.identity,
        'legs':legs,'median_cuda_ms':medians,'l0_over_stock':medians['l0']/medians['stock'],
        'median_event_energy_j_estimate':energies,'profiles':profiles}
    (root/'abba.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['T','E','index_mode','median_cuda_ms','l0_over_stock','median_event_energy_j_estimate']}))
if __name__=='__main__':main()
