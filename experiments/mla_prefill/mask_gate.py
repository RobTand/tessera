"""Bitwise gate for stock/copy/L0 over served chunks, masks and LSE (no speed claim)."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
import torch
from d1_bench import make_cache,make_indices,HEADS,D_LATENT,WIDTH,SM_SCALE,SHAPES
from library import Library

def eq(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--mutation',action='store_true');p.add_argument('--shapes',default=SHAPES);p.add_argument('--edges-only',action='store_true')
    p.add_argument('--build-dir');p.add_argument('--build-manifest-sha256')
    p.add_argument('--p0-buffers',action='store_true');p.add_argument('--p0-wrong-pass',action='store_true');args=p.parse_args()
    if bool(args.build_dir) != bool(args.build_manifest_sha256):p.error('retained build requires directory AND manifest SHA-256')
    root=Path(args.out);root.mkdir(parents=True,exist_ok=True)
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla as stock
    library=Library(args.build_dir or root/'build',mutation=args.mutation,
                    p0_buffers=args.p0_buffers,p0_wrong_pass=args.p0_wrong_pass,
                    require_retained=bool(args.build_dir),retained_manifest_sha256=args.build_manifest_sha256)
    ws=torch.zeros(128*1024*1024,dtype=torch.uint8,device='cuda')
    shapes=[tuple(map(int,s.split('@'))) for s in args.shapes.split(',')]
    rows=[];passed=True
    cases=['pools','random']
    if args.edges_only:shapes=[(65,256)];cases=['all-masked','tile-boundaries','hole-tile','signed-zero','extreme-scale']
    for T,E in shapes:
      for case in cases:
        gen=torch.Generator(device='cuda').manual_seed(1000+E+100000*T)
        kv=make_cache(E,gen,'cuda');idx=make_indices(E,T,gen,'cuda',case if case in ['pools','random'] else 'pools')
        q=(torch.randn(T,HEADS,D_LATENT,generator=gen,device='cuda')*2).to(torch.bfloat16)
        if case=='all-masked':idx.fill_(-1)
        if case=='tile-boundaries':
            idx.fill_(-1)
            for offset in [0,63,64,127,128,2175]:idx[:,offset]=offset%E
        if case=='hole-tile':idx[:,64:128]=-1
        if case=='signed-zero':
            q.zero_();q[:,:,::2]=-0.0
            kv[:,:,:512]=0;kv[:,:,::2][:,:,:256]=128
        if case=='extreme-scale':q[:,:16]*=256;q[:,16:]/=256
        ref=torch.empty_like(q);ref_lse=torch.empty(T,HEADS,dtype=torch.float32,device='cuda')
        stock(query=q.unsqueeze(1),kv_cache=kv.unsqueeze(1),workspace_buffer=ws,
              qk_nope_head_dim=256,kv_lora_rank=D_LATENT,qk_rope_head_dim=0,
              block_tables=idx.unsqueeze(1),seq_lens=None,max_seq_len=WIDTH,
              out=ref.unsqueeze(1),bmm1_scale=SM_SCALE,bmm2_scale=1.0,
              sparse_mla_top_k=WIDTH,kv_scale_format='arbitrary_fp32',
              lse=ref_lse,return_lse=True,return_lse_base='base2')
        torch.cuda.synchronize()
        row={'T':T,'E':E,'case':case,'variants':{}}
        for variant in [0,1]:
            out=torch.full_like(q,float('nan'));lse=torch.full_like(ref_lse,float('nan'))
            library.call(variant,q,kv,idx,out,lse);torch.cuda.synchronize()
            match=eq(out,ref) and eq(lse,ref_lse)
            row['variants'][str(variant)]={'output_bitwise':eq(out,ref),'lse_bitwise':eq(lse,ref_lse),
               'output_different_elements':int((out.view(torch.int16)!=ref.view(torch.int16)).sum()),
               'lse_different_elements':int((lse.view(torch.int32)!=ref_lse.view(torch.int32)).sum())}
            passed=passed and match
        rows.append(row);print(json.dumps(row),flush=True)
    report={'schema':'tessera.mla_mask_gate.v1','passed':passed,'rows':rows,'library':library.path,
            'source_sha256':library.source_sha256,'native_identity':library.identity,'build_manifest':library.build_manifest,
            'mutation':args.mutation,'p0_buffers':args.p0_buffers,'p0_wrong_pass':args.p0_wrong_pass,
            'mutation_detected': (args.mutation or args.p0_wrong_pass) and any(not r['variants']['1']['output_bitwise'] or not r['variants']['1']['lse_bitwise'] for r in rows),
            'host':os.environ.get('HOST_NAME'),'finished_unix':time.time()}
    (root/'gate.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if passed else 1
if __name__=='__main__':raise SystemExit(main())
