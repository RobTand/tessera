"""One coupled, finite baseline/candidate action; PB owns its placement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from paired_k32_qualification import compare_reports


def identities(paths):
    result={}
    for path in paths:
        path=Path(path);st=path.stat()
        result[str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes':st.st_size,'mtime_ns':st.st_mtime_ns}
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True)
    parser.add_argument('--source-manifest',required=True)
    parser.add_argument('--input-manifest',required=True)
    parser.add_argument('--baseline-so',required=True)
    parser.add_argument('--baseline-sha256',required=True)
    parser.add_argument('--candidate-so',required=True)
    parser.add_argument('--candidate-sha256',required=True)
    args=parser.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    bindings=json.loads(Path(args.source_manifest).read_bytes())
    def source_check():
        for name,digest in bindings.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
                raise ValueError('numeric action source differs: '+name)
    source_check()
    natives=[Path(args.baseline_so),Path(args.candidate_so)]
    paths=[p for native in natives for p in (native,native.parent/'routed_fused_window.cuda.o',native.parent/'build.ninja',native.parent/'.ninja_log')]
    before=identities(paths)
    for path,digest in [(args.baseline_so,args.baseline_sha256),(args.candidate_so,args.candidate_sha256)]:
        if before[path]['sha256']!=digest:
            raise ValueError('original native bank differs before numeric action')
    (out/'native-before.json').write_text(json.dumps(before,indent=2)+'\n')
    results={}
    for arm,choice,path,digest in [('baseline','0',args.baseline_so,args.baseline_sha256),
                                 ('candidate','1',args.candidate_so,args.candidate_sha256)]:
        env=dict(os.environ,TESSERA_ROUTED_FUSED_PAIRED_K32=choice,
                 BENCH_EXPECT_LIBRARY_SHA256=digest)
        argv=['bash','experiments/t8r_speed/bench_t8r.sh','.',str(out/arm),
            '--paired-k32-numerics','--paired-k32-source-sha256',bindings['src/tessera/serving/csrc/routed_fused_window.cu'],
            '--artifact','/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported',
            '--groups','experts.R1024.L10','--ms','1,512,2048','--no-graph',
            '--power-s','0','--warmup','0','--iters','0','--input-manifest',args.input_manifest,
            '--profile-native-file',path]
        with (out/(arm+'.log')).open('xb') as log:
            run=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT)
        if run.returncode:
            raise RuntimeError(arm+' numeric arm exited '+str(run.returncode))
        source_check()
        results[arm]=json.loads((out/arm/'bench_t8r.json').read_bytes())
    result=compare_reports(results['baseline'],results['candidate'])
    after=identities(paths)
    if before!=after:
        raise ValueError('native ELF/object/flags/ninja bank changed during numeric action')
    (out/'native-after.json').write_text(json.dumps(after,indent=2)+'\n')
    result['source_bindings']=bindings
    result['native_files']=before
    result['input_manifest_sha256']=hashlib.sha256(Path(args.input_manifest).read_bytes()).hexdigest()
    result['gpu_timing_claim']=None
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=='__main__':
    sys.exit(main())
