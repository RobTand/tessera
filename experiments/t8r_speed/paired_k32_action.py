"""One finite baseline/candidate run through existing bench/native/input owners.

Stock vLLM custom ops run directly under its execution exemption. Preparation
and CPU controls use PrismaBuild. This is not a dispatcher or residency cache.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import importlib.util
import re
import secrets
import signal
import time

from paired_k32_qualification import compare_reports


def owned_cleanup(arm_out, *, run=subprocess.run):
    """Remove only the exact CID whose independent owner label matches."""
    cid_path=arm_out/'owned.cid'
    if not cid_path.exists():return {'state':'no CID created'}
    cid=cid_path.read_text().strip()
    token=(arm_out/'owner-token.txt').read_text().strip()
    if not re.fullmatch('[0-9a-f]{64}',cid) or not re.fullmatch('[0-9a-f]{32}',token):
        raise ValueError('invalid owned container identity')
    inspect=run(['docker','inspect',cid],capture_output=True,text=True,timeout=5)
    if inspect.returncode:
        # --rm normally removes a completed container. A daemon failure is
        # distinct from absence and must not silently certify cleanup.
        if 'No such object' not in inspect.stderr and 'No such container' not in inspect.stderr:
            raise RuntimeError('owned container inspection failed: '+inspect.stderr)
        return {'state':'already absent','cid':cid,'owner':token}
    objects=json.loads(inspect.stdout)
    if (len(objects)!=1 or objects[0]['Id']!=cid
            or objects[0]['Config'].get('Labels',{}).get('tessera.paired_numeric_owner')!=token):
        raise ValueError('foreign container: cleanup refused')
    removed=run(['docker','rm','-f',cid],capture_output=True,text=True,timeout=15)
    if removed.returncode:raise RuntimeError('owned container removal failed: '+removed.stderr)
    return {'state':'removed owned CID','cid':cid,'owner':token}


def run_direct_arm(argv,env,log,arm_out,guard_path,*,pressure=None,popen=subprocess.Popen):
    # Reuse the existing routed-load instrument's host UMA/PSI reader.
    if pressure is None:
        path=Path(__file__).resolve().parent.parent/'bench_routed_load.py'
        spec=importlib.util.spec_from_file_location('_paired_host_pressure_owner',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        pressure=module._host_pressure
    floor=24*(1<<30)
    available,psi=pressure()
    if available < floor+16*(1<<30) or psi>=20:
        raise RuntimeError('direct numeric launch lacks 40GiB host headroom or PSI full avg10<20')
    child=popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    started=time.monotonic()
    try:
        with guard_path.open('x') as guard:
            while child.poll() is None:
                available,psi=pressure()
                elapsed=time.monotonic()-started
                guard.write(json.dumps({'elapsed_s':elapsed,'available_bytes':available,
                    'psi_full_avg10':psi,'floor_bytes':floor})+'\n');guard.flush()
                if available < floor or psi>=20 or elapsed>=240:
                    raise RuntimeError('direct numeric memory/pressure/time bound reached')
                time.sleep(1)
        return child.returncode
    finally:
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
        cleanup=owned_cleanup(arm_out)
        (arm_out.parent/(arm_out.name+'-cleanup.json')).write_text(json.dumps(cleanup,indent=2)+'\n')


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
    if os.environ.get('PRISMABUILD_ACTION_KEY') or os.environ.get('BENCH_STRICT_STAGED'):
        raise ValueError('stock vLLM numeric execution must run directly, outside PB admission')
    def interrupted(signum,frame):raise KeyboardInterrupt('numeric run interrupted: '+str(signum))
    signal.signal(signal.SIGTERM,interrupted)
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
                 BENCH_EXPECT_LIBRARY_SHA256=digest,BENCH_DIRECT_VLLM='1',
                 BENCH_OWNER_TOKEN=secrets.token_hex(16),
                 BENCH_RO_MOUNTS=' '.join(str(p.parent) for p in natives))
        argv=['bash','experiments/t8r_speed/bench_t8r.sh','.',str(out/arm),
            '--paired-k32-numerics','--direct-vllm-inputs','--paired-k32-source-sha256',bindings['src/tessera/serving/csrc/routed_fused_window.cu'],
            '--artifact','/mnt/shared/tessera-runs/moe/glm53-a8-bf16menu-20260930/release/exported',
            '--groups','experts.R1024.L10','--ms','1,512,2048','--no-graph',
            '--power-s','0','--warmup','0','--iters','0','--input-manifest',args.input_manifest,
            '--profile-native-file',path]
        with (out/(arm+'.log')).open('xb') as log:
            rc=run_direct_arm(argv,env,log,out/arm,out/(arm+'-memory.jsonl'))
        if rc:
            raise RuntimeError(arm+' numeric arm exited '+str(rc))
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
