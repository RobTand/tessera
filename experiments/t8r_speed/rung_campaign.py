"""Journal public PrismaBuild campaign keys immediately; PB owns all pacing."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--journal',required=True)
    ap.add_argument('--max-inflight',type=int,required=True)
    ap.add_argument('--wait-s',type=int,default=172800)
    ap.add_argument('--require-data-manifest',action='store_true')
    args=ap.parse_args()
    journal=Path(args.journal)
    journal.parent.mkdir(parents=True,exist_ok=True)
    state={'manifest':args.manifest,'max_inflight':args.max_inflight,'publications':[],
           'completion_client':'published pbcampaign.py','returncode':None}
    def save():
        temporary=journal.with_suffix(journal.suffix+'.tmp')
        temporary.write_text(json.dumps(state,indent=2,allow_nan=False))
        temporary.replace(journal)
    command=[sys.executable,'/mnt/shared/prismabuild-fleet/repo/tools/pbcampaign.py',
             '--max-inflight',str(args.max_inflight),'--wait-s',str(args.wait_s)]
    if args.require_data_manifest:command.append('--require-data-manifest')
    command.append(args.manifest)
    save()
    with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          text=True,bufsize=1) as child:
        try:
            for line in child.stdout:
                print(line,end='',flush=True)
                match=re.fullmatch(r'pbcampaign: row (\d+) (submitted|attached|cache_hit) ([0-9a-f]{64})\n?',line)
                if match:
                    state['publications'].append({'row':int(match[1]),'status':match[2],'action_key':match[3]})
                    save()
            state['returncode']=child.wait()
            save()
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait()
    return state['returncode']


if __name__=='__main__':raise SystemExit(main())
