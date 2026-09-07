"""Install canonical Tessera inside the actual upstream image, without core edits.

Run through PB's owned docker run. The launcher must use the official base
image itself; its image identity is checked outside and supplied explicitly.
Evidence goes to a fresh per-job directory; historical receipts are untouched.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

BASE='vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33'
SOURCE='382a1a97dc89618173a2c0799ac569d473b9dc2a'
SOURCE_SHA='2703772a800c14ac1288df49e652913dc6d3678ec8a4c07ffd1f8ebe8372dbe1'
CORE_SHA='d4aa04edf388da29679585f4688a2b62d879bcc65349866ba37ed96b7b1e467e'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def files(root):
    return {str(p.relative_to(root)):{'sha256':digest(p),'bytes':p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,required=True)
    parser.add_argument('--launcher-image-id',required=True)
    parser.add_argument('--launcher-image-inspect',type=Path,required=True)
    parser.add_argument('--runtime-uid',type=int,default=1000)
    parser.add_argument('--runtime-gid',type=int,default=1000)
    parser.add_argument('--source-archive',type=Path,default=Path('/mnt/shared/tessera-clean-runtime-20260907/tessera-382-source.tar'))
    parser.add_argument('--core-manifest',type=Path,default=Path('/mnt/shared/tessera-clean-runtime-20260907/official-primary/runtime-inventory.json'))
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    inspected=json.loads(args.launcher_image_inspect.read_text())
    if isinstance(inspected,list):
        assert len(inspected)==1
        inspected=inspected[0]
    assert inspected['Id']==args.launcher_image_id,'Launcher image ID must match actual image inspection'
    assert BASE in inspected.get('RepoDigests',[]),'Official immutable base must appear in inspected RepoDigests'
    assert digest(args.source_archive)==SOURCE_SHA
    assert digest(args.core_manifest)==CORE_SHA
    assert not (args.evidence_dir/'per-job-runtime.json').exists(),'Refuse to overwrite prior runtime evidence'
    stock=json.loads(args.core_manifest.read_text())
    core=Path(importlib.util.find_spec('vllm').origin).parent
    assert files(core)==stock['files'],'Initial installed vLLM differs from the attested official image'
    with tempfile.TemporaryDirectory(prefix='tessera-canonical-') as temp:
        with tarfile.open(args.source_archive) as archive:archive.extractall(temp,filter='data')
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-build-isolation','--no-cache-dir',temp],check=True)
    assert files(core)==stock['files'],'Plugin installation changed vLLM core files'
    plugin=Path(importlib.util.find_spec('tessera').origin).parent
    plugin_files=files(plugin)
    entries=[{'name':e.name,'value':e.value} for e in importlib.metadata.entry_points(group='vllm.general_plugins') if e.name=='tessera']
    assert entries==[{'name':'tessera','value':'tessera.serving:register'}]
    record={'launcher_image_inspect_sha256':digest(args.launcher_image_inspect),'launcher_image_inspect':inspected,'registry_base':BASE,'launcher_declared_image_id':args.launcher_image_id,'identity_scope':'The PB Docker launch binds the image reference/ID; this installer verifies its complete vLLM file manifest before and after plugin installation.',
            'upstream_commit':'1970f3ed4be7fa8620e4ddc4a12c36a8384cfc27','core_manifest_sha256':CORE_SHA,'core_files_unchanged':len(stock['files']),
            'plugin_source_commit':SOURCE,'plugin_archive_sha256':SOURCE_SHA,'plugin_files':plugin_files,'plugin_entrypoints':entries,'vllm_version':importlib.metadata.version('vllm'),'tessera_version':importlib.metadata.version('tessera-quant'),'affinity':sorted(os.sched_getaffinity(0))}
    if os.getuid()==0:
        os.setgroups([]);os.setgid(args.runtime_gid);os.setuid(args.runtime_uid)
    else:
        assert os.getuid()==args.runtime_uid and os.getgid()==args.runtime_gid
    args.evidence_dir.mkdir(parents=True,exist_ok=True)
    path=args.evidence_dir/'per-job-runtime.json';path.write_text(json.dumps(record,indent=2))
    print(json.dumps({'artifact':str(path),'sha256':digest(path),'bytes':path.stat().st_size,'core_files_unchanged':record['core_files_unchanged'],'registry_base':BASE,'launcher_declared_image_id':args.launcher_image_id}),flush=True)
    command=args.command
    if command and command[0]=='--':command=command[1:]
    if command:os.execvp(command[0],command)


if __name__=='__main__':main()
