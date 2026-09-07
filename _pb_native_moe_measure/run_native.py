"""Run the native harness in a fresh process and verify stock files afterward."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from per_job_install import CORE_SHA, digest, files

evidence = Path(sys.argv[1])
arguments = sys.argv[2:]
if arguments[:1] == ['--activation-diagnostic']:
    command = [sys.executable, '/control/native_quant_diagnostic.py', *arguments[1:]]
elif arguments[:1] == ['--kv-capacity-diagnostic']:
    command = [sys.executable, '/control/kv_capacity_diagnostic.py', *arguments[1:]]
else:
    command = [sys.executable, '/control/run_native_child.py', str(evidence),
               digest(evidence / 'per-job-runtime.json'), *arguments]
result = subprocess.run(command)
manifest = Path('/mnt/shared/tessera-clean-runtime-20260907/official-primary/runtime-inventory.json')
assert digest(manifest) == CORE_SHA
stock = json.loads(manifest.read_text())
core = Path(importlib.util.find_spec('vllm').origin).parent
assert files(core) == stock['files'], 'Native execution changed stock vLLM files'
path = evidence / 'post-native-core.json'
path.write_text(json.dumps({'stock_files_unchanged': len(stock['files']),
    'manifest_sha256': CORE_SHA, 'native_returncode': result.returncode}) + '\n')
print(json.dumps({'artifact': str(path), 'sha256': digest(path)}), flush=True)
raise SystemExit(result.returncode)
