"""One admitted original-wire whole-owner run in the immutable official base."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import uuid

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evidence-dir', type=Path, required=True)
parser.add_argument('--cpu-only', action='store_true')
parser.add_argument('--serving-config', type=Path, default=Path('experiments/configs/lfm25_first_model_fixed_kv_20260907.json'))
parser.add_argument('arguments', nargs=argparse.REMAINDER)
args = parser.parse_args()
arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
root = args.evidence_dir
root.mkdir(exist_ok=False, parents=True)
cache = Path('/mnt/shared/tessera-native376-resource/native-moe-original-r1024/triton-cache-1970-382')
seal_path = cache.with_name(cache.name + '.seal.json')
cache.mkdir(parents=True, exist_ok=True)

def cache_files():
    return {str(path.relative_to(cache)): {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'bytes': path.stat().st_size} for path in sorted(cache.rglob('*')) if path.is_file()}

seal = json.loads(seal_path.read_text()) if seal_path.exists() else None
if seal is None:
    assert '--prepare' in arguments, 'Only untimed preparation may establish a new Triton cache seal'
else:
    assert cache_files() == seal['files'], 'Existing Triton cache bytes differ from their seal'
control = Path.cwd() / '_pb_native_moe_measure'
assert hashlib.sha256((control / 'per_job_install.py').read_bytes()).hexdigest() == '7b981538b0b56ebac7b5c080175246ae3338d34167356e70c7faf60b3df63fa7'
configuration_sha256 = hashlib.sha256(args.serving_config.read_bytes()).hexdigest()
base = json.loads(args.serving_config.read_text())['runtime_image']
if '--request' in arguments:
    request_path = Path(arguments[arguments.index('--request') + 1])
    request = json.loads(request_path.read_text())
    requested_config = Path(request['serving_config_path'])
    if not requested_config.is_absolute():
        requested_config = request_path.parent / requested_config
    assert hashlib.sha256(requested_config.read_bytes()).hexdigest() == configuration_sha256, 'Launcher and native request select different serving configurations'
    assert request['runtime_image'] == base
subprocess.run(['docker', 'pull', base], check=True)
inspected = json.loads(subprocess.check_output(['docker', 'image', 'inspect', base]))[0]
assert base in inspected.get('RepoDigests', [])
image_id = inspected['Id']
# The coordinator's explicit research config selects this immutable base.
# Reuse the runtime resolver for the observed Docker identity and retain the
# distinction from Tessera's packaged production default in the declaration.
spec = importlib.util.spec_from_file_location('tessera_runtime_image_launcher',
    Path.cwd() / 'src/tessera/serving/runtime_image.py')
runtime_image = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime_image)
contract = json.loads(Path('src/tessera/serving/runtime_contract.json').read_text())
packaged_default = runtime_image.pinned_reference(contract)
contract['versions']['default_serve_image'] = base
declaration = runtime_image.require_pinned(base, contract=contract,
    inspector=lambda reference: {'present': True, 'local_id': image_id,
        'repo_digests': sorted(inspected['RepoDigests']), 'error': None})
declaration['selection'] = {
    'scope': 'explicit_research_configuration_not_packaged_default_or_cell_promotion',
    'configuration_sha256': configuration_sha256,
    'packaged_default_reference': packaged_default}
image_environment = runtime_image.container_env(declaration)
assert image_environment and declaration['resolved_reference'] == base
image_path = root / 'launcher-image-inspect.json'
image_path.write_text(json.dumps(inspected, indent=2))
(root / 'runtime-image-declaration.json').write_text(json.dumps(declaration, indent=2))
cidfile = root / 'container.cid'
command = ['docker', 'run', *([] if args.cpu_only else ['--gpus', 'all']), '--ipc', 'host', '--cidfile', str(cidfile),
 '--name', 'tessera-moe-native-' + uuid.uuid4().hex, '--network', 'none',
 '--volume', f'{Path.cwd()}:/work:ro', '--volume', f'{control}:/control:ro',
 '--volume', '/mnt/shared:/mnt/shared', '--workdir', '/work',
 '--env', 'OMP_NUM_THREADS', '--env', 'MKL_NUM_THREADS', '--env', 'OPENBLAS_NUM_THREADS',
 '--env', 'PYTHONPATH=/work', '--env', 'TESSERA_SERVE_MODE=resident',
 '--env', 'TRITON_CACHE_DIR=' + str(cache),
 '--env', 'FLASHINFER_WORKSPACE_BASE=/tmp/tessera-flashinfer',
 '--env', 'XDG_CACHE_HOME=/tmp/tessera-cache', '--env', 'TORCH_EXTENSIONS_DIR=/tmp/tessera-extensions',
 *[item for key, value in image_environment.items() for item in ('--env', key + '=' + value)],
 '--entrypoint', 'python3', image_id, '/control/per_job_install.py',
 '--evidence-dir', str(root), '--launcher-image-id', image_id,
 '--launcher-image-inspect', str(image_path), '--',
 'python3', '/control/run_native.py', str(root), *arguments]
(root / 'launch-command.json').write_text(json.dumps(command, indent=2))
result = subprocess.run(command)
cid = cidfile.read_text().strip()
container = json.loads(subprocess.check_output(['docker', 'inspect', cid]))[0]
assert container['Id'] == cid and container['Image'] == image_id
assert container['Config']['Labels']['prismabuild.action'] == os.environ['PRISMABUILD_CONTAINER_OWNER']
assert not container['State']['Running']
(root / 'container-inspect.json').write_text(json.dumps(container, indent=2))
subprocess.run(['docker', 'rm', cid], check=True)
after_cache = cache_files()
cache_unchanged = seal is None or after_cache == seal['files']
cache_record = {'cache': str(cache), 'seal_path': str(seal_path),
    'mode': 'establish_after_untimed_preparation' if seal is None else 'verify_existing_seal',
    'unchanged': cache_unchanged, 'files': after_cache}
if seal is None and result.returncode == 0 and container['State']['ExitCode'] == 0:
    assert any(name.endswith('.so') for name in after_cache), 'No generated native host binary'
    assert any(name.endswith('.cubin') for name in after_cache), 'No generated CUDA kernel binary'
    seal = {'schema': 'tessera.native_triton_cache_seal.v1', 'cache': str(cache),
        'runtime_image': base, 'files': after_cache}
    with seal_path.open('x') as stream:
        stream.write(json.dumps(seal, sort_keys=True, indent=2) + '\n')
if seal_path.exists():
    cache_record['seal_sha256'] = hashlib.sha256(seal_path.read_bytes()).hexdigest()
(root / 'triton-cache-verification.json').write_text(json.dumps(cache_record, indent=2) + '\n')
for path in sorted(root.iterdir()):
 if path.is_file():
  print(json.dumps({'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                   'bytes': path.stat().st_size}), flush=True)
assert cache_unchanged, 'Native execution changed or added sealed Triton cache artifacts'
raise SystemExit(result.returncode or container['State']['ExitCode'])
