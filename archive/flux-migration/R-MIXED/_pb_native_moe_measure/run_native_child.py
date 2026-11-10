"""Keep actual loaded package evidence after the native harness has finished."""
import hashlib
import json
from pathlib import Path
import runpy
import sys

evidence = Path(sys.argv[1])
installer_sha256 = sys.argv[2]
sys.argv = ['experiments.bench_native_moe_operator', *sys.argv[3:]]
try:
    runpy.run_module('experiments.bench_native_moe_operator', run_name='__main__')
finally:
    import tessera
    import tessera.cached_unit as cached_unit
    from per_job_install import files

    package = Path(tessera.__file__).resolve().parent
    package_files = files(package)
    installer_bytes = (evidence / 'per-job-runtime.json').read_bytes()
    assert hashlib.sha256(installer_bytes).hexdigest() == installer_sha256, 'Installer evidence changed after native child launch'
    installed = json.loads(installer_bytes)['plugin_files']
    cached_unit.encoder_source_sha256.cache_clear()
    modules, errors = {}, []
    for name, module in sorted(sys.modules.copy().items()):
        if name != 'tessera' and not name.startswith('tessera.'):
            continue
        filename = getattr(module, '__file__', None)
        origin = getattr(getattr(module, '__spec__', None), 'origin', None)
        row = {'file': filename, 'origin': origin}
        try:
            if not filename or not origin:
                raise ValueError('module has no verifiable file and spec origin')
            actual = Path(filename).resolve(strict=True)
            source = Path(origin).resolve(strict=True)
            if actual != source:
                raise ValueError('module file and spec origin differ')
            relative = actual.relative_to(package).as_posix()
            sha = hashlib.sha256(actual.read_bytes()).hexdigest()
            if relative not in installed or installed[relative]['sha256'] != sha:
                raise ValueError('module bytes differ from the canonical installed package')
            row.update(file=str(actual), origin=str(source), sha256=sha)
        except (OSError, ValueError) as exc:
            errors.append({'module': name, 'error': str(exc)})
        modules[name] = row
    record = {
        'schema': 'tessera.loaded_package_identity.v1',
        'observation_scope': 'same native child process, after harness completion and outside measurement',
        'installer_evidence_sha256': installer_sha256,
        'package_path': str(package),
        'cached_unit_path': str(Path(cached_unit.__file__).resolve()),
        'encoder_source_sha256': cached_unit.encoder_source_sha256(),
        'sys_path': sys.path,
        'loaded_tessera_modules': modules,
        'module_identity_errors': errors,
        'package_files': package_files,
        'package_files_unchanged_from_installer': package_files == installed,
    }
    path = evidence / 'post-native-package.json'
    with path.open('x') as stream:
        stream.write(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'artifact': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}), flush=True)
    assert record['package_files_unchanged_from_installer'], 'Native execution changed installed Tessera package files'
    assert not errors, 'Native loaded package origins are incomplete or differ from canonical installed bytes'
