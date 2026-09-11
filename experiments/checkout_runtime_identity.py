"""Record the runtime identity of an encode job that installs nothing.

The 09-07 launcher used
``/mnt/shared/tessera-clean-runtime-20260907/control-854e672/per_job_install.py``,
which pip-installs Tessera from a frozen source archive and asserts that
archive's digest against a module constant.  The archive is the tree at
``382a1a97``, which predates ``tessera.moe_execution`` entirely, so an exporter
from this branch cannot run under it: the first import fails.  Raising the
constant is not the fix either, because the point of that installer is that the
plugin bytes it installs are the bytes it names.

An encode does not need the plugin installed.  It needs three things to be
true, and this module asserts all three and then execs the command:

* the container is the official image the launcher named, checked the same way
  the frozen installer checks it (declared ID equals the inspected ID, and the
  pinned registry digest appears in its ``RepoDigests``);
* the vLLM core in that image is byte-identical to the attested inventory, and
  stays that way, because nothing is installed into it;
* the Tessera that will be imported is this checkout and only this checkout.
  There is no installed ``tessera-quant`` distribution to shadow it, and the
  code is identified by the same tree hash the cached-unit intake gate reads,
  so the receipt names the encoder that ran.

The vLLM ``general_plugins`` entry point is deliberately absent.  It is what a
serve needs to register ``tessera.serving``; an export reads source weights,
encodes them and writes a checkpoint, and never enters the serving path.  A
serve job is a different job and states its own identity.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

BASE = 'vllm/vllm-openai@sha256:4e31c581716a5cb9ef31eddb0a425842b75cab07d5cd63fb9572e69ae8794c33'
CORE_SHA = 'd4aa04edf388da29679585f4688a2b62d879bcc65349866ba37ed96b7b1e467e'
DISTRIBUTION = 'tessera-quant'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(root):
    root = Path(root)
    return {str(p.relative_to(root)): {'sha256': digest(p), 'bytes': p.stat().st_size}
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir', type=Path, required=True)
    parser.add_argument('--launcher-image-id', required=True)
    parser.add_argument('--launcher-image-inspect', type=Path, required=True)
    parser.add_argument('--checkout', type=Path, default=Path('/control'))
    parser.add_argument('--core-manifest', type=Path,
                        default=Path('/mnt/shared/tessera-clean-runtime-20260907/'
                                     'official-primary/runtime-inventory.json'))
    parser.add_argument('--runtime-uid', type=int, default=1000)
    parser.add_argument('--runtime-gid', type=int, default=1000)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    inspected = json.loads(args.launcher_image_inspect.read_text())
    if isinstance(inspected, list):
        assert len(inspected) == 1
        inspected = inspected[0]
    assert inspected['Id'] == args.launcher_image_id, 'Launcher image ID must match actual image inspection'
    assert BASE in inspected.get('RepoDigests', []), 'Official immutable base must appear in inspected RepoDigests'
    assert digest(args.core_manifest) == CORE_SHA
    assert not (args.evidence_dir / 'per-job-runtime.json').exists(), 'Refuse to overwrite prior runtime evidence'

    stock = json.loads(args.core_manifest.read_text())
    core = Path(importlib.util.find_spec('vllm').origin).parent
    assert files(core) == stock['files'], 'Installed vLLM differs from the attested official image'

    # Both scans below read sys.path, so they run before the checkout joins it.
    # What they mean is 'the image carries no Tessera', and only a pre-insert scan
    # says that: a wheel build leaves src/tessera_quant.egg-info behind, and a
    # snapshot carrying one would otherwise make this module refuse its own job
    # for the opposite of the reason the message gives.
    from importlib import metadata
    try:
        installed = metadata.version(DISTRIBUTION)
    except metadata.PackageNotFoundError:
        installed = None
    assert installed is None, (
        f'{DISTRIBUTION} {installed} is installed in this image. Two Tesseras are then '
        'importable and the receipt cannot say which one encoded. Run this job in the '
        'stock image, where the only Tessera is the checkout.')
    entries = [{'name': e.name, 'value': e.value}
               for e in metadata.entry_points(group='vllm.general_plugins')]
    assert not [e for e in entries if e['name'] == 'tessera'], (
        'a Tessera vLLM plugin entry point is registered. This job encodes and does not '
        'serve; a registered plugin means an installed Tessera this receipt does not name.')

    src = (args.checkout / 'src').resolve()
    assert (src / 'tessera' / '__init__.py').is_file(), f'no Tessera checkout at {src}'
    sys.path.insert(0, str(src))
    import tessera
    from tessera.cached_unit import encoder_source_sha256
    from tessera.encoder_identity import encoder_fixture_id
    assert Path(tessera.__file__).resolve().parent == src / 'tessera'

    record = {
        'launcher_image_inspect_sha256': digest(args.launcher_image_inspect),
        'launcher_image_inspect': inspected,
        'registry_base': BASE,
        'launcher_declared_image_id': args.launcher_image_id,
        'identity_scope': ('The PB Docker launch binds the image reference/ID; this recorder '
                           'verifies the complete vLLM file manifest and installs nothing into '
                           'it. Tessera is imported from the read-only checkout bind, named by '
                           'its encoder tree hash and fixture id.'),
        'core_manifest_sha256': CORE_SHA,
        'core_files_unchanged': len(stock['files']),
        'plugin_install': None,
        'plugin_entrypoints': entries,
        'tessera_checkout': str(src),
        'tessera_version': tessera.__version__,
        'tessera_installed_distribution': installed,
        'encoder_source_sha256': encoder_source_sha256(),
        'encoder_fixture_id': encoder_fixture_id().hex(),
        'vllm_version': metadata.version('vllm'),
        'affinity': sorted(os.sched_getaffinity(0)),
    }
    if os.getuid() == 0:
        os.setgroups([])
        os.setgid(args.runtime_gid)
        os.setuid(args.runtime_uid)
    else:
        assert os.getuid() == args.runtime_uid and os.getgid() == args.runtime_gid
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    path = args.evidence_dir / 'per-job-runtime.json'
    path.write_text(json.dumps(record, indent=2))
    print(json.dumps({'artifact': str(path), 'sha256': digest(path), 'bytes': path.stat().st_size,
                      'core_files_unchanged': record['core_files_unchanged'],
                      'registry_base': BASE, 'tessera_checkout': str(src),
                      'encoder_source_sha256': record['encoder_source_sha256']}), flush=True)
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if command:
        env = os.environ.get('PYTHONPATH')
        os.environ['PYTHONPATH'] = str(src) if not env else f'{src}{os.pathsep}{env}'
        os.execvp(command[0], command)


if __name__ == '__main__':
    main()
