"""The bank wrapper exposes its mounted checkout to the real command."""
import subprocess
from pathlib import Path


def test_bank_wrapper_exposes_the_mounted_checkout_package(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / 'checkout'
    (checkout / 'experiments').mkdir(parents=True)
    # Image resolution is outside this command-construction control.
    (checkout / 'experiments/runtime_image.sh').write_text('runtime_image_require() { :; }\n')
    capture = tmp_path / 'command'
    boundary = tmp_path / 'capture.sh'
    boundary.write_text('exec() { printf "%s\\0" "$@" > "$COMMAND_CAPTURE"; }\n')
    paths = [tmp_path / name for name in ('old.so', 'new.so', 'bank.json', 'control.json')]
    for path in paths:
        path.touch()
    command = ['bash', str(root / 'experiments/kda/verify_conv_bank.sh'), str(checkout),
               str(tmp_path / 'out'), *map(str, paths), 'bank-sha', 'control-sha', 'module-sha']
    done = subprocess.run(command, env={'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path),
        'BASH_ENV': str(boundary), 'COMMAND_CAPTURE': str(capture), 'ORACLE_IMAGE': 'fixture-image'},
        capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    argv = capture.read_bytes().decode().rstrip('\0').split('\0')
    assert argv[:2] == ['docker', 'run']
    assert argv[argv.index('-v') + 1] == str(checkout) + ':/work:ro'
    assert 'PYTHONPATH=/work/src' in [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == '-e']
    container_script = argv[argv.index('-c') + 1]
    assert 'python3 /work/experiments/kda/verify_conv_bank.py' in container_script
