"""The vendored secret scanner is PrismaQuant's, byte for byte, and every upload runs it first."""
import hashlib
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def test_the_vendored_scanner_is_the_recorded_copy():
    recorded = (ROOT / 'tools/ci/check_no_secret_leak.sha256').read_text().split()[0]
    assert hashlib.sha256((ROOT / 'tools/ci/check_no_secret_leak.py').read_bytes()).hexdigest() == recorded


def test_every_artifact_upload_is_scanned_first():
    """Each upload-artifact step follows a scan of what it uploads, with the secrets to look for."""
    for workflow in sorted((ROOT / '.github/workflows').glob('*.yml')):
        steps = re.split(r'\n      - ', workflow.read_text())
        for index, step in enumerate(steps):
            if 'actions/upload-artifact@' not in step:
                continue
            scan = steps[index - 1]
            assert 'check_no_secret_leak.py' in scan, f'{workflow.name}: an upload is not scanned first'
            assert '--env BUILDBUDDY_API_KEY' in scan and '--env MERGIFY_TOKEN' in scan, workflow.name
