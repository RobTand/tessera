"""The archived scheduler's watcher must not survive in the live census recipe."""
from pathlib import Path
import re
import shutil


def test_census_only_watches_guards_it_can_evaluate():
    text = (Path(__file__).resolve().parents[1] / 'experiments/t16_census/census_stub.sh').read_text()
    assert 'measurement_claims' not in text
    assert 'PrismaBuild measurement row' not in text
    assert 'if [ -e "$WINDOW" ]' in text
    assert 'serve_lock_acquire' in text
    assert 'avail_gib' in text


def test_census_command_substitutions_resolve_to_functions_or_tools():
    root = Path(__file__).resolve().parents[1]
    text = (root / 'experiments/t16_census/census_stub.sh').read_text()
    definitions = set(re.findall(r'^([A-Za-z_]\w*)\s*\(\)\s*\{', text, re.MULTILINE))
    commands = set(re.findall(r'\$\(([A-Za-z_]\w*)\b', text))
    unresolved = {command for command in commands if command not in definitions and shutil.which(command) is None}
    assert not unresolved, f'undefined census command substitutions: {sorted(unresolved)}'
