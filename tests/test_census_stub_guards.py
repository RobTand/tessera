"""The archived scheduler's watcher must not survive in the live census recipe."""
from pathlib import Path


def test_census_only_watches_guards_it_can_evaluate():
    text = (Path(__file__).resolve().parents[1] / 'experiments/t16_census/census_stub.sh').read_text()
    assert 'measurement_claims' not in text
    assert 'PrismaBuild measurement row' not in text
    assert 'if [ -e "$WINDOW" ]' in text
    assert 'serve_lock_acquire' in text
    assert 'avail_gib' in text
