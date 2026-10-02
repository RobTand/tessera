"""A fresh extension directory is a valid pre-build measurement state."""
from pathlib import Path
import subprocess


def extension_listing(directory):
    path=Path(__file__).resolve().parents[1]/'experiments/t8r_speed/bench_t8r.sh'
    source=path.read_text()
    start=source.index('ext_libs() {')
    end=source.index('\nEXT_BEFORE=',start)
    command=source[start:end]+'\nEXT_DIR=$1\nresult=$(ext_libs)\nprintf "%s" "$result"\n'
    return subprocess.run(['bash','-euo','pipefail','-c',command,'listing',str(directory)],
                          capture_output=True,text=True,check=True).stdout


def test_fresh_extension_directory_is_valid(tmp_path):
    assert extension_listing(tmp_path)==''


def test_existing_extension_identity_stays_visible(tmp_path):
    directory=tmp_path/'tessera_native';directory.mkdir()
    library=directory/'fixture.so';library.write_bytes(b'not-a-build-probe')
    result=extension_listing(tmp_path)
    timestamp,name=result.strip().split(maxsplit=1)
    assert int(timestamp)==int(library.stat().st_mtime)
    assert name.endswith('tessera_native/fixture.so')
