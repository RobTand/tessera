"""The real wrapper argv must request the support size the server allows."""
from pathlib import Path
import os
import re
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("wrapper", ["serve_and_dump_kl.sh", "tessera_plugin_served.sh"])
@pytest.mark.parametrize("topk", [None, 37])
def test_dump_receives_the_wrappers_requested_topk(wrapper, topk):
    body = (ROOT / "experiments" / wrapper).read_text()
    if wrapper == "serve_and_dump_kl.sh":
        assignment = re.search(r"ARGS=\(dump .*?\)\n", body, re.S)
        assert assignment is not None
        command = assignment.group() + '\nprintf "%s\\n" "${ARGS[@]}"'
    else:
        invocation = re.search(r"if ! (\$PY /home/rob/dq-runs/kl_tool.py dump .*?); then", body, re.S)
        assert invocation is not None
        command = 'capture() { printf "%s\\n" "$@"; }; PY=capture; ' + invocation.group(1)
    env = os.environ.copy()
    env.pop("TESSERA_KL_TOPK", None)
    if topk is not None:
        env["TESSERA_KL_TOPK"] = str(topk)
    result = subprocess.run(
        ["bash", "-c", 'OUT=out; DUMP=out; PORT=1; CORPUS=corpus; MODEL=model; ' + command],
        env=env, check=True, capture_output=True, text=True)
    argv = result.stdout.splitlines()
    assert "--top-k" in argv, "server max-logprobs is not the dump request's top-K"
    assert argv[argv.index("--top-k") + 1] == str(1024 if topk is None else topk)
