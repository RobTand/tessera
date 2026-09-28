"""tessera#508: capture a torch-profiler window of steady decode steps.

The serve must be started with the torch profiler configured, e.g.
  --profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/out/prof-ARM
  --profiler-config.ignore_frontend=true --profiler-config.delay_iterations=8
  --profiler-config.max_iterations=4 --profiler-config.torch_profiler_with_stack=false
so the worker skips the prefill and the first decode steps and records four
decode iterations of one 128-token-prompt request.  prof-508.py PORT [BATCH]
"""
import json, sys, threading, time, urllib.request
import os

# The served model name; the full-model graph smoke reuses this probe on its serve.
MODEL = os.environ.get("T508_MODEL", "glm53-stub")

port = sys.argv[1]
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1
URL = f"http://127.0.0.1:{port}"


def post(path, payload=None, timeout=900):
    data = json.dumps(payload).encode() if payload is not None else b""
    req = urllib.request.Request(URL + path, data=data, headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
    return json.loads(body) if body.strip().startswith(b"{") else body.decode()


para = ("The quick brown fox jumps over the lazy dog near the riverbank while "
        "seventeen curious ravens observe the scene from a weathered oak branch. ")
ids = post("/tokenize", dict(model=MODEL, prompt=para * 40, add_special_tokens=False))["tokens"]
post("/v1/completions", dict(model=MODEL, prompt=ids[:32], max_tokens=4, temperature=0))
print("start_profile", post("/start_profile"), flush=True)
t0 = time.time()
threads = [threading.Thread(target=post, args=("/v1/completions", dict(
    model=MODEL, prompt=ids[7 * i:7 * i + 128], max_tokens=48, temperature=0, ignore_eos=True)))
    for i in range(batch)]
for t in threads:
    t.start()
for t in threads:
    t.join()
print("requests done in", round(time.time() - t0, 3), "s", flush=True)
print("stop_profile", post("/stop_profile"), flush=True)
