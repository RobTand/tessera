"""The served-KL instrument's decode regime, and the refusals around it (#102).

``kl_tool.py`` lives outside this repository -- ``/home/rob/dq-runs/kl_tool.py``
is one shared, unversioned instrument used by PrismaQuant and by every Tessera
serving receipt -- so these tests import it by path.  ``KL_TOOL_DIR`` overrides
that path, which is how the pre-fix failure was recorded: point it at a copy of
the tool from before the change and every test below fails on the missing
``--regime``.

What is pinned here, and why each one is a bug that already happened:

* the prefill request body, byte for byte.  The default mode is the repo's
  authoritative metric; a decode mode that perturbed it would invalidate every
  frozen dump on disk.
* the decode regime's request shape, its warm-up accounting and its regime
  record, round-tripped through a written payload.
* three refusals that make the M=1 claim a measurement rather than an
  assertion: no cached-token accounting, more than one row forwarded, and
  token-TEXT keys instead of token ids.
* the cross-regime compare refusal -- the deliverable of #102.  A prefill dump
  and a decode dump over one corpus look identical on a page and are two
  different metrics.
* a payload with no regime field reads as prefill, so the dumps already frozen
  on disk stay comparable with fresh ones.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
import box_artifacts

KL_TOOL_DIR = box_artifacts.root("kl_instrument")
box_artifacts.require_module("kl_instrument", "kl_tool.py")

# The instrument is untracked and per-box, so the fleet can hold two versions
# of it at once and did: sparklina's copy predated the decode regime while
# sparky's carried it.  It is also not one file -- ``kl_tool`` is a front end
# over ``kl_estimator``, they move together, and a half-copied pair is its own
# failure.  Both halves being stale gave ten ``unrecognized arguments:
# --regime`` failures and one ``KeyError: 'regime'``; copying only the front
# end then gave a collection ``ImportError`` on ``DEFAULT_REGIME``.  Twelve
# symptoms, one fact, and none of them named it.
#
# Deliberately a refusal and not a skip: a box whose instrument cannot tell a
# decode dump from a prefill one still writes receipts under the same metric
# name, so skipping would let it look green while producing them.
_STALE = (
    "the kl instrument under {dir} is stale: {why}.\n"
    "It is untracked and lives outside every checkout, one copy per box, and "
    "it is a set -- kl_tool.py and kl_estimator.py move together, so copying "
    "one of them leaves the pair inconsistent in a new way.  Copy the current "
    "set from a box that has it rather than editing this guard: a served-KL "
    "receipt taken with either version is labelled the same way, which is "
    "what makes the drift dangerous rather than merely inconvenient."
)

if '"--regime"' not in (KL_TOOL_DIR / "kl_tool.py").read_text():
    raise RuntimeError(_STALE.format(
        dir=KL_TOOL_DIR,
        why="kl_tool.py has no --regime flag, so every test below would fail "
            "in argparse rather than in the behaviour it is checking (#102)"))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"_kl102_{name}", KL_TOOL_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    kl_tool = _load("kl_tool")
except ImportError as exc:                                  # a half-copied set
    raise RuntimeError(_STALE.format(
        dir=KL_TOOL_DIR, why=f"kl_tool.py imports a name its own dependency "
                             f"does not provide ({exc})")) from exc


# --------------------------------------------------------------------------
# a corpus contract and a fake serve
# --------------------------------------------------------------------------
N_CHUNKS, SEQLEN, STRIDE = 2, 32, 8
VOCAB = 512


def _contract(tmp_path, *, bos=None):
    chunks = [[(c * SEQLEN + i) % VOCAB for i in range(SEQLEN)]
              for c in range(N_CHUNKS)]
    scored = N_CHUNKS * (SEQLEN - 1) if bos is None else N_CHUNKS * SEQLEN
    contract = {
        "schema": kl_tool.CORPUS_SCHEMA,
        "source_text": str(tmp_path / "corpus.txt"),
        "source_sha256": "a" * 64,
        "source_bytes": 123,
        "offset": 0,
        "n_chunks": N_CHUNKS,
        "seqlen": SEQLEN,
        "tokens": N_CHUNKS * SEQLEN,
        "scored_positions": scored,
        "prepends_bos": bos is not None,
        "bos_token_id": bos,
        "tokenizer": {"schema": "prismaquant.tokenizer_identity/1",
                      "path": "/nonexistent/tok", "files": {"tokenizer.json": "b" * 64},
                      "vocab_size": VOCAB, "bos_token_id": bos,
                      "identity_sha256": "c" * 64},
        "chunks": chunks,
        "contract_sha256": "d" * 64,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract))
    return path


class _Resp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeServe:
    """A vLLM-shaped serve: it records what was asked of it.

    ``cached`` and ``key_style`` are the knobs the refusal tests turn; the
    default is a correct decode-regime serve (one uncached row per scored
    request, token-id keys).
    """

    def __init__(self, *, cached="prefix", key_style="token_id",
                 omit_details=False, top_k=8):
        self.requests = []
        self.cached = cached
        self.key_style = key_style
        self.omit_details = omit_details
        self.top_k = top_k

    def get(self, url, timeout=None):
        return _Resp({"object": "list", "data": [{"id": "kl-target"}]})

    def post(self, url, json=None, timeout=None):  # noqa: A002 -- requests' name
        body = json
        self.requests.append(body)
        prompt = body["prompt"]
        n = len(prompt)
        if "logprobs" not in body and "prompt_logprobs" not in body:
            # the warm-up: a real prefill, nothing scored off it
            return _Resp({"choices": [{"text": "x", "logprobs": None}],
                          "usage": {"prompt_tokens": n, "completion_tokens": 1}})
        if self.cached == "prefix":
            cached = n - 1
        elif self.cached == "none":
            cached = 0
        else:
            cached = int(self.cached)
        entry = {}
        for j in range(self.top_k + 1):
            token = (prompt[-1] + j) % VOCAB
            key = (f"token_id:{token}" if self.key_style == "token_id"
                   else f"<text-{token}>")
            entry[key] = -0.5 - 0.01 * j
        usage = {"prompt_tokens": n, "completion_tokens": 1}
        if not self.omit_details:
            usage["prompt_tokens_details"] = {"cached_tokens": cached}
        return _Resp({"choices": [{"text": "x",
                                   "logprobs": {"tokens": ["token_id:1"],
                                                "token_logprobs": [-0.5],
                                                "top_logprobs": [entry]}}],
                      "usage": usage})


@pytest.fixture
def serve(monkeypatch):
    fake = FakeServe()
    monkeypatch.setitem(sys.modules, "requests", fake)
    return fake


def _dump_argv(tmp_path, contract, out, *extra):
    return ["dump", "--model", "kl-target", "--out", str(out),
            "--url", "http://127.0.0.1:8000/v1/completions",
            "--corpus-contract", str(contract), "--role", "student",
            "--top-k", "8", *extra]


# --------------------------------------------------------------------------
# the default mode does not move
# --------------------------------------------------------------------------
def test_prefill_request_body_is_unchanged():
    body = kl_tool._request_body("vllm", "kl-target", [1, 2, 3], 1024)
    assert body == {"model": "kl-target", "prompt": [1, 2, 3], "max_tokens": 1,
                    "temperature": 0.0, "prompt_logprobs": 1024,
                    "add_special_tokens": False}


def test_prefill_is_the_default_regime(tmp_path, monkeypatch):
    fake = FakeServe()

    def post(url, json=None, timeout=None):  # noqa: A002
        fake.requests.append(json)
        positions = [{} if i == 0 else {str(i): -0.5, str(i + 1): -1.0}
                     for i in range(len(json["prompt"]))]
        return _Resp({"choices": [{"prompt_logprobs": positions}]})

    fake.post = post
    monkeypatch.setitem(sys.modules, "requests", fake)
    contract = _contract(tmp_path)
    out = tmp_path / "prefill"
    assert kl_tool.main(_dump_argv(tmp_path, contract, out)) == 0
    meta = json.loads((tmp_path / "prefill.meta.json").read_text())
    assert meta["regime"]["name"] == "prefill"
    assert all("prompt_logprobs" in r for r in fake.requests)
    assert kl_tool.payload_regime(meta) == "prefill"


# --------------------------------------------------------------------------
# the decode regime
# --------------------------------------------------------------------------
def test_decode_regime_scores_one_row_forwards_and_records_itself(
        tmp_path, serve):
    contract = _contract(tmp_path)
    out = tmp_path / "decode"
    argv = _dump_argv(tmp_path, contract, out,
                      "--regime", "decode", "--decode-stride", str(STRIDE))
    assert kl_tool.main(argv) == 0

    meta = json.loads((tmp_path / "decode.meta.json").read_text())
    regime = meta["regime"]
    expected_per_chunk = len(range(1, SEQLEN, STRIDE))
    assert regime["name"] == "decode"
    assert regime["stride"] == STRIDE
    assert regime["warmup_prefills"] == N_CHUNKS
    assert regime["rows_per_scored_forward"] == 1
    assert regime["scored_positions"] == N_CHUNKS * expected_per_chunk
    assert regime["prefix_lengths"] == [list(range(1, SEQLEN, STRIDE))] * N_CHUNKS
    assert meta["payload"]["positions"] == N_CHUNKS * expected_per_chunk

    warmups = [r for r in serve.requests if "logprobs" not in r]
    scored = [r for r in serve.requests if "logprobs" in r]
    assert len(warmups) == N_CHUNKS
    assert [len(r["prompt"]) for r in warmups] == [SEQLEN] * N_CHUNKS
    assert len(scored) == N_CHUNKS * expected_per_chunk
    assert all(r["max_tokens"] == 1 and r["return_tokens_as_token_ids"]
               and r["temperature"] == 0.0 for r in scored)
    # every scored prefix is block-aligned but for its last token: that is the
    # whole mechanism by which the serve computes exactly one row.
    assert all((len(r["prompt"]) - 1) % STRIDE == 0 for r in scored)

    # and the ids the serve keyed by token_id survive into the payload
    _meta, ids, lps = kl_tool.read_payload(str(out) + ".npz")
    assert ids.shape[0] == N_CHUNKS * expected_per_chunk
    assert int(ids.min()) >= 0
    assert float(lps.max()) <= 0.0


def test_decode_regime_refuses_a_serve_without_cached_token_accounting(
        tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "requests",
                        FakeServe(omit_details=True))
    contract = _contract(tmp_path)
    with pytest.raises(SystemExit) as exc:
        kl_tool.main(_dump_argv(tmp_path, contract, tmp_path / "d",
                                "--regime", "decode",
                                "--decode-stride", str(STRIDE)))
    assert "enable-prompt-tokens-details" in str(exc.value)


def test_decode_regime_refuses_when_more_than_one_row_was_forwarded(
        tmp_path, monkeypatch):
    # a stride that is not the serve's KV block size: the prefix cache hits on
    # whole blocks, so the tail of the last block is recomputed
    monkeypatch.setitem(sys.modules, "requests", FakeServe(cached="none"))
    contract = _contract(tmp_path)
    with pytest.raises(SystemExit) as exc:
        kl_tool.main(_dump_argv(tmp_path, contract, tmp_path / "d",
                                "--regime", "decode",
                                "--decode-stride", str(STRIDE)))
    message = str(exc.value)
    assert "rows, not 1" in message
    assert "KV block size" in message


def test_decode_regime_refuses_token_text_keys(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "requests", FakeServe(key_style="text"))
    contract = _contract(tmp_path)
    with pytest.raises(SystemExit) as exc:
        kl_tool.main(_dump_argv(tmp_path, contract, tmp_path / "d",
                                "--regime", "decode",
                                "--decode-stride", str(STRIDE)))
    assert "token TEXT" in str(exc.value)


def test_decode_regime_is_vllm_only(tmp_path, serve):
    contract = _contract(tmp_path)
    with pytest.raises(SystemExit) as exc:
        kl_tool.main(_dump_argv(tmp_path, contract, tmp_path / "d",
                                "--regime", "decode", "--runtime", "sglang"))
    assert "vLLM surface only" in str(exc.value)


# --------------------------------------------------------------------------
# compare: the cross-regime refusal
# --------------------------------------------------------------------------
def _payload(tmp_path, name, *, regime, role="student", label=None,
             positions=6, seed=0.0):
    import numpy as np

    ids = np.arange(positions * 4, dtype=np.int32).reshape(positions, 4) % 97
    lps = (np.full((positions, 4), -1.0, dtype=np.float32)
           - seed - 0.1 * np.arange(4, dtype=np.float32))
    meta = {
        "schema": kl_tool.DUMP_SCHEMA,
        "role": role,
        "teacher_label": label,
        "metric": {"requested_top_k": 4, "full_vocab": False},
        "corpus": {"source_sha256": "a" * 64, "tokens": 64,
                   "contract_sha256": "d" * 64},
        "tokenizer": {"identity_sha256": "c" * 64, "path": "/nonexistent/tok",
                      "files": {"tokenizer.json": "b" * 64}},
    }
    if regime is not None:
        meta["regime"] = {"name": regime}
    out = tmp_path / name
    kl_tool.write_payload(out, meta, ids, lps)
    return str(out) + ".npz"


def test_compare_refuses_a_cross_regime_pair(tmp_path):
    teacher = _payload(tmp_path, "t", regime="prefill", role="teacher",
                       label="BF16")
    student = _payload(tmp_path, "s", regime="decode", seed=0.2)
    with pytest.raises(SystemExit) as exc:
        kl_tool.main(["compare", teacher, student])
    message = str(exc.value)
    assert "cross-regime" in message
    assert "regime=prefill" in message and "regime=decode" in message
    # and it is NOT reachable by the alignment escape hatch
    with pytest.raises(SystemExit) as exc2:
        kl_tool.main(["compare", teacher, student, "--allow-mismatch"])
    assert "cross-regime" in str(exc2.value)


def test_compare_within_one_regime_reports_the_regime(tmp_path, capsys):
    teacher = _payload(tmp_path, "t", regime="decode", role="teacher",
                       label="BF16")
    student = _payload(tmp_path, "s", regime="decode", seed=0.2)
    out = tmp_path / "cmp.json"
    assert kl_tool.main(["compare", teacher, student, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "regime=decode" in printed
    result = json.loads(out.read_text())
    assert result["regime"] == "decode"
    assert result["metric_identity"]["regime"] == "decode"


def test_a_payload_with_no_regime_field_is_a_prefill_payload(tmp_path):
    """Every dump frozen on disk before 2026-09-03 must stay comparable."""
    legacy = _payload(tmp_path, "legacy", regime=None, role="teacher",
                      label="BF16")
    fresh = _payload(tmp_path, "fresh", regime="prefill", seed=0.2)
    assert kl_tool.main(["compare", legacy, fresh]) == 0
    decode = _payload(tmp_path, "decode", regime="decode", seed=0.2)
    with pytest.raises(SystemExit):
        kl_tool.main(["compare", legacy, decode])


# --------------------------------------------------------------------------
# fingerprint: the identity check behind "the prefill mode did not move"
# --------------------------------------------------------------------------
def test_fingerprint_ignores_the_wall_clock_and_pins_the_regime(tmp_path):
    import numpy as np

    ids = np.arange(24, dtype=np.int32).reshape(6, 4)
    lps = np.full((6, 4), -1.5, dtype=np.float32)
    base = {
        "schema": kl_tool.DUMP_SCHEMA, "role": "student", "teacher_label": None,
        "metric": {"requested_top_k": 4},
        "corpus": {"source_sha256": "a" * 64, "contract_sha256": "d" * 64},
        "tokenizer": {"identity_sha256": "c" * 64},
    }
    first = dict(base, produced_at_utc="2026-09-03T00:00:00Z", elapsed_s=1.0,
                 argv=["a"], host="sparky")
    second = dict(base, regime={"name": "prefill"},
                  produced_at_utc="2026-09-03T09:99:99Z", elapsed_s=2.5,
                  argv=["b"], host="sparklina")
    kl_tool.write_payload(tmp_path / "one", first, ids, lps)
    kl_tool.write_payload(tmp_path / "two", second, ids, lps)
    fp1 = kl_tool.payload_fingerprint(tmp_path / "one.npz")
    fp2 = kl_tool.payload_fingerprint(tmp_path / "two.npz")
    assert fp1["fingerprint"] == fp2["fingerprint"]
    assert fp1["regime"] == "prefill"

    third = dict(base, regime={"name": "decode"})
    kl_tool.write_payload(tmp_path / "three", third, ids, lps)
    fp3 = kl_tool.payload_fingerprint(tmp_path / "three.npz")
    assert fp3["fingerprint"] != fp1["fingerprint"]

    # and one changed logprob changes it
    lps2 = lps.copy()
    lps2[0, 0] = -1.4
    kl_tool.write_payload(tmp_path / "four", second, ids, lps2)
    assert (kl_tool.payload_fingerprint(tmp_path / "four.npz")["fingerprint"]
            != fp2["fingerprint"])
