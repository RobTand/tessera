"""The census launchers refuse an unqualified producer before any skip.

tessera#944: the census export wrappers ignored the requested producer
interpreter -- the t8 routed launcher never read ``TESSERA_PRODUCER_PYTHON``
and ran container python under a source-shadowing PYTHONPATH; the t16 dense
wrapper hard-coded a venv python the same way.  The corrected contract under
test here:

* both ``TESSERA_PRODUCER_PYTHON`` and an EXPLICIT, absolute
  ``TESSERA_PRODUCER_SOURCE`` are required by name -- the source reference is
  never derived from ``$PWD`` (a PrismaBuild snapshot is intentionally
  parentless and cannot qualify) and never overwritten;
* authentication (the real ``tessera.export_serving.authenticate_producer_python``,
  whose byte/lineage contract tests/test_t8_batch_probe.py pins) happens
  BEFORE a done marker can short-circuit anything;
* the done marker binds the CONTENT digests of plan/Hessian/authority/input
  scales, the FULL producer receipt and the output manifest's sha256, and the
  skip path re-verifies the manifest under the serving_parts owners
  (schema, membership, output seal, sealed encode_batch, producer receipt) --
  a same-path mutation of a bound file is caught by its digest;
* the selected interpreter and source travel, unchanged, into the actual
  exporter process's environment.

The producer interpreter here is a stub that answers the launcher's
authentication call with a fixed receipt, so the launcher's refusal order and
marker logic are what runs. Both stamp and source-aware serving_parts owner
checks execute for real; only the authentication/export legs are stubbed.
The real producer API is exercised separately through the launcher. These
CPU control-flow tests make no GPU, container or native-byte claim.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[1]
T8_LAUNCHER = CHECKOUT / "experiments/t8_census/export_routed_part.sh"
T16_LAUNCHER = CHECKOUT / "experiments/t16_census/export_census_stub.sh"
STUB = "T8R2"  # plan-T8R2.json ships in the checkout, so no repo writes
IMG = "localhost/prismaquant/spark-vllm-nccl230@sha256:" + "ab" * 32
BOUND_CAP = 2700
STUB_SHA = "b770" * 16
STUB_HEAD = "5" * 40
HESSIAN = None
PLAN = CHECKOUT / "experiments/t8_census/plan-T8R2.json"

#: The receipt the stub producer answers with; source_root comes from the
#: TESSERA_PRODUCER_SOURCE the launcher passes through, so receipt equality
#: across a re-run proves the reference travelled unchanged.
STUB_RECEIPT_CODE = '''import json, os
print(json.dumps({
    "schema": "tessera.producer_python.v1",
    "selection": "TESSERA_PRODUCER_PYTHON",
    "requested_interpreter": "stub-producer",
    "interpreter": "stub-producer",
    "executable_sha256": "%s",
    "sys_prefix": "stub",
    "python_version": "stub",
    "torch_version": "stub",
    "source": "TESSERA_PRODUCER_SOURCE",
    "source_root": os.environ.get("TESSERA_PRODUCER_SOURCE", "<unset>"),
    "git_head": "%s",
    "descends_from": "b770727c50eef822132518bdc4fd6efe84359c9e",
    "installed_package": "stub",
    "expected_package_sha256": "%s",
    "package_sha256": "%s",
    "shipped_files": 0,
    "runtime_contract_sha256": "%s",
}))''' % ("e" * 64, STUB_HEAD, STUB_SHA, STUB_SHA, "0" * 64)

AUTH_SENTINEL = "authenticate_producer_python"
OWNER_SENTINEL = "TESSERA_PARTITION_MEMBERSHIP_OWNER_STEP"

_STUB_COMMON = (
    "import json, os, sys\n"
    "args = sys.argv[1:]\n"
    "if args[:2] == ['-m', 'tessera.export_serving']:\n"
    "    dump = os.environ.get('TESSERA_ENV_DUMP')\n"
    "    if dump:\n"
    "        open(dump, 'w').write(json.dumps(dict(os.environ)))\n"
    "    raise SystemExit(0)\n"
    "if args[:1] == ['-c']:\n"
    "    code = args[1]\n"
)
_STUB_RECEIPT_INDENTED = "".join(
    "        " + line + "\n" for line in STUB_RECEIPT_CODE.splitlines())
STUB_SCRIPT = (
    _STUB_COMMON
    + f"    if {AUTH_SENTINEL!r} in code:\n"
    + _STUB_RECEIPT_INDENTED
    + "        raise SystemExit(0)\n"
    + "    sys.argv = ['stub-producer'] + args[2:]\n"
    + "    exec(compile(code, '<launcher-verification>', 'exec'))\n"
    + "    raise SystemExit(0)\n"
    + "raise SystemExit(2)\n"
)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@pytest.fixture(scope="session", autouse=True)
def local_hessian_capture(tmp_path_factory):
    global HESSIAN
    HESSIAN = tmp_path_factory.mktemp("capture") / "hessian_capture.references.json"
    HESSIAN.write_text('{"schema":"launcher-input-fixture"}\n')



@pytest.fixture(scope="session")
def producer_python(tmp_path_factory):
    stub = tmp_path_factory.mktemp("producer") / "stub-producer.py"
    stub.write_text(f"#!{sys.executable}\n" + STUB_SCRIPT)
    stub.chmod(0o755)
    return stub


@pytest.fixture(scope="session")
def qualified_source(tmp_path_factory):
    """An explicit qualified source reference, NOT the checkout under test."""
    source = tmp_path_factory.mktemp("genuine") / "qualified-checkout" / "src" / "tessera"
    source.mkdir(parents=True)
    return source


def _receipt(producer_python, source):
    """The exact receipt the stub answers with for this source reference."""
    env = dict(os.environ, TESSERA_PRODUCER_SOURCE=str(source))
    done = subprocess.run([str(producer_python), "-c", STUB_RECEIPT_CODE],
                          env=env, capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def _launch_env(tmp_path, **over):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("TESSERA_PRODUCER")
           and k not in ("INPUT_SCALES", "PART_BOUND_S", "ENCODE_BATCH",
                         "BEST_FORM", "PROFILE", "PRODUCER_AUTHORITY",
                         "PART_IMAGE", "CENSUS_ROOT", "TESSERA_ENV_DUMP")}
    env["PRODUCER_AUTHORITY"] = str(tmp_path / "producer_authority.py")
    (tmp_path / "producer_authority.py").write_text("authority-placeholder\n")
    env["PART_IMAGE"] = IMG
    env["CENSUS_ROOT"] = str(tmp_path / "census")
    env["HESSIAN_CAPTURE"] = str(HESSIAN)
    env.update({k: str(v) for k, v in over.items()})
    return env


def _full_stamp(tmp_path, producer, source, input_scales=None,
                input_scales_bytes=None):
    authority = tmp_path / "producer_authority.py"
    # The authority file the launcher will be handed; same bytes _launch_env
    # writes, created here so the stamp's digest is over a real file.
    authority.write_text("authority-placeholder\n")
    scales = None
    if input_scales is not None:
        Path(input_scales).write_text(input_scales_bytes)
        scales = str(input_scales)
    stamp = {
        "stub": STUB, "partition": "0/8",
        "producer_python": str(producer),
        "producer_source": str(source),
        "image": IMG, "encode_batch": 1, "window_best_form": None,
        "input_scales": scales,
        "content": {
            "plan_sha256": sha_file(PLAN),
            "hessian_sha256": sha_file(HESSIAN),
            "authority_sha256": sha_file(authority),
            "input_scales_sha256": sha_file(input_scales) if scales else None,
        },
        "producer_receipt": _receipt(producer, source),
    }
    return stamp


def _manifest_bytes(authority, scales=None, plan_entries=None,
                    input_scales_seal=None, hessian_seal=None, index=0, count=8):
    """Intentionally incomplete stamp-check fixture, never a skip witness."""
    options = {
        "plan": plan_entries if plan_entries is not None else json.loads(PLAN.read_text()),
        "hessian_sha256": hessian_seal if hessian_seal is not None else sha_file(HESSIAN),
        "producer_authority_sha256": sha_file(authority),
        "input_scales_sha256": input_scales_seal,
    }
    return (json.dumps({
        "schema": "tessera.serving-part.v1",
        "encode_batch": 1,
        "export_partition": {
            "schema": "tessera.serving-part.v1",
            "index": index, "count": count,
            "identity": {"options": options},
            "output_sha256": {},
        },
    }, indent=1) + "\n").encode()


def _write_marker(tmp_path, stamp, manifest_bytes=None):
    parts = Path(tmp_path, "census", "stubs", f"parts-{STUB}")
    parts.mkdir(parents=True, exist_ok=True)
    part_out = parts / "part-0"
    part_out.mkdir(exist_ok=True)
    manifest = part_out / "tessera_serving_manifest.json"
    manifest.write_bytes(manifest_bytes if manifest_bytes is not None else b"{}\n")
    stamp = dict(stamp)
    stamp.setdefault("output", {"manifest_sha256": sha_file(manifest)})
    mark = parts / "part-0.done.json"
    mark.write_text(json.dumps(stamp, indent=1) + "\n")
    return mark, manifest


def _run_t8(tmp_path, env):
    return subprocess.run(
        ["bash", str(T8_LAUNCHER), STUB, "0", "8"],
        env=env, cwd=CHECKOUT, capture_output=True, text=True, timeout=900)


def _run_t16(tmp_path, env):
    return subprocess.run(
        ["bash", str(T16_LAUNCHER), "D1"],
        env=env, cwd=CHECKOUT, capture_output=True, text=True, timeout=900)


# --- required-by-name selector and source reference -------------------------


def test_t8_refuses_missing_producer_before_done_marker(tmp_path, qualified_source):
    """A present done marker must not short-circuit a missing selector (#944)."""
    parts = Path(tmp_path, "census", "stubs", f"parts-{STUB}")
    parts.mkdir(parents=True)
    (parts / "part-0.done.json").write_text('{"stub": "%s", "partition": "0/8"}\n' % STUB)
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_SOURCE=qualified_source))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text
    assert "already done" not in text


def test_t8_refuses_missing_source_reference(tmp_path, producer_python):
    """The source reference is required by name; $PWD is never a fallback."""
    mark, _ = _write_marker(tmp_path, _full_stamp(tmp_path, producer_python, Path("/genuine/src/tessera")))
    before = mark.read_bytes()
    proc = _run_t8(tmp_path, _launch_env(tmp_path, TESSERA_PRODUCER_PYTHON=producer_python))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_SOURCE" in text
    assert "already done" not in text
    assert mark.read_bytes() == before, "refusal must not touch the old marker"


def test_t8_refuses_relative_source_reference(tmp_path, producer_python):
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python, TESSERA_PRODUCER_SOURCE="src/tessera"))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_SOURCE" in text
    assert "absolute" in text


def test_t8_refuses_nonexecutable_producer(tmp_path, qualified_source):
    absent = tmp_path / "absent" / "bin" / "python"
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=absent, TESSERA_PRODUCER_SOURCE=qualified_source))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text
    assert str(absent) in text


def test_t8_refuses_real_unqualified_interpreter(tmp_path):
    """The running pytest interpreter is not a qualified producer install.

    Through the launcher, the real API must refuse it -- whichever leg fires
    (no installed distribution behind the import, dirty reference, ancestry) --
    and the refusal must name the selector, never proceed to an export.
    """
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=sys.executable,
        TESSERA_PRODUCER_SOURCE=CHECKOUT / "src" / "tessera"))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text


# --- bounds and machine settings --------------------------------------------


def test_t8_refuses_part_bound_over_cap(tmp_path, producer_python, qualified_source):
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
        TESSERA_PRODUCER_SOURCE=qualified_source, PART_BOUND_S=BOUND_CAP + 1))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "PART_BOUND_S" in text
    assert str(BOUND_CAP) in text


def test_t8_refuses_part_bound_garbage(tmp_path, producer_python, qualified_source):
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
        TESSERA_PRODUCER_SOURCE=qualified_source, PART_BOUND_S="45min"))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "PART_BOUND_S" in text


def test_t8_refuses_encode_batch_garbage(tmp_path, producer_python, qualified_source):
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
        TESSERA_PRODUCER_SOURCE=qualified_source, ENCODE_BATCH="many"))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "ENCODE_BATCH" in text


# --- the done marker binds content, receipt and sealed output ---------------


def test_t8_refuses_same_path_scale_file_mutation(tmp_path, producer_python, qualified_source):
    """A bound file mutated in place is caught by its digest, not its name."""
    scales = tmp_path / "input_scales.safetensors"
    stamp = _full_stamp(tmp_path, producer_python, qualified_source,
                        input_scales=scales, input_scales_bytes="scale bytes v1\n")
    mark, _ = _write_marker(tmp_path, stamp, _manifest_bytes(
        tmp_path / "producer_authority.py", input_scales_seal=sha_file(scales)))
    before = mark.read_bytes()
    part_out = mark.parent / "part-0"
    Path(scales).write_text("scale bytes v2\n")
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
        TESSERA_PRODUCER_SOURCE=qualified_source, INPUT_SCALES=scales))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "input_scales_sha256" in text, "refusal must name the changed content digest"
    assert "CENSUS_ROOT" in text, "refusal must direct the operator to a fresh root"
    assert mark.read_bytes() == before, "refusal must not touch the old marker"
    assert (part_out / "tessera_serving_manifest.json").exists(), \
        "refusal must not touch the old part directory"


def test_t8_refuses_mutation_behind_restamped_marker(tmp_path, producer_python, qualified_source):
    """A file mutated after the exporter's snapshot cannot ride a new stamp.

    The exporter sealed the bytes it consumed into the manifest's identity
    options.  A marker re-stamped to claim the mutated bytes must still be
    refused: the manifest's consumed seal is the original, and current,
    marker and manifest must agree three ways.
    """
    scales = tmp_path / "input_scales.safetensors"
    open(scales, "w").write("scale bytes v1\n")
    v1_digest = sha_file(scales)
    stamp = _full_stamp(tmp_path, producer_python, qualified_source,
                        input_scales=scales, input_scales_bytes="scale bytes v2\n")
    mark, _ = _write_marker(tmp_path, stamp, _manifest_bytes(
        tmp_path / "producer_authority.py", input_scales_seal=v1_digest))
    before = mark.read_bytes()
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
        TESSERA_PRODUCER_SOURCE=qualified_source, INPUT_SCALES=scales))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "consumed.input_scales_sha256" in text, \
        "refusal must name the manifest's consumed seal, not just the marker claim"
    assert mark.read_bytes() == before


def test_t8_refuses_unvalidated_historical_marker(tmp_path, producer_python, qualified_source):
    """Old-schema markers are conservatively refused, never silently reused."""
    parts = Path(tmp_path, "census", "stubs", f"parts-{STUB}")
    parts.mkdir(parents=True)
    mark = parts / "part-0.done.json"
    mark.write_text(json.dumps({"stub": STUB, "partition": "0/8", "image": IMG}, indent=1) + "\n")
    before = mark.read_bytes()
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
        TESSERA_PRODUCER_SOURCE=qualified_source))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "content" in text and "unvalidated historical marker" in text
    assert mark.read_bytes() == before


def test_t8_honors_exact_done_marker(tmp_path, producer_python, qualified_source):
    """The skip must have passed the actual authentication and stamp checks.

    Asserting the verified-skip wording (not merely a zero exit) is what
    makes this test fail on the old wrapper, which skipped ANY marker with
    no producer, source or content validation at all.
    """
    from test_b770_producer_review import bundle
    source, _out, stamp, manifest = bundle(tmp_path, producer_python, qualified_source)
    _write_marker(tmp_path, stamp, json.dumps(manifest).encode())
    proc = _run_t8(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
        TESSERA_PRODUCER_SOURCE=qualified_source, SOURCE_CHECKPOINT=source,
        PYTHONPATH=CHECKOUT / "src"))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"exact marker must skip cleanly, got rc={proc.returncode}\n{text}"
    assert "already done and verified" in text


# --- the probe caller (profile_unit_encode.sh, lead-migrated) ----------------


def test_profile_unit_encode_refuses_missing_producer(tmp_path):
    """The probe caller honours the same selector contract as the exporters.

    A missing TESSERA_PRODUCER_PYTHON must refuse by name before anything
    runs -- not die on the old PART_IMAGE/container failure the pre-migration
    script hit with this exact invocation.
    """
    out = tmp_path / "prof-out"
    proc = subprocess.run(
        ["bash", str(CHECKOUT / "experiments/t8_census/profile_unit_encode.sh"), str(out)],
        env=_launch_env(tmp_path), cwd=CHECKOUT, capture_output=True, text=True, timeout=600)
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text


# --- the t16 dense (D1) wrapper ---------------------------------------------


def test_t16_refuses_missing_producer(tmp_path, qualified_source):
    proc = _run_t16(tmp_path, _launch_env(tmp_path, TESSERA_PRODUCER_SOURCE=qualified_source))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text


def test_t16_refuses_missing_source_reference(tmp_path, producer_python):
    proc = _run_t16(tmp_path, _launch_env(tmp_path, TESSERA_PRODUCER_PYTHON=producer_python))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_SOURCE" in text


def test_t16_refuses_nonexecutable_producer(tmp_path, qualified_source):
    absent = tmp_path / "absent" / "bin" / "python"
    proc = _run_t16(tmp_path, _launch_env(
        tmp_path, TESSERA_PRODUCER_PYTHON=absent, TESSERA_PRODUCER_SOURCE=qualified_source))
    text = proc.stdout + proc.stderr
    assert proc.returncode == 2, f"expected refusal, got rc={proc.returncode}\n{text}"
    assert "TESSERA_PRODUCER_PYTHON" in text
    assert str(absent) in text


def test_t16_forwards_selected_env_to_the_actual_exporter(tmp_path, producer_python, qualified_source):
    """The exporter process itself sees the SAME interpreter and source ref.

    The source must arrive as the caller's explicit reference -- never
    rewritten to $PWD/src/tessera -- and the scratch TMPDIR must be the
    census root's, not the inherited environment's.
    """
    dump = tmp_path / "exporter-env.json"
    env = _launch_env(tmp_path, TESSERA_PRODUCER_PYTHON=producer_python,
                      TESSERA_PRODUCER_SOURCE=qualified_source, TESSERA_ENV_DUMP=dump)
    env["TMPDIR"] = str(tmp_path / "inherited-tmpdir")
    proc = _run_t16(tmp_path, env)
    text = proc.stdout + proc.stderr
    assert proc.returncode == 0, f"stub exporter must run, got rc={proc.returncode}\n{text}"
    seen = json.loads(dump.read_text())
    assert seen["TESSERA_PRODUCER_PYTHON"] == str(producer_python)
    assert seen["TESSERA_PRODUCER_SOURCE"] == str(qualified_source)
    assert seen["TESSERA_PRODUCER_SOURCE"] != str(CHECKOUT / "src" / "tessera")
    assert seen["TMPDIR"] == str(Path(tmp_path, "census", "stubs", "tmp"))
