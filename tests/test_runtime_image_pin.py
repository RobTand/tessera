"""The serve-image pin is a digest, and a mismatch refuses (issue #100).

NOTE ON WHAT IS *NOT* HERE: the digest itself.  The pin lives in exactly one
place -- ``runtime_contract.json``'s ``versions.attested_on.image`` -- and a
test that repeated the 64-hex string would be the second copy the whole design
exists to avoid: it would pass while the pin was wrong, and it would have to be
edited every time the runtime is re-attested.  These tests pin the RULE (the
shape of the value, the membership check, the refusal) and read the value.

No test here shells out to ``docker``.  The daemon's answer is injected, which
is what lets the cross-box case -- identical bytes, different local ``.Id`` --
be tested at all: there is no second box inside a test run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving.contract import contract_path
from tessera.serving.runtime_image import (  # noqa: E402
    PIN_CONTRACT_FIELD,
    RuntimeImageError,
    parse_reference,
    pinned_reference,
    require_pinned,
    resolve,
)

#: ``repository@sha256:<64 hex>``.  A tag reference cannot match, which is the
#: property under test.
DIGEST_REFERENCE = re.compile(r"^[a-z0-9][a-z0-9._/-]*[a-z0-9]@sha256:[0-9a-f]{64}$")

CONTRACT = json.loads(contract_path().read_text(encoding="utf-8"))


def _inspector(*, repo_digests, local_id="sha256:" + "ab" * 32, present=True):
    """A stand-in for the local docker daemon."""
    def inspect(_reference):
        return {"present": present, "local_id": local_id,
                "repo_digests": list(repo_digests)}
    return inspect


# ------------------------------------------------------- the pin is a pin ---

def test_the_pin_is_a_digest_reference_not_a_tag():
    pin = pinned_reference()
    assert DIGEST_REFERENCE.match(pin), (
        f"the serve-image pin is {pin!r}. A tag is a name upstream can "
        "repoint, so a receipt naming one has recorded nothing about what ran.")


def test_the_pin_is_the_contract_field_and_nothing_else_holds_it():
    node = CONTRACT
    for key in PIN_CONTRACT_FIELD:
        node = node[key]
    assert node == pinned_reference()

    digest = parse_reference(node)[2]
    assert digest is not None
    # Every tracked file that ACTS must read the pin, never repeat it: a second
    # copy of a 64-hex string is a second thing to forget to update.
    #
    # RECEIPTS ARE EXEMPT, and the exemption is the point rather than a hole in
    # it.  A measurement doc or a committed .build.json sidecar recording the
    # digest its serve ran under is not a copy of the pin -- it is the thing
    # the pin exists to make possible, and a test that failed on one would
    # forbid the receipts this issue is about.  So the scan covers the code
    # that decides (src/, tests/, tools/, and the wrapper scripts) and skips
    # docs/ and experiments/results/, which only record.
    acting_suffixes = {".py", ".sh", ".json", ".toml", ".cfg", ".ini",
                       ".yaml", ".yml"}
    acting_roots = ("src/", "tests/", "tools/", "experiments/")
    exempt_prefixes = ("experiments/results/",)
    listed = subprocess.run(["git", "-C", str(ROOT), "ls-files"],
                            capture_output=True, text=True, check=True)
    hex_body = digest.split(":", 1)[1]
    holders = []
    for name in listed.stdout.split():
        path = ROOT / name
        if (path.name == "runtime_contract.json"
                or not name.startswith(acting_roots)
                or name.startswith(exempt_prefixes)
                or path.suffix not in acting_suffixes
                or not path.is_file()):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if hex_body in text:
            holders.append(name)
    assert holders == [], (
        f"the pin digest is copied into {holders}; it lives in "
        "runtime_contract.json and is read from there")


def test_a_tag_in_the_contract_is_refused_rather_than_honoured():
    """The gate must not be able to become vacuous by a contract edit."""
    tagged = json.loads(json.dumps(CONTRACT))
    tagged["versions"]["attested_on"]["image"] = "vllm/vllm-openai:latest"
    with pytest.raises(RuntimeImageError, match="not a digest reference"):
        pinned_reference(tagged)


# ------------------------------------------------------- the refusal path ---

def test_the_pinned_image_passes():
    pin = pinned_reference()
    record = require_pinned(pin, inspector=_inspector(repo_digests=[pin]))
    assert record["refused"] is False and record["gated"] is True
    assert record["resolved_digest"] == parse_reference(pin)[2]


def test_a_tag_resolving_to_the_pin_passes_and_records_the_digest():
    """The check is on BYTES, not on how the caller spelled them."""
    pin = pinned_reference()
    record = require_pinned("vllm/vllm-openai:latest",
                            inspector=_inspector(repo_digests=[pin]))
    assert record["requested_tag"] == "latest"
    assert record["resolved_digest"] == parse_reference(pin)[2]


def test_a_tag_resolving_to_other_bytes_is_refused_not_warned():
    other = "vllm/vllm-openai@sha256:" + "0" * 64
    with pytest.raises(RuntimeImageError) as exc:
        require_pinned("vllm/vllm-openai:latest",
                       inspector=_inspector(repo_digests=[other]))
    payload = exc.value.payload
    assert payload["refused"] is True
    assert payload["reason"] == "image_pin_mismatch"
    # The refusal must be readable by a program, not only by a human reading a
    # log, and it must name the command that fixes it.
    assert payload["fix"] == f"docker pull {pinned_reference()}"
    assert payload["repo_digests"] == [other]


def test_an_absent_image_is_the_same_refusal_with_the_same_fix():
    with pytest.raises(RuntimeImageError) as exc:
        require_pinned("vllm/vllm-openai:latest",
                       inspector=_inspector(repo_digests=[], present=False))
    assert exc.value.payload["reason"] == "image_absent"
    assert exc.value.payload["fix"] == f"docker pull {pinned_reference()}"


def test_the_local_id_never_decides_anything():
    """The cross-box case, and the reason the gate exists in this shape.

    Sparky's docker reports the manifest digest as ``.Id``; sparklina's reports
    the config digest.  Identical bytes, two ids.  A gate that compared ``.Id``
    would refuse one box forever -- and a refusal that permanently disables one
    box is not a fix.
    """
    pin = pinned_reference()
    sparky = resolve(pin, inspector=_inspector(
        repo_digests=[pin], local_id=pin.split("@", 1)[1]))
    sparklina = resolve(pin, inspector=_inspector(
        repo_digests=[pin], local_id="sha256:" + "89" * 32))
    assert sparky["refused"] is False and sparklina["refused"] is False
    assert sparky["resolved_digest"] == sparklina["resolved_digest"]
    assert sparky["local_id"] != sparklina["local_id"]


def test_an_unpinned_repository_is_stamped_not_refused():
    """Mia's GLM image is a different runtime and carries no pin.

    Gating it against a pin that does not exist would break GLM serves to
    enforce nothing; it is resolved so the receipt still names bytes.
    """
    mia = "prismaquant/glm53-mia-sm121@sha256:" + "7f" * 32
    record = require_pinned("prismaquant/glm53-mia-sm121:487ecf187",
                            inspector=_inspector(repo_digests=[mia]))
    assert record["gated"] is False and record["refused"] is False
    assert record["resolved_digest"] == mia.split("@", 1)[1]


# ------------------------------------------------------------- the wiring ---

def _cli(*args, env_extra=None):
    env = dict(os.environ, TMPDIR="/home/rob/tmp", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(ROOT / "src"), **(env_extra or {}))
    return subprocess.run([sys.executable, "-m", "tessera.serving.runtime_image",
                           *args], env=env, capture_output=True, text=True)


def test_the_cli_prints_the_pin_the_wrappers_default_to():
    proc = _cli("pin")
    assert proc.returncode == 0
    assert proc.stdout.strip() == pinned_reference()


def test_every_wrapper_that_starts_a_container_gates_and_names_no_digest():
    """The wrappers' own text: a `docker run` behind no gate is the defect."""
    exp = ROOT / "experiments"
    starters = sorted(
        p for p in exp.glob("*.sh")
        if re.search(r"^\s*(exec\s+)?docker run", p.read_text(), re.M))
    assert starters, "no container-starting wrapper found; the glob moved"
    for path in starters:
        text = path.read_text()
        assert "runtime_image_require" in text, (
            f"{path.name} starts a container without gating its image")
        assert "vllm/vllm-openai:latest" not in text, (
            f"{path.name} still names the floating tag")


def test_no_campaign_overrides_the_runtime_pin_with_a_floating_image():
    """A delegating campaign must preserve the leaf wrapper's exact pin."""
    assignment = re.compile(
        r"^\s*(?:export\s+)?(?:TESSERA_KL_IMAGE|IMAGE)="
        r"[^\n]*vllm/vllm-openai:latest",
        re.M,
    )
    offenders = [
        str(path.relative_to(ROOT))
        for path in sorted((ROOT / "experiments").rglob("*.sh"))
        if assignment.search(path.read_text())
    ]
    assert offenders == [], (
        "campaigns must derive the runtime contract pin, not override it: "
        + ", ".join(offenders)
    )


def test_the_shell_helper_refuses_and_prints_json_a_program_can_read(tmp_path):
    """``experiments/runtime_image.sh`` is what the wrappers source.

    Driven with a fake ``docker`` on PATH so no daemon is touched: the point
    under test is the wrapper's own control flow -- does a mismatch stop it --
    not whether this box happens to hold the right image today.
    """
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "docker").write_text(
        "#!/usr/bin/env bash\n"
        'printf "sha256:%s\\t[\\"vllm/vllm-openai@sha256:%s\\"]\\n" '
        f'"{"cc" * 32}" "{"00" * 32}"\n')
    (fake / "docker").chmod(0o755)

    script = (f'source "{ROOT}/experiments/runtime_image.sh"\n'
              'runtime_image_require vllm/vllm-openai:latest || exit 2\n'
              'echo "SHOULD NOT REACH"\n')
    env = dict(os.environ, TMPDIR="/home/rob/tmp",
               PATH=f"{fake}:{os.environ['PATH']}")
    proc = subprocess.run(["bash", "-c", script], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "SHOULD NOT REACH" not in proc.stdout
    # The record is on stdout (a program reads it); the prose is on stderr.
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "REFUSED" in proc.stderr
    assert payload["refused"] is True and payload["reason"] == "image_pin_mismatch"
