"""The serve-image pin is a digest, and a mismatch refuses (issue #100).

NOTE ON WHAT IS *NOT* HERE: the digest itself.  The pin lives in exactly one
place -- ``runtime_contract.json``'s ``versions.default_serve_image`` -- and a
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
import box_artifacts

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tessera.serving.contract import contract_path
from tessera.serving.runtime_image import (  # noqa: E402
    ATTESTATION_SCHEMA,
    ATTESTATION_SOURCE,
    CENSUS_ATTESTATION_ENV,
    CENSUS_IMAGE_ENV,
    PIN_CONTRACT_FIELD,
    RuntimeImageError,
    attested_reference,
    container_env,
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
    tagged["versions"]["default_serve_image"] = "vllm/vllm-openai:latest"
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


def test_an_explicit_digest_outside_the_default_repository_is_verified():
    requested = "example/runtime@sha256:" + "f" * 64
    record = require_pinned(requested, inspector=_inspector(repo_digests=[requested]))
    assert record["gated"] is True
    assert record["reason"] == "explicit_digest"
    assert record["resolved_digest"] == requested.split("@", 1)[1]


@pytest.mark.parametrize("present", [False, True])
def test_an_explicit_digest_cannot_borrow_another_images_stamp(present):
    requested = "example/runtime@sha256:" + "f" * 64
    other = "example/runtime@sha256:" + "0" * 64
    with pytest.raises(RuntimeImageError) as exc:
        require_pinned(requested, inspector=_inspector(
            repo_digests=[other] if present else [], present=present))
    payload = exc.value.payload
    assert payload["reason"] == ("image_digest_mismatch" if present else "image_absent")
    assert payload["required"] == requested
    assert payload["fix"] == f"docker pull {requested}"


def test_explicit_digest_stamp_does_not_depend_on_repository_digest_order():
    requested = "example/runtime@sha256:" + "f" * 64
    other = "example/runtime@sha256:" + "0" * 64
    for digests in ([other, requested], [requested, other]):
        record = require_pinned(requested, inspector=_inspector(repo_digests=digests))
        assert record["resolved_digest"] == requested.split("@", 1)[1]


def test_explicit_requested_digest_is_stamped_even_when_the_default_pin_is_an_alias():
    pin = pinned_reference()
    requested = pin.split("@", 1)[0] + "@sha256:" + "f" * 64
    record = require_pinned(requested, inspector=_inspector(repo_digests=[pin, requested]))
    assert record["resolved_digest"] == requested.split("@", 1)[1]


# ----------------------------------------------- the reference it resolved ---

def test_the_record_names_the_resolved_reference_not_only_its_digest():
    """A caller that must NAME the bytes downstream should not rebuild it.

    ``resolved_digest`` is half a reference.  Every consumer that has to hand
    the identity to another process -- a container launcher writing it into the
    environment the census reads -- would otherwise concatenate repository and
    digest itself, which is the second spelling rule 4 exists to prevent.
    """
    pin = pinned_reference()
    record = resolve(pin, inspector=_inspector(repo_digests=[pin]))
    repository, _, digest = parse_reference(pin)
    assert record["resolved_digest"] == digest
    assert record["resolved_reference"] == f"{repository}@{digest}" == pin


def test_a_tag_resolves_to_the_reference_a_launcher_must_pass_on():
    """The caller spelled a tag; what ran is a digest, and that is the value.

    A launcher that passed its own ``IMG`` string into the container would
    export a floating tag, which identifies no runtime at all.
    """
    pin = pinned_reference()
    record = resolve("vllm/vllm-openai:latest", inspector=_inspector(repo_digests=[pin]))
    assert record["requested_tag"] == "latest"
    assert record["resolved_reference"] == pin


def test_an_image_with_no_manifest_digest_names_no_reference():
    """A locally built image carries no ``RepoDigests``; say so, do not invent."""
    record = resolve("local/built:dev", inspector=_inspector(repo_digests=[]))
    assert record["refused"] is False and record["resolved_digest"] is None
    assert record["resolved_reference"] is None


# ------------------------------------------- attesting from inside the run ---
#
# A process inside a container cannot ask the daemon what image it is running:
# there is no socket, and ``/proc/self/mountinfo`` names overlay layer paths
# whose spelling differs between the two snapshotters on the two GB10s and
# never carries a manifest digest at all.  What CAN be done is to have the
# launcher transcribe ``docker image inspect``'s answer verbatim into the
# environment, and have the process inside check its own claim against that
# table.  That is not unforgeable -- a hand ``docker run -e`` can still lie --
# and it is exactly why the receipt names the mechanism instead of asserting
# the image (issue #132).

IMAGE = "example/runtime@sha256:" + "e" * 64


def _attested(reference=IMAGE, **overrides):
    record = resolve(reference, inspector=_inspector(repo_digests=[reference]))
    record.update(overrides)
    return container_env(record)


def test_the_launcher_exports_the_resolved_reference_and_the_whole_record():
    env = _attested()
    assert env[CENSUS_IMAGE_ENV] == IMAGE
    assert json.loads(env[CENSUS_ATTESTATION_ENV])["repo_digests"] == [IMAGE]
    assert set(env) == {CENSUS_IMAGE_ENV, CENSUS_ATTESTATION_ENV}


def test_an_unattestable_image_exports_nothing_rather_than_an_empty_claim():
    """No ``RepoDigests`` means no identity; an empty variable would read as one."""
    record = resolve("local/built:dev", inspector=_inspector(repo_digests=[]))
    assert container_env(record) == {}


def test_the_requested_image_is_checked_against_the_daemons_table():
    block = attested_reference(IMAGE, env=_attested())
    assert block["schema"] == ATTESTATION_SCHEMA
    # The mechanism, as a value a gate switches on -- not prose.
    assert block["source"] == ATTESTATION_SOURCE
    assert block["image"] == IMAGE
    # The daemon's own answer travels verbatim, so a reader can redo the join.
    assert block["record"]["resolved_reference"] == IMAGE
    assert IMAGE in block["record"]["repo_digests"]


@pytest.mark.parametrize("env", [
    {},
    {CENSUS_IMAGE_ENV: IMAGE},
    {CENSUS_ATTESTATION_ENV: "{}"},
    {CENSUS_IMAGE_ENV: "", CENSUS_ATTESTATION_ENV: ""},
])
def test_an_absent_attestation_refuses_rather_than_trusting_the_caller(env):
    with pytest.raises(RuntimeImageError) as exc:
        attested_reference(IMAGE, env=env)
    assert exc.value.payload["reason"] == "image_attestation_missing"


def test_an_image_the_launcher_did_not_start_is_refused():
    """The #132 defect, as a gate: the operator names other bytes."""
    other = "example/runtime@sha256:" + "d" * 64
    with pytest.raises(RuntimeImageError) as exc:
        attested_reference(other, env=_attested())
    payload = exc.value.payload
    assert payload["reason"] == "image_attestation_mismatch"
    assert payload["requested"] == other and payload["attested"] == IMAGE


@pytest.mark.parametrize("attestation", ["not json", "[]", '{"schema": "other/1"}'])
def test_an_unreadable_attestation_is_refused_not_ignored(attestation):
    env = {CENSUS_IMAGE_ENV: IMAGE, CENSUS_ATTESTATION_ENV: attestation}
    with pytest.raises(RuntimeImageError) as exc:
        attested_reference(IMAGE, env=env)
    assert exc.value.payload["reason"] == "image_attestation_unreadable"


def test_the_two_variables_must_agree_with_each_other():
    """Half a forged pair is still a forged pair."""
    env = dict(_attested(), **{CENSUS_IMAGE_ENV: "example/runtime@sha256:" + "d" * 64})
    with pytest.raises(RuntimeImageError) as exc:
        attested_reference(env[CENSUS_IMAGE_ENV], env=env)
    assert exc.value.payload["reason"] == "image_attestation_inconsistent"


def test_a_record_that_records_its_own_refusal_attests_nothing():
    env = _attested(refused=True, reason="image_digest_mismatch")
    with pytest.raises(RuntimeImageError) as exc:
        attested_reference(IMAGE, env=env)
    assert exc.value.payload["reason"] == "image_attestation_refused"


# ------------------------------------------------------------- the wiring ---

def _cli(*args, env_extra=None):
    env = dict(os.environ, TMPDIR=box_artifacts.scratch_tmpdir(), CUDA_VISIBLE_DEVICES="",
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
    env = dict(os.environ, TMPDIR=box_artifacts.scratch_tmpdir(),
               PATH=f"{fake}:{os.environ['PATH']}")
    proc = subprocess.run(["bash", "-c", script], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "SHOULD NOT REACH" not in proc.stdout
    # The record is on stdout (a program reads it); the prose is on stderr.
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "REFUSED" in proc.stderr
    assert payload["refused"] is True and payload["reason"] == "image_pin_mismatch"
