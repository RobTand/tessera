"""The exporter's producer selection, its receipt, and the joined encode.

Two contracts live here, both sealed into the artifacts the exporter writes.

**Producer selection** (``TESSERA_PRODUCER_PYTHON``): a partition producer
that claims a genuine lineage must run under the interpreter it named, from
that interpreter's *installed* distribution -- not a sys.path or PYTHONPATH
shadow, however equal the bytes -- against a clean checkout of a commit that
descends from the genuine ancestor, projected to the files a wheel actually
ships (``tessera._dev`` is repository tooling and no wheel carries it).  The
receipt binds every one of those facts, and the partition stamp seals it, so
two parts of one checkpoint cannot join unless one producer wrote both.

**Joined fresh encodes** (``--encode-batch``): N same-shape units of one
stack ride one ``encode_linears_planes`` call.  Every blob is byte-identical
to the one-unit encode, so this is a machine schedule -- and the value is
sealed into the part identity anyway (a conservative exact comparison: parts
batched differently refuse to merge), because a knob whose only defence is a
byte-equality test stays a knob only while someone runs the test.

The auth success path runs the real exporter in a subprocess under a fixture
install (a synthetic site-packages, the ``test_serving_plan_schema``
precedent): this process imports tessera from the checkout, which is exactly
the shadow the contract refuses.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

from tessera import export_serving as export
from tessera import serving_parts as parts
from tessera import source_profiles

ROOT = Path(__file__).resolve().parents[1]
HIDDEN, MOE_INTER, EXPERTS = 128, 256, 3
LAYER = "model.language_model.layers.1"
STACK = f"{LAYER}.mlp.experts"
PART_ARGV = ("--partition", "0/1", "--partition-runtime-image",
             "test/image@sha256:" + "a" * 64)


# --------------------------------------------------------------------------
# Fixtures: the miniature checkpoint and the in-process exporter run
# --------------------------------------------------------------------------

def _config(experts=EXPERTS):
    return {"architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {"hidden_size": HIDDEN, "moe_intermediate_size": MOE_INTER,
                            "num_hidden_layers": 2, "n_routed_experts": experts}}


def _checkpoint(experts=EXPERTS):
    """Unstacked 2-D expert projections -- the real census layout, one unit per name."""
    generator = torch.Generator().manual_seed(11)

    def normal(*shape):
        return torch.randn(*shape, generator=generator) * 0.02

    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": normal(2 * HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.up_proj.weight": normal(2 * HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.down_proj.weight": normal(HIDDEN, 2 * HIDDEN),
        f"{LAYER}.mlp.shared_experts.gate_proj.weight": normal(MOE_INTER, HIDDEN),
        f"{LAYER}.mlp.shared_experts.up_proj.weight": normal(MOE_INTER, HIDDEN),
        f"{LAYER}.mlp.shared_experts.down_proj.weight": normal(HIDDEN, MOE_INTER),
        f"{LAYER}.mlp.gate.weight": normal(experts, HIDDEN),
        "lm_head.weight": normal(HIDDEN, HIDDEN),
        "model.embed_tokens.weight": normal(HIDDEN, HIDDEN),
    }
    for index in range(experts):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            shape = (HIDDEN, MOE_INTER) if projection == "down_proj" else (MOE_INTER, HIDDEN)
            tensors[f"{STACK}.{index}.{projection}.weight"] = normal(*shape)
    return tensors


def _write(tmp_path, tensors, config=None):
    src = tmp_path / "src"
    src.mkdir(exist_ok=True)
    safetensors_torch.save_file({k: v.contiguous() for k, v in tensors.items()},
                                str(src / "model.safetensors"), metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps(config or _config()))
    return src


_FIXTURE_OUTPUT_SIZES = {
    "language_model.model.layers.*.mlp.down_proj": [128],
    "language_model.model.layers.*.mlp.gate_up_proj": [256, 256],
    "language_model.model.layers.*.mlp.shared_experts.down_proj": [128],
    "language_model.model.layers.*.mlp.shared_experts.gate_up_proj": [64, 64],
}


def _declare_fixture_geometry(monkeypatch):
    """Use the fixture's own partition lists, and nothing else."""
    import copy
    from tessera.serving.contract import construction_entry as live_entry
    real = live_entry

    def _entry(architectures, contract=None):
        entry = real(architectures) if contract is None else real(architectures, contract)
        if entry is None or entry.get("architecture") != "Glm5NextForConditionalGeneration":
            return entry
        entry = copy.deepcopy(entry)
        entry.setdefault("output_sizes", {}).update(_FIXTURE_OUTPUT_SIZES)
        return entry

    monkeypatch.setattr(export, "construction_entry", _entry)


def _export(tmp_path, monkeypatch, tensors, plan, *extra, config=None):
    _declare_fixture_geometry(monkeypatch)
    src = _write(tmp_path, tensors, config)
    out = tmp_path / "out"
    argv = ["export", str(src), str(out), "--grid", "E4M3", "--q256", "1024",
            "--passthrough-unrouted", *extra]
    if plan is not None:
        plan_path = tmp_path / "plan.json"
        plan_path.write_text(json.dumps(plan))
        argv += ["--plan-json", str(plan_path)]
    monkeypatch.setattr("sys.argv", argv)
    export.main()
    return out


def _part_manifest(out: Path) -> dict:
    return json.loads((out / "tessera_serving_manifest.json").read_text())


# --------------------------------------------------------------------------
# Producer selection: the refusals, in process
# --------------------------------------------------------------------------

def test_no_selection_authenticates_nothing(monkeypatch):
    monkeypatch.delenv("TESSERA_PRODUCER_PYTHON", raising=False)
    monkeypatch.delenv("TESSERA_PRODUCER_SOURCE", raising=False)
    assert export.authenticate_producer_python() is None


def test_a_relative_selection_refuses_by_variable_name(monkeypatch):
    monkeypatch.setenv("TESSERA_PRODUCER_PYTHON", "python3")
    with pytest.raises(SystemExit, match="TESSERA_PRODUCER_PYTHON"):
        export.authenticate_producer_python()


def test_a_selection_this_process_is_not_refuses_by_variable_name(monkeypatch):
    monkeypatch.setenv("TESSERA_PRODUCER_PYTHON", "/usr/bin/definitely-not-python3")
    with pytest.raises(SystemExit) as caught:
        export.authenticate_producer_python()
    message = str(caught.value)
    assert "TESSERA_PRODUCER_PYTHON" in message
    assert sys.executable in message, "the refusal names the interpreter that is running"


def test_a_selection_without_a_source_reference_refuses(monkeypatch):
    monkeypatch.setenv("TESSERA_PRODUCER_PYTHON", sys.executable)
    monkeypatch.delenv("TESSERA_PRODUCER_SOURCE", raising=False)
    with pytest.raises(SystemExit, match="TESSERA_PRODUCER_SOURCE"):
        export.authenticate_producer_python()


def _git(*args, cwd, check=True):
    return subprocess.run(["git", *args], cwd=str(cwd), check=check,
                          capture_output=True, text=True)


def _source_reference(tmp_path, *, dirty=False, ancestor=True) -> tuple[Path, str]:
    """A fixture checkout; returns (root, the fixture's ancestry reference).

    With ``ancestor`` the package commit is a real DESCENDANT of the
    reference commit; without it the package is the root commit itself, so
    nothing descends from the reference a caller names.  The package bytes
    are this tree's own and the pyproject is this tree's own, so the
    packaging projection is the real one.
    """
    root = tmp_path / "checkout"
    root.mkdir()
    _git("init", "-q", "-b", "main", cwd=root)
    _git("config", "user.email", "fixture@tessera", cwd=root)
    _git("config", "user.name", "fixture", cwd=root)
    if ancestor:
        (root / "README").write_text("the reference commit carries no package\n")
        shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
        _git("add", "-A", cwd=root)
        _git("commit", "-q", "-m", "reference", cwd=root)
        base = _git("rev-parse", "HEAD", cwd=root).stdout.strip()
        shutil.copytree(ROOT / "src" / "tessera", root / "src" / "tessera",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (root / "README").write_text("the package commit\n")
        _git("add", "-A", cwd=root)
        _git("commit", "-q", "-m", "package", cwd=root)
    else:
        shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
        shutil.copytree(ROOT / "src" / "tessera", root / "src" / "tessera",
                        ignore=shutil.ignore_patterns("__pycache__"))
        (root / "README").write_text("the only commit\n")
        _git("add", "-A", cwd=root)
        _git("commit", "-q", "-m", "package as root", cwd=root)
        base = _git("rev-parse", "HEAD", cwd=root).stdout.strip()
    if dirty:
        tracked = root / "src" / "tessera" / "exact.py"
        tracked.write_text(tracked.read_text() + "\n# dirty\n")
    return root, base


def test_a_dirty_source_reference_refuses_by_variable_name(monkeypatch, tmp_path):
    root, _base = _source_reference(tmp_path, dirty=True)
    monkeypatch.setenv("TESSERA_PRODUCER_PYTHON", sys.executable)
    monkeypatch.setenv("TESSERA_PRODUCER_SOURCE", str(root / "src" / "tessera"))
    with pytest.raises(SystemExit) as caught:
        export.authenticate_producer_python()
    assert "TESSERA_PRODUCER_SOURCE" in str(caught.value)
    assert "dirty" in str(caught.value)


def test_a_source_reference_outside_the_lineage_refuses(monkeypatch, tmp_path):
    root, _base = _source_reference(tmp_path, ancestor=False)
    unrelated = _git("commit-tree", "HEAD^{tree}", "-m", "unrelated root", cwd=root).stdout.strip()
    monkeypatch.setenv("TESSERA_PRODUCER_PYTHON", sys.executable)
    monkeypatch.setenv("TESSERA_PRODUCER_SOURCE", str(root / "src" / "tessera"))
    with pytest.raises(SystemExit) as caught:
        export.authenticate_producer_python(expected_package=root / "src" / "tessera",
                                            descends_from=unrelated)
    assert "TESSERA_PRODUCER_SOURCE" in str(caught.value)
    assert f"does not descend from {unrelated}" in str(caught.value)


def test_the_genuine_ancestor_reference_is_a_full_commit_name():
    assert export.GENUINE_PRODUCER_ANCESTOR == \
        "b770727c50eef822132518bdc4fd6efe84359c9e"


def test_this_process_imports_a_shadow_and_refuses_by_variable_name(monkeypatch, tmp_path):
    """The session imports tessera from the checkout, which is no install.

    That is precisely the legacy experiments shim: ``checkout/src`` on the
    path, bytes equal to anything you like, and no distribution behind it.
    Selection must refuse it by name whatever the bytes say.
    """
    root, _base = _source_reference(tmp_path)
    monkeypatch.setenv("TESSERA_PRODUCER_PYTHON", sys.executable)
    monkeypatch.setenv("TESSERA_PRODUCER_SOURCE", str(root / "src" / "tessera"))
    with pytest.raises(SystemExit) as caught:
        export.authenticate_producer_python(expected_package=root / "src" / "tessera",
                                            descends_from=_fixture_ancestor(root))
    message = str(caught.value)
    assert "TESSERA_PRODUCER_PYTHON" in message
    assert "installed" in message


def test_a_foreign_loaded_submodule_refuses_with_installed_init(tmp_path, monkeypatch):
    """THE MIXED ORIGIN: an installed ``__init__`` beside a submodule imported
    from elsewhere.  The fake init entry stands in for the install; the real
    ``tessera.export_serving`` of this process stays imported from the
    checkout -- exactly the shape that must refuse, naming the exporter
    module, before anything else in the authentication can pass."""
    site = _fixture_install(tmp_path / "site")
    installed_pkg = site / "tessera"
    payload = {p.relative_to(installed_pkg).as_posix(): p.read_bytes()
               for p in installed_pkg.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    fake = types.ModuleType("tessera")
    fake.__file__ = str(installed_pkg / "__init__.py")
    fake.__path__ = [str(installed_pkg)]
    monkeypatch.setitem(sys.modules, "tessera", fake)
    with pytest.raises(SystemExit) as caught:
        export._require_loaded_origins(installed_pkg, payload)
    message = str(caught.value)
    assert "TESSERA_PRODUCER_PYTHON" in message
    assert "loaded module" in message and "mixed origin" in message.lower()


def _fixture_ancestor(root):
    """The fixture's chosen ancestry reference: its base commit."""
    return _git("rev-list", "--max-parents=0", "HEAD", cwd=root).stdout.strip()


# --------------------------------------------------------------------------
# Producer selection: the receipt, in a real subprocess under a fixture install
# --------------------------------------------------------------------------

def _fixture_install(target: Path) -> Path:
    """A synthetic site-packages: the package plus a dist-info naming it."""
    target.mkdir()
    config = source_profiles.packaged_projection_config((ROOT / "pyproject.toml").read_bytes())
    tracked = [p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / "tessera").rglob("*")
               if p.is_file() and "__pycache__" not in p.parts]
    for name in source_profiles.shipped_payload_paths(config, tracked):
        original = ROOT / "src" / name
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, destination)
    from tessera import DISTRIBUTION

    dist_info = target / f"{DISTRIBUTION.replace('-', '_')}-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 0.1\n")
    return target


def _selected_export_argv(src, out, plan_path):
    return [str(src), str(out), "--device", "cpu",
            "--grid", "E4M3", "--q256", "1024", "--passthrough-unrouted",
            "--plan-json", str(plan_path), *PART_ARGV]


def _selected_export_env(site, repo):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(site)
    env["TESSERA_PRODUCER_PYTHON"] = sys.executable
    env["TESSERA_PRODUCER_SOURCE"] = str(repo / "src" / "tessera")
    env.pop("TESSERA_GIT", None)
    return env


def _run_selected_exporter(tmp_path, site, repo, out_name="out"):
    """The real exporter main, in a subprocess, under the fixture install.

    The driver binds only the ANCESTRY REFERENCE to the fixture's own
    (main passes the genuine b770 constant, which no fixture can forge);
    every check around it -- interpreter, install, projection, clean HEAD,
    loaded origins, the receipt and its sealing -- is the production code
    reached through ``export_serving.main`` itself.
    """
    src = _write(tmp_path, _checkpoint(), _config())
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({STACK: {"grid": "E4M3", "q256": 896}}))
    out = tmp_path / out_name
    base = _fixture_ancestor(repo)
    driver = tmp_path / "driver.py"
    sizes = dict(_FIXTURE_OUTPUT_SIZES)
    driver.write_text(
        "import sys\n"
        "from tessera import export_serving as export\n"
        "real = export.authenticate_producer_python\n"
        f"export.authenticate_producer_python = "
        f"lambda *a, **k: real(*a, descends_from={base!r}, **k)\n"
        # The fixture is a miniature; state its own partition lists here,
        # inside the subprocess, beside the ancestry binding above.
        "import copy\n"
        "from tessera.serving.contract import construction_entry as live_entry\n"
        f"small = copy.deepcopy(live_entry(['Glm5NextForConditionalGeneration']))\n"
        f"small['output_sizes'].update({sizes!r})\n"
        "export.construction_entry = lambda architectures, contract=None: small\n"
        "export.main()\n")
    done = subprocess.run(
        [sys.executable, str(driver), *_selected_export_argv(src, out, plan_path)],
        cwd=str(tmp_path), env=_selected_export_env(site, repo),
        capture_output=True, text=True, timeout=900)
    return done, out


def test_a_selected_producer_seals_its_receipt_and_commit(tmp_path, monkeypatch):
    """The one full-authentication path in this suite, end to end.

    The subprocess runs under the fixture install (its tessera is an
    installed distribution, not a path entry), against the fixture checkout
    (clean, a real descendant of the fixture's chosen reference).  The part
    identity carries the receipt; the manifest's commit stamp is the
    receipt's verified head -- a wheel install has no git and no direct_url
    commit, so without the receipt the exporter could not stamp at all.
    """
    repo, _base = _source_reference(tmp_path)
    site = _fixture_install(tmp_path / "site")
    monkeypatch.chdir(tmp_path)
    done, out = _run_selected_exporter(tmp_path, site, repo)
    assert done.returncode == 0, done.stderr[-4000:]
    manifest = _part_manifest(out)
    identity = manifest["export_partition"]["identity"]
    receipt = identity["producer"]
    head = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    assert receipt["schema"] == "tessera.producer_python.v1"
    assert receipt["requested_interpreter"] == receipt["interpreter"] == sys.executable
    assert receipt["executable_sha256"] == \
        hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
    assert receipt["sys_prefix"] == sys.prefix
    # The producer's observed execution environment, as observed -- never a
    # claim about the reader/runtime image the parts are destined for.
    assert receipt["python_version"] == platform.python_version()
    assert receipt["torch_version"] == torch.__version__
    assert receipt["git_head"] == head
    assert receipt["descends_from"] == _fixture_ancestor(repo)
    assert receipt["expected_package_sha256"] == receipt["package_sha256"]
    assert receipt["runtime_contract_sha256"] == hashlib.sha256(
        (site / "tessera" / "serving" / "runtime_contract.json").read_bytes()).hexdigest()
    # The projection is the wheel's roster: _dev is repository tooling and
    # ships in neither the expectation nor the install.
    assert "producer" in manifest
    assert manifest["git"] == head, "the selected producer stamps its verified head"
    assert manifest["encode_batch"] == 1
    assert identity["options"]["encode_batch"] == 1
    # No variable claim stands in for the verification: TESSERA_GIT was not
    # in the environment, and the stamp names the fixture's own HEAD.


def test_an_installed_payload_that_drifts_from_the_source_refuses(tmp_path, monkeypatch):
    """Installed-vs-source equality is checked on the bytes: a transient
    mismatch (an install one edit behind its checkout) refuses by name."""
    repo, _base = _source_reference(tmp_path)
    site = _fixture_install(tmp_path / "site")
    drifted = site / "tessera" / "exact.py"
    drifted.write_text(drifted.read_text() + "\n# one edit behind\n")
    monkeypatch.chdir(tmp_path)
    done, out = _run_selected_exporter(tmp_path, site, repo)
    assert done.returncode != 0
    assert "TESSERA_PRODUCER_SOURCE" in done.stderr
    assert not out.exists() or not (out / "tessera_serving_manifest.json").exists()


def test_an_installed_payload_with_an_extra_file_refuses_by_roster(tmp_path, monkeypatch):
    """The roster is compared before any byte: a file the source never shipped
    refuses by name, with the file in the message."""
    repo, _base = _source_reference(tmp_path)
    site = _fixture_install(tmp_path / "site")
    (site / "tessera" / "unshipped_module.py").write_text("# not in the source\n")
    monkeypatch.chdir(tmp_path)
    done, out = _run_selected_exporter(tmp_path, site, repo)
    assert done.returncode != 0
    assert "TESSERA_PRODUCER_PYTHON" in done.stderr
    assert "projected wheel roster" in done.stderr and "unshipped_module.py" in done.stderr
    assert not out.exists() or not (out / "tessera_serving_manifest.json").exists()


# --------------------------------------------------------------------------
# Merge: one producer per checkpoint, one scale file, one batch schedule
# --------------------------------------------------------------------------

def _merge(tmp_path, edit=None):
    """Two parts per the ``test_serving_parts`` fixture, both sealed by ONE
    receipt; ``edit`` then mutates part 1's identity.  The edits are what a
    real second part could differ by -- a different producer, a missing
    receipt, another batch schedule, another scale file."""
    from test_serving_parts import _fixture

    source, paths = _fixture(tmp_path)
    for path in paths:
        manifest_path = path / "tessera_serving_manifest.json"
        value = json.loads(manifest_path.read_text())
        value["export_partition"]["identity"]["producer"] = dict(_RECEIPT)
        manifest_path.write_text(json.dumps(value))
    if edit is not None:
        manifest_path = paths[1] / "tessera_serving_manifest.json"
        value = json.loads(manifest_path.read_text())
        edit(value["export_partition"]["identity"])
        manifest_path.write_text(json.dumps(value))
    return parts.merge_serving_parts(paths, tmp_path / "merged", source)


_RECEIPT = {"schema": "tessera.producer_python.v1", "git_head": "c" * 64}


def test_parts_from_different_producers_refuse_to_merge(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        _merge(tmp_path, lambda identity: identity.update(
            {"producer": {**_RECEIPT, "git_head": "d" * 64}}))


def test_a_part_without_a_producer_receipt_refuses_to_join_one_with(tmp_path):
    def _remove(identity):
        del identity["producer"]

    with pytest.raises(ValueError, match="identity"):
        _merge(tmp_path, _remove)


def test_parts_at_different_encode_batches_refuse_to_merge(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        _merge(tmp_path, lambda identity: identity["options"].update({"encode_batch": 4}))


def test_parts_with_different_scale_bindings_refuse_to_merge(tmp_path):
    with pytest.raises(ValueError, match="identity"):
        _merge(tmp_path, lambda identity: identity["options"].update(
            {"input_scales_sha256": "e" * 64}))


# --------------------------------------------------------------------------
# The input-scale binding seals the bytes the roles consumed
# --------------------------------------------------------------------------

NVFP4_HIDDEN = NVFP4_INTER = 64


def _nvfp4_scales(experts=1):
    return {f"{STACK}.{e}.{projection}.input_global_scale": float(3 + 2 * i + 7 * e)
            for e in range(experts)
            for i, projection in enumerate(("gate_proj", "up_proj", "down_proj"))}


def test_the_scale_binding_seals_the_bytes_the_roles_consumed(tmp_path, monkeypatch):
    """The identity's ``input_scales_sha256`` and every written scale come
    from ONE read.  The regression rewrites the file after the identity
    seals it but before the old code re-opened it: the roles had to carry
    the sealed scalars, or the export had to refuse -- never both."""
    from tessera import encoder_identity

    experts = 1
    scales = _nvfp4_scales(experts)
    donor = tmp_path / "input_scales.safetensors"
    safetensors_torch.save_file(
        {key: torch.tensor([value], dtype=torch.float32) for key, value in scales.items()},
        str(donor), metadata={"format": "pt"})
    original = donor.read_bytes()

    real_fixture_id = encoder_identity.encoder_fixture_id

    def _rewrite_donor_after_seal():
        # The partition block calls this after the options are sealed; the
        # rewrite lands in the window the old two-pass read raced through.
        try:
            return real_fixture_id()
        finally:
            safetensors_torch.save_file(
                {key: torch.tensor([value + 100.0], dtype=torch.float32)
                 for key, value in scales.items()},
                str(donor), metadata={"format": "pt"})

    monkeypatch.setattr(encoder_identity, "encoder_fixture_id",
                        _rewrite_donor_after_seal)
    generator = torch.Generator().manual_seed(492)
    tensors = {f"{STACK}.0.{projection}.weight":
               torch.randn(NVFP4_INTER, NVFP4_HIDDEN, generator=generator) * 0.02
               for projection in ("gate_proj", "up_proj", "down_proj")}
    tensors["model.language_model.layers.0.norm.weight"] = torch.randn(
        NVFP4_HIDDEN, generator=generator).bfloat16()
    config = _config(experts)
    config["text_config"].update(hidden_size=NVFP4_HIDDEN, moe_intermediate_size=NVFP4_INTER)
    out = _export(tmp_path, monkeypatch, tensors, {STACK: {"grid": "E2M1x2", "q256": 896}},
                  "--device", "cpu", "--input-scales", str(donor), *PART_ARGV, config=config)
    manifest = _part_manifest(out)
    options = manifest["export_partition"]["identity"]["options"]
    assert options["input_scales_sha256"] == hashlib.sha256(original).hexdigest()
    with safetensors_torch.safe_open(str(out / "model.safetensors"), framework="pt") as handle:
        for key, value in scales.items():
            assert handle.get_tensor(key).tolist() == [value], \
                f"{key} carries scalars the sealed identity does not describe"


# --------------------------------------------------------------------------
# Joined fresh encodes: planning, staging, and the CLI
# --------------------------------------------------------------------------

def test_joined_encodes_take_each_key_in_loop_order():
    keys = ["a", "b", "a", "a", "b", "a", "c"]
    assert export.plan_joined_encodes(keys, 2) == [[0, 2], [1, 4], [3, 5], [6]]
    assert export.plan_joined_encodes(keys, 1) == [[i] for i in range(len(keys))]
    assert export.plan_joined_encodes(keys, 9) == [[0, 2, 3, 5], [1, 4], [6]]
    with pytest.raises(ValueError, match="at least one unit"):
        export.plan_joined_encodes(keys, 0)


def test_a_differing_per_unit_block_schedule_splits_and_keeps_order():
    """``--ldlq-block-budget`` derives each unit's LDLQ block from its own
    Hessian, so one join key can hold differing shared schedules: the
    exporter splits where they differ -- the batched entry's grammar owns
    what must be shared and refuses a mixed mapping by key -- and commits
    every unit in its own order."""
    prepared = [({"tensor": f"u{i}"}, None, mapping)
                for i, mapping in enumerate([
                    {"ldl": "H0", "ldl_block": 32}, {"ldl_block": 64},
                    {"ldl_block": 32}, {"refit_reach_floor": True}])]
    assert export.split_by_shared_schedule(prepared) == [[0, 2], [1], [3]]


def test_a_joined_batch_reads_each_tensor_once_and_hands_it_back_to_the_loop():
    reads = []

    class Handle:
        def get_tensor(self, name):
            reads.append(name)
            return f"T:{name}"

    units = [(f"n{i}", {"tensor": f"u{i}", "stack": "s", "rows": 4, "cols": 8})
             for i in range(5)]
    # u2 is another shape: it rides alone and the a-key batches skip over it.
    units[2][1]["rows"] = 8
    calls = []

    def encode(members):
        calls.append([unit["tensor"] for unit, _source in members])
        return [(unit["tensor"], source) for unit, source in members]

    joined = export.JoinedExpertEncode(Handle(), units, 2, encode)
    got = []
    for name, unit in units:
        got.append(joined.take(name, unit, joined.source(name)))
    assert calls == [["u0", "u1"], ["u2"], ["u3", "u4"]]
    assert got == [(f"u{i}", f"T:n{i}") for i in range(5)]
    assert reads == [f"n{i}" for i in range(5)], "each source tensor is read exactly once"
    assert not joined.staged and not joined.done


@pytest.mark.parametrize("batch", ["2", "5"])
def test_joined_expert_encodes_group_across_names_and_write_the_one_unit_bytes(
        tmp_path, monkeypatch, batch):
    """Real unstacked source: every expert projection is its own tensor name,
    so ``expert_units[name]`` holds ONE unit and only grouping ACROSS names
    within the shard ever joins anything.  ``encode_linears_planes`` must
    observe the requested batch (this asserts the joined call, not a mock of
    it), and the bytes are the one-unit export's bytes."""
    config = _config()
    plan = {STACK: {"grid": "E4M3", "q256": 896}}
    outs = {}
    for label, extra in (("one", []), ("joined", ["--encode-batch", batch])):
        work = tmp_path / label
        work.mkdir()
        outs[label] = _export(work, monkeypatch, _checkpoint(), plan,
                              "--device", "cpu", *PART_ARGV, *extra, config=config)
        if label == "one":
            observed = []
            real = export.encode_linears_planes

            def spy(weights, **kwargs):
                observed.append((len(weights), tuple(kwargs.get("names") or ())))
                return real(weights, **kwargs)

            monkeypatch.setattr(export, "encode_linears_planes", spy)
    assert (outs["one"] / "model.safetensors").read_bytes() == \
        (outs["joined"] / "model.safetensors").read_bytes()
    joined_sizes = [size for size, _names in observed if size > 1]
    assert joined_sizes, "the joined entry never saw a batch"
    assert any(size == int(batch) for size in joined_sizes)
    for size, names in observed:
        assert len(names) == size
        if size > 1:
            # One join key: one stack, and every member a DISTINCT source
            # name -- cross-name grouping is what a real census needs, and
            # per-name grouping would never have produced this call.
            assert len({name.split(".experts.")[0] for name in names}) == 1
            assert len(set(names)) == size
    identities = [_part_manifest(out)["export_partition"]["identity"]["options"]
                  for out in outs.values()]
    assert identities[0]["encode_batch"] == 1 and identities[1]["encode_batch"] == int(batch)
    differing = {k for k in set(identities[0]) | set(identities[1])
                 if identities[0].get(k) != identities[1].get(k)}
    assert differing == {"encode_batch"}, \
        "the batch schedule is sealed, and nothing else moved"
    joined_manifest = _part_manifest(outs["joined"])
    assert joined_manifest["encode_batch"] == int(batch)
    observed = joined_manifest["encode_batch_observed"]
    # THE EFFECTIVE WIDTHS, as evidence: at least one call ran at the
    # requested width, every call is at most it, and the histogram accounts
    # for every joined call the two join keys produced (three projections
    # x three experts = nine units: gate/up key of six, down key of three).
    assert observed[str(int(batch))] >= 1
    assert max(map(int, observed)) == int(batch)
    assert sum(observed.values()) == (-(-6 // int(batch)) + -(-3 // int(batch)))


@pytest.mark.parametrize("extra", [["--encode-batch", "0"],
                                   ["--encode-batch", "2", "--cached-units", "x.json"]])
def test_an_encode_batch_the_exporter_cannot_honour_is_refused(tmp_path, monkeypatch, extra):
    with pytest.raises(SystemExit):
        _export(tmp_path, monkeypatch, _checkpoint(experts=1), None, "--device", "cpu",
                *extra, config=_config(1))


# --------------------------------------------------------------------------
# The packaging projection: what a wheel ships of a tracked tree
# --------------------------------------------------------------------------

def test_the_projection_omits_exactly_what_the_wheel_omits():
    config = source_profiles.packaged_projection_config(
        (ROOT / "pyproject.toml").read_bytes())
    tracked = ["src/tessera/__init__.py", "src/tessera/exact.py",
               "src/tessera/_dev/suite_source.py", "src/tessera/_dev/native_identity.py",
               "src/tessera/serving/contract.py", "src/tessera/serving/runtime_contract.json",
               "src/tessera/serving/csrc/window_gemv.cu",
               "src/tessera/serving/csrc/undeclared.c",
               "src/tessera/py.typed", "src/tessera/README.md"]
    shipped = source_profiles.shipped_payload_paths(config, tracked)
    assert "tessera/__init__.py" in shipped and "tessera/exact.py" in shipped
    assert "tessera/serving/contract.py" in shipped
    assert "tessera/serving/runtime_contract.json" in shipped
    assert "tessera/serving/csrc/window_gemv.cu" in shipped
    assert not any(name.startswith("tessera/_dev") for name in shipped)
    assert "tessera/serving/csrc/undeclared.c" not in shipped
    assert "tessera/py.typed" not in shipped and "tessera/README.md" not in shipped
    assert shipped == sorted(shipped)
