"""The published serving-plan schema and the supported exporter entry point.

Tessera #687.  ``--plan-json`` used to be an exporter-internal convention: the
entry shape lived in ``PlanSnapshot``'s argument-time loop, and a producer
that wanted the fused-group rule before writing a plan had to import an
experiments driver to get ``module_scheme_key``.  This file pins what #687
publishes:

* ``tessera.serving_plan.v1`` -- the schema ``tessera.serving_plan`` validates
  a plan against, refusing an out-of-schema entry BY NAME;
* ``module_scheme_key`` and ``family_for`` importable from the package, with
  the same semantics the exporter's fused-group check applies;
* ``python -m tessera.export_serving`` as the supported entry point, exporting
  a small fixture checkpoint on CPU from a schema-valid plan.

The producer sidecar is a neutral ``producer_annotations`` object the schema
admits and the exporter copies through verbatim without reading (#691 item
6): whatever the producer's own accounting knows rides under the producer's
own names.  The ``prismaquant_charged_bits*`` field names the first #687 cut
published are retired -- a plan carrying them is refused as unknown fields.
Tessera names the fields a plan may carry, not the producer's modules --
nothing here imports or reads PrismaQuant (#599).

A plan may declare its schema under the reserved top-level ``"schema"`` key
(#691 item 5); the exporter records the declaration in the manifest beside
the published plan.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="the exporter imports torch at module level")
from safetensors.torch import save_file  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

BODY = "model.language_model.layers."
TENSOR = BODY + "0.mlp.down_proj.weight"
STACK = BODY + "3.mlp.experts"


def _plan() -> dict:
    """One plan holding every documented entry shape at once."""
    return {
        TENSOR: {"grid": "E4M3", "q256": 1024},
        BODY + "1.mlp.down_proj.weight": "BF16",
        BODY + "2.mlp.down_proj.weight": "PASSTHROUGH",
        STACK: {"grid": "E2M1x2", "q256": 896,
                "source_layout": "out_first_chunked"},
    }


def _validate(entries):
    from tessera.serving_plan import validate_serving_plan
    return validate_serving_plan(entries)


def test_the_schema_accepts_every_documented_entry_shape():
    _validate(_plan())


def test_the_schema_accepts_neutral_producer_annotations():
    annotated = _plan()
    annotated[TENSOR]["producer_annotations"] = {
        "charged_bits": 4096.0, "charged_bits_exact": [4096, 1]}
    annotated[STACK]["producer_annotations"] = {"charged_bits": 1048576.0}
    _validate(annotated)


def test_producer_annotations_must_be_an_object():
    plan = _plan()
    plan[TENSOR]["producer_annotations"] = 4096.0
    with pytest.raises(ValueError, match="producer_annotations"):
        _validate(plan)


def test_the_retired_pq_named_fields_are_no_longer_schema():
    """#691 item 6 retired the ``prismaquant_*`` names the first #687 cut
    published: a plan carrying them is refused as unknown fields."""
    plan = _plan()
    plan[TENSOR]["prismaquant_charged_bits"] = 4096.0
    with pytest.raises(ValueError, match="prismaquant_charged_bits"):
        _validate(plan)


def test_a_plan_may_declare_its_schema():
    plan = _plan()
    plan["schema"] = "tessera.serving_plan.v1"
    _validate(plan)


def test_a_plan_declaring_another_schema_is_refused():
    plan = _plan()
    plan["schema"] = "tessera.serving_plan.v2"
    with pytest.raises(ValueError, match="schema"):
        _validate(plan)


def test_schema_is_reserved_not_an_entry():
    """An object NAMED ``"schema"`` is read as the declaration, never as a
    tensor entry -- so a mistyped declaration is refused, not planned."""
    with pytest.raises(ValueError, match="schema"):
        _validate({"schema": {"grid": "E4M3", "q256": 1024}})


def test_a_plan_must_be_an_object():
    with pytest.raises(ValueError, match="object mapping tensor or stack names"):
        _validate([_plan()[TENSOR]])


@pytest.mark.parametrize(
    "entry, mutate, needle",
    [
        # An entry that is neither string nor object.
        (TENSOR, lambda p: p.__setitem__(TENSOR, ["E4M3", 1024]), TENSOR),
        # Missing owned fields.
        (TENSOR, lambda p: p[TENSOR].pop("grid"), "grid"),
        (TENSOR, lambda p: p[TENSOR].pop("q256"), "q256"),
        # A field the schema does not define -- a typo, not a sidecar.
        (TENSOR, lambda p: p[TENSOR].__setitem__("grdi", "E4M3"), "grdi"),
        # Values of the wrong type.
        (TENSOR, lambda p: p[TENSOR].__setitem__("q256", "1024"), "q256"),
        (TENSOR, lambda p: p[TENSOR].__setitem__("q256", 1024.0), "q256"),
        (TENSOR, lambda p: p[TENSOR].__setitem__("grid", "E5M3"), "E5M3"),
        # source_layout is a stack field: closed vocabulary, stacks only.
        (STACK, lambda p: p[STACK].__setitem__("source_layout", "diagonal"),
         "diagonal"),
        (TENSOR, lambda p: p[TENSOR].__setitem__("source_layout", "in_first_interleaved"),
         TENSOR),
    ])
def test_refusals_name_the_entry(entry, mutate, needle):
    plan = _plan()
    mutate(plan)
    with pytest.raises(ValueError, match=re.escape(needle)) as caught:
        _validate(plan)
    # The match above proves the needle; this proves the ENTRY (#691 item
    # 7): the old ``or needle in str(...)`` disjunct was already proven by
    # the match, so a refusal naming the wrong tensor still passed.
    assert entry in str(caught.value), (
        f"refusal names no entry: {caught.value}")


def test_module_scheme_key_is_importable_from_the_package():
    from tessera.alphabet import E4M3_GRID
    from tessera.export import served_recipe
    from tessera.serving.scheme import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE
    from tessera.serving_plan import family_for, module_scheme_key

    grid, q256 = E4M3_GRID, 1024
    recipe = served_recipe(grid, q256, STRUCTURE_DENSE)
    assert module_scheme_key(grid, q256) == (
        family_for(grid), grid.name, recipe.body.name, recipe.scale_plane.name)
    assert family_for(grid) == "TESSERA_FP8"


def test_module_scheme_key_separates_structures_through_the_served_recipe():
    """The routed span-2 promotion is part of the key, not just of the decode.

    A sub-cap E2M1x2 rung keeps the WINDOW body when served dense but is
    promoted to TCQ when served routed, so one (grid, q256) names two served
    wires and the key separates them.  At the cap (q256 896) both structures
    already decode TCQ, which is why the sub-cap rung is the one that pins
    this.
    """
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.serving.scheme import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE
    from tessera.serving_plan import module_scheme_key

    grid, q256 = tuple_grid(E2M1_GRID, 2), 512
    assert module_scheme_key(grid, q256, STRUCTURE_DENSE) == (
        "TESSERA_NVFP4", "E2M1x2", "WINDOW", "LUT")
    assert module_scheme_key(grid, q256, STRUCTURE_ROUTED_MOE) == (
        "TESSERA_NVFP4", "E2M1x2", "TCQ", "LUT")


def _write_checkpoint(tmp_path: Path) -> Path:
    import torch

    src = tmp_path / "src"
    src.mkdir()
    generator = torch.Generator().manual_seed(0)
    tensor = torch.randn(32, 32, generator=generator).bfloat16()
    save_file({TENSOR: tensor.contiguous()}, str(src / "model.safetensors"),
              metadata={"format": "pt"})
    (src / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": 32, "moe_intermediate_size": 32},
    }))
    return src


def test_the_supported_entry_point_exports_a_fixture_on_cpu(tmp_path):
    """``python -m tessera.export_serving`` -- the entry point #687 promotes.

    The smallest checkpoint the exporter's own tests use (one 32x32 Linear
    under a GLM text config), exported on CPU from a schema-valid plan, with
    PYTHONPATH pointing at this checkout so the module resolves from the tree
    under test and not from an installed Tessera.
    """
    src = _write_checkpoint(tmp_path)
    entries = {TENSOR: {"grid": "E4M3", "q256": 1024}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"schema": "tessera.serving_plan.v1",
                                     **entries}))
    out = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, "-m", "tessera.export_serving", str(src), str(out),
         "--grid", "E4M3", "--q256", "1024", "--device", "cpu", "--no-verify",
         "--plan-json", str(plan_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"python -m tessera.export_serving exited {proc.returncode}.\n"
        f"stderr:\n{proc.stderr[-4000:]}")
    manifest_path = out / "tessera_serving_manifest.json"
    assert manifest_path.is_file(), sorted(p.name for p in out.iterdir())
    manifest = json.loads(manifest_path.read_text())
    assert TENSOR in json.dumps(manifest)
    # The declaration is popped before the entry loop and recorded beside
    # the published plan (#691 item 5).
    assert manifest["plan"] == entries
    assert manifest["plan_schema"] == "tessera.serving_plan.v1"


def test_the_legacy_shim_path_drives_a_real_export_on_cpu(tmp_path):
    """The old path still drives a real export (#691 item 2).

    The shim is a separate namespace that re-exports the supported module,
    so this runs the fixture through ``experiments/export_tessera_serving.py``
    by path -- the legacy script shape -- and checks the manifest it wrote.
    A plan with no declaration records no schema: honest absence, not a
    default.
    """
    src = _write_checkpoint(tmp_path)
    entries = {TENSOR: {"grid": "E4M3", "q256": 1024}}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(entries))
    out = tmp_path / "out"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "experiments" / "export_tessera_serving.py"),
         str(src), str(out), "--grid", "E4M3", "--q256", "1024",
         "--device", "cpu", "--no-verify", "--plan-json", str(plan_path)],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"the shim exited {proc.returncode}.\n"
        f"stderr:\n{proc.stderr[-4000:]}")
    manifest = json.loads((out / "tessera_serving_manifest.json").read_text())
    assert manifest["plan"] == entries
    assert manifest["plan_schema"] is None


FAKE_COMMIT = "f" * 40


def _write_fake_installation(target: Path) -> Path:
    """A synthetic installed layout: the tree's package plus a dist-info.

    ``target`` plays ``site-packages``: it holds ``tessera/`` with no
    checkout around it (no ``src/`` name, no ``experiments/`` sibling), and
    a dist-info whose ``direct_url.json`` records a git commit, which is
    what a non-editable pip install from git records.
    """
    import shutil

    target.mkdir()
    shutil.copytree(ROOT / "src" / "tessera", target / "tessera",
                      ignore=shutil.ignore_patterns("__pycache__"))
    # The dist-info must name the real distribution: ``_resolve_version``
    # (src/tessera/__init__.py) reads the version of ``DISTRIBUTION`` when no
    # pyproject sits beside the package, and refuses when it finds none.  A
    # made-up name ("tessera-fake") left nothing to read, so the driver died
    # in ``import tessera`` (#721).
    from tessera import DISTRIBUTION

    dist_info = target / f"{DISTRIBUTION.replace('-', '_')}-0.1.dist-info"
    dist_info.mkdir()
    (dist_info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {DISTRIBUTION}\nVersion: 0.1\n")
    (dist_info / "top_level.txt").write_text("tessera\n")
    (dist_info / "direct_url.json").write_text(json.dumps({
        "url": "https://example.invalid/tessera.git",
        "vcs_info": {"vcs": "git", "commit_id": FAKE_COMMIT}}))
    return target


def test_the_exporter_resolves_an_installed_layout(tmp_path):
    """#691 item 4: the entry point works from an installed wheel.

    From an install there is no checkout: two parents up from the module is
    ``lib/python3.x``, so the old code hashed that directory and stamped
    ``unknown``.  The driver below runs with ``PYTHONPATH`` pointing ONLY at
    a synthetic site-packages holding this tree's package (plus a commit in
    ``direct_url.json``), from a working directory outside any checkout, and
    checks the root derivation, the install-commit fallback and the
    installed ``export_identity`` walk.
    """
    fake = _write_fake_installation(tmp_path / "site-packages")
    work = tmp_path / "work"
    work.mkdir()
    program = (
        "import sys, json, pathlib\n"
        "import tessera\n"
        f"assert tessera.__file__.startswith({str(fake)!r}), tessera.__file__\n"
        "from tessera.export_serving import exporter_code_root, git_hash\n"
        "from tessera.serving_parts import export_identity\n"
        f"expected = pathlib.Path({str(fake)!r}).resolve()\n"
        "root = exporter_code_root(); assert root == expected, root\n"
        f"commit = git_hash(); assert commit == {FAKE_COMMIT!r}, commit\n"
        "import pathlib, torch\n"
        "from safetensors.torch import save_file\n"
        "srcdir = pathlib.Path('srcdir'); srcdir.mkdir()\n"
        "(srcdir / 'config.json').write_text('{}')\n"
        "save_file({'w': torch.zeros(4, 4)}, str(srcdir / 'model.safetensors'))\n"
        "identity = export_identity(srcdir, {}, "
        "'example/runtime@sha256:' + '0' * 64, root)\n"
        "assert len(identity['code_sha256']) == 64, identity\n"
        "print('INSTALLED_LAYOUT_OK')\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(fake)
    env.pop("TESSERA_GIT", None)
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=work, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, (
        f"installed-layout driver exited {proc.returncode}.\n"
        f"stdout:\n{proc.stdout[-2000:]}\n"
        f"stderr:\n{proc.stderr[-4000:]}")
    assert "INSTALLED_LAYOUT_OK" in proc.stdout


def test_no_prismaquant_import_under_src():
    """#599: the package never imports the producer it serves (#687 item 4)."""
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(?:from|import)\s+prismaquant\b", text, re.MULTILINE):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, offenders
