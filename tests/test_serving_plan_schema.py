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

The producer sidecar (``prismaquant_charged_bits`` /
``prismaquant_charged_bits_exact``) is not parsed and never was: the schema
admits those two field names as producer annotations the exporter copies
through verbatim (#687 item 4).  Tessera names the fields a plan may carry, not
the producer's modules -- nothing here imports or reads PrismaQuant (#599).
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


def test_the_schema_accepts_the_producer_sidecar_fields():
    annotated = _plan()
    annotated[TENSOR]["prismaquant_charged_bits"] = 4096.0
    annotated[TENSOR]["prismaquant_charged_bits_exact"] = [4096, 1]
    annotated[STACK]["prismaquant_charged_bits"] = 1048576.0
    _validate(annotated)


def test_a_plan_must_be_an_object():
    with pytest.raises(ValueError, match="object mapping tensor or stack names"):
        _validate([_plan()[TENSOR]])


@pytest.mark.parametrize(
    "mutate, needle",
    [
        # An entry that is neither string nor object.
        (lambda p: p.__setitem__(TENSOR, ["E4M3", 1024]), TENSOR),
        # Missing owned fields.
        (lambda p: p[TENSOR].pop("grid"), "grid"),
        (lambda p: p[TENSOR].pop("q256"), "q256"),
        # A field the schema does not define -- a typo, not a sidecar.
        (lambda p: p[TENSOR].__setitem__("grdi", "E4M3"), "grdi"),
        # Values of the wrong type.
        (lambda p: p[TENSOR].__setitem__("q256", "1024"), "q256"),
        (lambda p: p[TENSOR].__setitem__("q256", 1024.0), "q256"),
        (lambda p: p[TENSOR].__setitem__("grid", "E5M3"), "E5M3"),
        # source_layout is a stack field: closed vocabulary, stacks only.
        (lambda p: p[STACK].__setitem__("source_layout", "diagonal"), "diagonal"),
        (lambda p: p[TENSOR].__setitem__("source_layout", "in_first_interleaved"),
         TENSOR),
    ])
def test_refusals_name_the_entry(mutate, needle):
    plan = _plan()
    mutate(plan)
    with pytest.raises(ValueError, match=re.escape(needle)) as caught:
        _validate(plan)
    assert TENSOR in str(caught.value) or needle in str(caught.value)


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
    """The routed span-2 promotion is part of the key, not just of the decode."""
    from tessera.alphabet import E2M1_GRID, tuple_grid
    from tessera.serving.scheme import STRUCTURE_DENSE, STRUCTURE_ROUTED_MOE
    from tessera.serving_plan import module_scheme_key

    grid, q256 = tuple_grid(E2M1_GRID, 2), 896
    assert (module_scheme_key(grid, q256, STRUCTURE_ROUTED_MOE)
            != module_scheme_key(grid, q256, STRUCTURE_DENSE))


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
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({TENSOR: {"grid": "E4M3", "q256": 1024}}))
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


def test_no_prismaquant_import_under_src():
    """#599: the package never imports the producer it serves (#687 item 4)."""
    offenders = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(?:from|import)\s+prismaquant\b", text, re.MULTILINE):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, offenders
