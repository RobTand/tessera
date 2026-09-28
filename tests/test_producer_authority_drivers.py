"""Every export driver takes ``--producer-authority``, and the contract says which.

A Hessian reference document binds a producer's calibration cache, so since
tessera#599 step 2 it refuses by name unless its caller supplies the producer's
canonical capture.  A driver that opens one with no way to pass that pair
cannot do its job on a producer's capture.  So every driver on the export path
declares ``--producer-authority`` through ``tessera.producer_authority`` -- one
help text, one set of refusals -- and the packaged runtime contract publishes
the list (``producer_interface.reuse_authority.drivers``) for a producer to
read before it passes the option.

These tests hold three things:

- the published list is the list the tree declares, derived from the drivers'
  own source, so the contract states what the tree does;
- each driver, given a producer's reference capture, refuses by name without
  the option and accepts with it, through its own parser and its own opener;
- the option's refusals are one set of words, wherever it is taken.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import pytest

from tessera import producer_authority
from tessera.errors import GrammarError
from tessera.serving.contract import load_serving_contract
from test_hessian_reference_capture import build_reference

ROOT = Path(__file__).resolve().parents[1]

#: A producer's canonical capture, named as PrismaQuant names it: the pair a
#: reference binds and Tessera no longer knows.
CLIENT_CANONICAL_CAPTURE = ("prismaquant.tessera_calibration_cache.v2",
                            "tessera_campaign_prefix_f32_v1")

MISSING = r"need the producer canonical capture \(canonical_capture=\(schema, source\)\)"


def declares_the_option(source: str) -> bool:
    """Does this module declare the option through ``tessera.producer_authority``?"""
    tree = ast.parse(source)
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tessera" and node.level == 0:
            aliases |= {a.asname or a.name for a in node.names if a.name == "producer_authority"}
        elif isinstance(node, ast.Import):
            aliases |= {a.asname for a in node.names
                        if a.name == "tessera.producer_authority" and a.asname}
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and node.func.attr == "add_argument" and isinstance(node.func.value, ast.Name)
               and node.func.value.id in aliases
               for node in ast.walk(tree))


def declared_drivers() -> list[str]:
    found = []
    for top in ("experiments", "tools"):
        for path in sorted((ROOT / top).rglob("*.py")):
            if declares_the_option(path.read_text(encoding="utf-8")):
                found.append(path.relative_to(ROOT).as_posix())
    return sorted(found)


def test_the_contract_lists_exactly_the_drivers_that_declare_the_option():
    published = load_serving_contract()["producer_interface"]["reuse_authority"]
    assert published["option"] == producer_authority.OPTION
    assert published["drivers"] == declared_drivers()


def test_the_scanner_sees_both_import_spellings():
    assert declares_the_option(
        "from tessera import producer_authority\nproducer_authority.add_argument(p)\n")
    assert declares_the_option(
        "from tessera import producer_authority as opt\nopt.add_argument(p)\n")
    assert declares_the_option(
        "import tessera.producer_authority as opt\nopt.add_argument(p)\n")
    assert not declares_the_option("from tessera import producer_authority\n")
    assert not declares_the_option("p.add_argument('--producer-authority')\n")


def _module(relative: str):
    path = ROOT / relative
    name = "producer_authority_driver_" + path.stem
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _client_authority(tmp_path: Path) -> Path:
    """A producer authority file whose canonical capture is the client's pair."""
    path = tmp_path / "client_authority.py"
    path.write_text(
        "class ClientAuthority:\n"
        f"    canonical_hessian_capture = {CLIENT_CANONICAL_CAPTURE!r}\n"
        "    def check_document(self, role, document):\n"
        "        raise ValueError('not exercised')\n"
        "    def adoption_proof(self, unit, adoption, identity, original):\n"
        "        raise ValueError('not exercised')\n"
        "    def proof_authorizes(self, proof, adoption, original):\n"
        "        return False\n"
        "    def served_activations(self, policy, adoptions, units):\n"
        "        return {}\n"
        "PRODUCER_AUTHORITY = ClientAuthority()\n")
    return path


def _export_glm53(module, tmp_path, reference, extra):
    args = module.build_parser().parse_args(["--hessian", str(reference), *extra])
    return module.activation_source(args)


def _routed_owner_inputs(module, tmp_path, reference, extra):
    args = module.build_parser().parse_args(
        ["--mode", "source", "--export", str(tmp_path / "export"),
         "--bundle", str(tmp_path / "bundle.json"), "--out", str(tmp_path / "out"), *extra])
    return module.activation_source({"hessian": {"path": str(reference)}},
                                    args.producer_authority)


def _bf16_reach_roster(module, tmp_path, reference, extra):
    args = module.build_parser().parse_args(
        ["--out", str(tmp_path / "out.json"), "--rows-dir", str(tmp_path / "rows"),
         "--production", str(reference), *extra])
    return module.production_capture(args)


def _pack_probe(module, tmp_path, reference, extra):
    args = module.build_parser().parse_args(["--out", str(tmp_path / "out.json"), *extra])
    return module.activation_source(reference, args.producer_authority)


#: Each driver's own parser and its own capture opener.
DRIVERS = {
    "experiments/export_glm53_tessera.py": _export_glm53,
    "experiments/glm_routed_owner_inputs.py": _routed_owner_inputs,
    "experiments/bf16_reach_roster.py": _bf16_reach_roster,
    "tools/glm_cpu_cached_pack_probe.py": _pack_probe,
}


def test_every_listed_driver_but_the_exporter_is_exercised_here():
    # The exporter's reference and rooted-bundle intake is exercised in
    # test_rooted_cached_bundle.py and test_reuse_authority_boundary.py.
    assert set(DRIVERS) | {"experiments/export_tessera_serving.py"} == set(declared_drivers())


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_a_driver_refuses_a_producer_capture_by_name_without_the_option(tmp_path, driver):
    reference = build_reference(tmp_path, CLIENT_CANONICAL_CAPTURE)[0]
    with pytest.raises(GrammarError, match=MISSING):
        DRIVERS[driver](_module(driver), tmp_path, reference, [])


@pytest.mark.parametrize("driver", sorted(DRIVERS))
def test_a_driver_accepts_a_producer_capture_with_the_option(tmp_path, driver):
    reference = build_reference(tmp_path, CLIENT_CANONICAL_CAPTURE)[0]
    authority = _client_authority(tmp_path)
    activation = DRIVERS[driver](_module(driver), tmp_path, reference,
                                 [producer_authority.OPTION, str(authority)])
    try:
        assert set(activation.hessians) == {"a", "b"}
    finally:
        activation.hessians.close()


@pytest.mark.parametrize("driver", sorted(DRIVERS) + ["experiments/export_tessera_serving.py"])
def test_every_driver_declares_one_option_with_one_help(driver):
    module = _module(driver)
    parser = (module.build_parser() if hasattr(module, "build_parser") else None)
    if parser is None:
        # The exporter builds its parser inside main(); its declaration is the
        # shared call, which declares_the_option already found.
        assert declares_the_option((ROOT / driver).read_text(encoding="utf-8"))
        return
    action = next(a for a in parser._actions if producer_authority.OPTION in a.option_strings)
    assert action.help == producer_authority.HELP and action.default is None


def test_the_option_refuses_in_one_set_of_words(tmp_path):
    with pytest.raises(SystemExit, match="^--producer-authority must name an absolute regular file: "):
        producer_authority.load(Path("relative.py"))
    with pytest.raises(SystemExit, match="^--producer-authority must name an absolute regular file: "):
        producer_authority.load(tmp_path / "absent.py")
    empty = tmp_path / "empty.py"
    empty.write_text("PRODUCER_AUTHORITY = object()\n")
    with pytest.raises(SystemExit,
                       match="^--producer-authority .* defines no ReuseAuthority PRODUCER_AUTHORITY$"):
        producer_authority.load(empty)
    assert producer_authority.canonical_capture(None) is None
    assert producer_authority.canonical_capture(_client_authority(tmp_path)) == CLIENT_CANONICAL_CAPTURE


def test_the_exporter_delegates_to_the_shared_loader(tmp_path):
    exporter = _module("experiments/export_tessera_serving.py")
    authority, canonical = exporter.load_producer_authority(_client_authority(tmp_path))
    assert canonical == CLIENT_CANONICAL_CAPTURE
    with pytest.raises(SystemExit, match="^--producer-authority must name an absolute regular file: "):
        exporter.load_producer_authority(Path("relative.py"))


def test_the_block_is_validated_against_the_shared_constants():
    from tessera.serving.contract import validate_producer_interface
    block = json.loads(json.dumps(load_serving_contract()["producer_interface"]))
    validate_producer_interface(block, "producer_interface")
    for key, bad in (("option", "--authority"), ("attribute", "AUTHORITY"),
                     ("drivers", ["b.py", "a.py"]), ("drivers", []), ("drivers", ["/abs.py"])):
        broken = json.loads(json.dumps(block))
        broken["reuse_authority"][key] = bad
        with pytest.raises(ValueError, match=f"reuse_authority.{key}"):
            validate_producer_interface(broken, "producer_interface")
