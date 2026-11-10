"""S7 consumer regressions, independent of changes to the older probe tests."""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _retired_api_references(source):
    """Include Python programs embedded in the subprocess probes, not prose."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "require_host_class_scope":
            yield node.lineno
        elif isinstance(node, ast.Attribute) and node.attr == "require_host_class_scope":
            yield node.lineno
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value == "require_host_class_scope":
                yield node.lineno  # getattr/setattr-based compatibility is not a migration
            elif "require_host_class_scope" in node.value:
                # Multiline probe programs are indented inside their enclosing test.
                import textwrap
                try:
                    embedded = list(_retired_api_references(textwrap.dedent(node.value)))
                except SyntaxError:
                    continue
                if embedded:
                    yield node.lineno


def test_stageprev_consumers_do_not_reference_removed_host_scope_api():
    paths = sorted((ROOT / "experiments/t8r_speed").glob("stageprev_793_*.py"))
    paths += sorted((ROOT / "tests").glob("test_stageprev_793_*.py"))
    offenders = {
        str(path.relative_to(ROOT)): lines
        for path in paths if path.resolve() != Path(__file__).resolve()
        if (lines := list(_retired_api_references(path.read_text())))
    }
    assert offenders == {}, f"S7 removed require_host_class_scope; migrate its callers: {offenders}"


def test_published_s7_observer_attests_manifest_and_refuses_api_defects():
    from test_stageprev_793_prepare import _run_published_sdk

    _run_published_sdk('''
        import json
        import pytest
        import stageprev_793_claim_contract as contract

        manifest = json.loads((Path(sys.argv[1]) / "RUNTIME_VERSION.json").read_text())
        observed = contract.observe_current_claim_contract(manifest)
        assert observed["verified"] is True, observed
        assert observed["api_problems"] == [], observed
        assert observed["identity_verified"] is True, observed
        assert observed["chain_verified"] is True, observed
        assert observed["positive_reservation_refusal_verified"] is True, observed
        assert observed["generation"] == manifest["generation"], observed
        assert observed["observation_only"] is True, observed
        assert observed["claim_invoked"] is False, observed
        assert observed["denial_synthesized"] is False, observed

        for defect in ("absent_name", "incompatible_signature"):
            with pytest.MonkeyPatch.context() as patch:
                if defect == "absent_name":
                    patch.delattr(client.PoolQueue, "offers")
                else:
                    patch.setattr(client.PoolQueue, "offers",
                                  staticmethod(lambda self, *, max_age_s=120.0: []))
                refused = contract.observe_current_claim_contract(manifest)
                assert any("PoolQueue.offers" in p for p in refused["api_problems"]), refused
                assert refused["api_verified"] is False, refused
                assert refused["identity_verified"] is False, refused
                assert refused["verified"] is False, refused
                assert refused["claim_invoked"] is False, refused
                assert refused["denial_synthesized"] is False, refused
    ''')
