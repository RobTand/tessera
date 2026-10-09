"""Nondestructive current public-claim identity/semantics inspection.

No claim is attempted and no denial is synthesized. The inspected branch is
owned by the actual loaded PoolQueue and bound to its published source digest.
"""
import ast
import hashlib
import inspect
from pathlib import Path
import textwrap

OWNER = "prismabuild.client.PoolQueue.claim"
CHAIN = ["PoolQueue.claim", "PoolQueue._claim", "PoolQueue._claim_pass"]

REQUIRED_QUEUE_METHODS = ("claim", "_claim", "_claim_pass", "ledger", "latest_denials", "offers")
REQUIRED_LEDGER_METHODS = ("capacity_census", "available")


def _signature_of(owner, name):
    try:
        return inspect.signature(getattr(owner, name))
    except (TypeError, ValueError):
        return None


def required_api_problems(client_module, ledger_cls=None):
    """Name required public-claim API entries the loaded client misses.

    The proof reads the ledger through ``PoolQueue.ledger(host)``,
    ``capacity_census()``, ``available()``,
    ``latest_denials(keys, include_local=False)`` and
    ``offers(max_age_s=...)``, and inspects the
    ``claim``/``_claim``/``_claim_pass`` source chain. Each returned entry
    names a missing call or an incompatible signature. An empty list means
    the loaded client serves the contract the proof uses, at any version.
    """
    problems = []
    queue_cls = getattr(client_module, "PoolQueue", None)
    if queue_cls is None:
        return ["PoolQueue"]
    missing_queue = {name for name in REQUIRED_QUEUE_METHODS
                     if not callable(getattr(queue_cls, name, None))}
    problems.extend("PoolQueue." + name for name in REQUIRED_QUEUE_METHODS if name in missing_queue)
    if ledger_cls is None:
        try:
            from prismabuild.pool import ResourceLedger as ledger_cls
        except Exception:
            ledger_cls = None
    if ledger_cls is None:
        problems.append("ResourceLedger")
        missing_ledger = set(REQUIRED_LEDGER_METHODS)
    else:
        missing_ledger = {name for name in REQUIRED_LEDGER_METHODS
                          if not callable(getattr(ledger_cls, name, None))}
        problems.extend("ResourceLedger." + name for name in REQUIRED_LEDGER_METHODS if name in missing_ledger)
    ledger_signature = _signature_of(queue_cls, "ledger") if "ledger" not in missing_queue else None
    if "ledger" not in missing_queue and ledger_signature is None:
        problems.append("PoolQueue.ledger(signature)")
    elif ledger_signature is not None:
        host = ledger_signature.parameters.get("host")
        if host is None or host.kind not in (inspect.Parameter.POSITIONAL_ONLY,
                                             inspect.Parameter.POSITIONAL_OR_KEYWORD):
            problems.append("PoolQueue.ledger(host)")
    denials_signature = _signature_of(queue_cls, "latest_denials") if "latest_denials" not in missing_queue else None
    if "latest_denials" not in missing_queue and denials_signature is None:
        problems.append("PoolQueue.latest_denials(signature)")
    elif denials_signature is not None:
        entries = [(name, parameter) for name, parameter in denials_signature.parameters.items() if name != "self"]
        include_local = dict(entries).get("include_local")
        positional = [parameter for _, parameter in entries
                      if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                            inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        if (include_local is None or include_local.kind not in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                or not positional):
            problems.append("PoolQueue.latest_denials(keys,include_local)")
    offers_signature = _signature_of(queue_cls, "offers") if "offers" not in missing_queue else None
    if "offers" not in missing_queue and offers_signature is None:
        problems.append("PoolQueue.offers(signature)")
    elif offers_signature is not None:
        window = offers_signature.parameters.get("max_age_s")
        if window is None or window.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                                 inspect.Parameter.KEYWORD_ONLY):
            problems.append("PoolQueue.offers(max_age_s)")
    if ledger_cls is not None:
        for name in REQUIRED_LEDGER_METHODS:
            if name in missing_ledger:
                continue
            census_signature = _signature_of(ledger_cls, name)
            if census_signature is None:
                problems.append("ResourceLedger." + name + "(signature)")
                continue
            required = [parameter for parameter in list(census_signature.parameters.values())[1:]
                        if parameter.default is inspect.Parameter.empty and parameter.kind not in (
                            inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
            if required:
                problems.append("ResourceLedger." + name + "()")
    return problems


def _calls_self(tree, name):
    return any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
               and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
               and node.func.attr == name for node in ast.walk(tree))


def positive_kind_refusal_branch(source):
    """Check the real reservation predicate and fail-closed continuation together."""
    tree = ast.parse(textwrap.dedent(source))
    for branch in ast.walk(tree):
        if not isinstance(branch, ast.If):
            continue
        test = branch.test
        if (not isinstance(test, ast.Call) or not isinstance(test.func, ast.Name)
                or test.func.id != "any" or len(test.args) != 1
                or not isinstance(test.args[0], ast.GeneratorExp)):
            continue
        generator = test.args[0]
        if len(generator.generators) != 1:
            continue
        iteration = generator.generators[0]
        if (not isinstance(iteration.target, ast.Tuple)
                or len(iteration.target.elts) != 2
                or any(not isinstance(value, ast.Name) for value in iteration.target.elts)
                or not isinstance(iteration.iter, ast.Call)
                or not isinstance(iteration.iter.func, ast.Attribute)
                or iteration.iter.func.attr != "items"
                or not isinstance(iteration.iter.func.value, ast.Name)
                or iteration.iter.func.value.id != "reservation_demand"
                or iteration.ifs):
            continue
        kind, need = (value.id for value in iteration.target.elts)
        comparison = generator.elt
        if (not isinstance(comparison, ast.Compare) or len(comparison.ops) != 1
                or not isinstance(comparison.ops[0], ast.Lt)
                or len(comparison.comparators) != 1
                or not isinstance(comparison.comparators[0], ast.Name)
                or comparison.comparators[0].id != need):
            continue
        left = comparison.left
        if (not isinstance(left, ast.Call) or not isinstance(left.func, ast.Attribute)
                or left.func.attr != "get" or not isinstance(left.func.value, ast.Name)
                or left.func.value.id != "total" or len(left.args) != 2
                or not isinstance(left.args[0], ast.Name) or left.args[0].id != kind
                or not isinstance(left.args[1], ast.Constant) or left.args[1].value != 0):
            continue
        refused = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and isinstance(node.func.value, ast.Name) and node.func.value.id == "self"
                      and node.func.attr == "record_denial" and len(node.args) >= 2
                      and isinstance(node.args[1], ast.Constant)
                      and node.args[1].value == "never_fits_capacity"
                      for statement in branch.body for node in ast.walk(statement))
        if refused and any(isinstance(statement, ast.Continue) for statement in branch.body):
            return True
    return False


def observe_current_claim_contract(runtime_manifest):
    from prismabuild import client
    from prismabuild.client import PoolQueue, SDK_VERSION
    from stageprev_793_prepare import PB_ROOT
    expected = (Path(PB_ROOT) / "src/prismabuild/pool.py").resolve(strict=True)
    source_file = Path(inspect.getsourcefile(PoolQueue.claim)).resolve(strict=True)
    before = source_file.read_bytes()
    claim_source = inspect.getsource(PoolQueue.claim)
    wrapper_source = inspect.getsource(PoolQueue._claim)
    decision_source = inspect.getsource(PoolQueue._claim_pass)
    after = source_file.read_bytes()
    digest = hashlib.sha256(after).hexdigest()
    manifest_digest = runtime_manifest.get("files", {}).get("src/prismabuild/pool.py")
    chain_verified = (_calls_self(ast.parse(textwrap.dedent(claim_source)), "_claim")
                      and _calls_self(ast.parse(textwrap.dedent(wrapper_source)), "_claim_pass"))
    predicate_verified = positive_kind_refusal_branch(decision_source)
    api_problems = required_api_problems(client)
    api_verified = api_problems == []
    identity_verified = (api_verified and source_file == expected and before == after
                         and digest == manifest_digest
                         and all(method.__module__ == "prismabuild.pool" for method in
                                 (PoolQueue.claim, PoolQueue._claim, PoolQueue._claim_pass)))
    return {"owner": OWNER, "chain": CHAIN, "source_file": str(source_file),
            "pool_sha256": digest, "published_pool_sha256": manifest_digest,
            "generation": runtime_manifest.get("generation"),
            "sdk_version": SDK_VERSION, "api_verified": api_verified, "api_problems": api_problems,
            "identity_verified": identity_verified,
            "chain_verified": chain_verified,
            "positive_reservation_refusal_verified": predicate_verified,
            "verified": identity_verified and chain_verified and predicate_verified,
            "decision_source_sha256": hashlib.sha256(decision_source.encode()).hexdigest(),
            "predicate": "any(total.get(kind,0)<need for kind,need in reservation_demand.items()) ->never_fits_capacity +continue",
            "observation_only": True, "claim_invoked": False, "denial_synthesized": False}
