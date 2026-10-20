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


def _consumer_signature(owner, name):
    """Return the signature of one required call bound as the proof makes it.

    The check binds through a blank owner instance, so the descriptor
    protocol removes the receiver whatever name it uses. ``staticmethod``
    and ``classmethod`` bind the same way, and a receiver named anything
    other than ``self`` binds too. Construction runs no owner code beyond
    blank instance creation, so the check performs no claim or read.
    A raw signature means no blank instance binds the attribute; the
    caller then refuses it. ``None`` means the attribute has no call
    signature at all.
    """
    try:
        raw = inspect.getattr_static(owner, name)
    except AttributeError:
        return None
    try:
        probe = object.__new__(owner)
    except Exception:
        probe = None
    if probe is not None:
        try:
            return inspect.signature(getattr(probe, name)), False
        except (AttributeError, TypeError, ValueError):
            pass
    try:
        return inspect.signature(raw), True
    except (TypeError, ValueError):
        return None


def _consumer_call_binds(entry, args, kwargs):
    """Check one exact proof consumer call binds with no missing argument.

    A raw entry keeps the receiver in the signature, so an exact call
    that needs the receiver still refuses. Extra required arguments and
    keyword collisions then refuse, while extra optional arguments pass.
    """
    if entry is None:
        return False
    signature, raw = entry
    if raw:
        return False
    try:
        signature.bind(*args, **kwargs)
    except TypeError:
        return False
    return True


def required_api_problems(client_module, ledger_cls=None):
    """Name required public-claim API entries the loaded client misses.

    The proof reads the ledger through exactly these consumer calls:
    ``PoolQueue.ledger(host)``, ``capacity_census()``, ``available()``,
    ``latest_denials(keys, include_local=False)`` and
    ``offers(max_age_s=...)``. It also inspects the
    ``claim``/``_claim``/``_claim_pass`` source chain. Each returned entry
    names a missing call or a signature that refuses the exact call the
    proof makes. An empty list means the loaded client serves the contract
    the proof uses, at any version.
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
    ledger_entry = _consumer_signature(queue_cls, "ledger") if "ledger" not in missing_queue else None
    if "ledger" not in missing_queue and ledger_entry is None:
        problems.append("PoolQueue.ledger(signature)")
    elif ledger_entry is not None and not _consumer_call_binds(
            ledger_entry, ("sparky",), {}):
        problems.append("PoolQueue.ledger(host)")
    denials_entry = _consumer_signature(queue_cls, "latest_denials") if "latest_denials" not in missing_queue else None
    if "latest_denials" not in missing_queue and denials_entry is None:
        problems.append("PoolQueue.latest_denials(signature)")
    elif denials_entry is not None and not _consumer_call_binds(
            denials_entry, (set(),), {"include_local": False}):
        problems.append("PoolQueue.latest_denials(keys,include_local)")
    offers_entry = _consumer_signature(queue_cls, "offers") if "offers" not in missing_queue else None
    if "offers" not in missing_queue and offers_entry is None:
        problems.append("PoolQueue.offers(signature)")
    elif offers_entry is not None and not _consumer_call_binds(
            offers_entry, (), {"max_age_s": 120.0}):
        problems.append("PoolQueue.offers(max_age_s)")
    if ledger_cls is not None:
        for name in REQUIRED_LEDGER_METHODS:
            if name in missing_ledger:
                continue
            census_entry = _consumer_signature(ledger_cls, name)
            if census_entry is None:
                problems.append("ResourceLedger." + name + "(signature)")
            elif not _consumer_call_binds(census_entry, (), {}):
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


def _same_ast(left, right):
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def _statements(source):
    return ast.parse(textwrap.dedent(source)).body


def delegated_positive_kind_refusal(decision_source, preparation_source, predicate_source):
    """Recognize the inspected delegation without executing any scheduler code.

    Deliberately narrow: unrecognized rewrites refuse rather than inferring a
    denial from a helper name. The predicate, demand provenance, total-capacity
    argument, returned reason/evidence and both recording paths must agree.
    """
    predicate = _statements(predicate_source)
    expected_return = _statements('''
        return any(total.get(kind, 0) < need for kind, need in reservation_demand.items())
    ''')[0]
    if (len(predicate) != 1 or not isinstance(predicate[0], ast.FunctionDef)
            or [arg.arg for arg in predicate[0].args.args] != ['reservation_demand', 'total']
            or predicate[0].args.posonlyargs or predicate[0].args.kwonlyargs
            or predicate[0].args.vararg or predicate[0].args.kwarg
            or predicate[0].decorator_list
            or len(predicate[0].body) != 1
            or not _same_ast(predicate[0].body[0], expected_return)):
        return False
    preparation = _statements(preparation_source)[0]
    if not isinstance(preparation, ast.FunctionDef):
        return False
    expected_prefix = _statements('''
        item = candidate.item
        demand = candidate.demand.host
        reservation_demand = candidate.demand.reservation
        sealed_host_demand = candidate.demand.sealed_host
        scratch_intent = candidate.scratch_intent
        allowance = None
        dependent_owner = candidate.dependent_owner
        scratch_evidence = scratch_boundary = None
        if candidate_policy.never_fits(reservation_demand, total):
            return candidate_policy.CandidateDenial('never_fits_capacity', {
                'capacity_total': total, 'demand': demand,
                'reservation_demand': reservation_demand}, scan)
    ''')
    # Recognize the whole entry-to-refusal path, not disconnected statements.
    # Only the inspected field bindings may precede the predicate: an earlier
    # exit could bypass it, and arbitrary calls or assignments could mutate
    # either input (including through aliases) without a reservation name store.
    if (len(preparation.body) < len(expected_prefix)
            or not all(_same_ast(actual, expected) for actual, expected
                       in zip(preparation.body, expected_prefix))):
        return False
    expected_denial = _statements('''
        if isinstance(prepared, candidate_policy.CandidateDenial):
            order = prepared.order_before
            if prepared.evidence is None:
                self.record_denial(item, prepared.reason)
            else:
                self.record_denial(item, prepared.reason, prepared.evidence)
            continue
    ''')[0]
    tree = ast.parse(textwrap.dedent(decision_source))
    # The call and denial handling must be consecutive in the same block.
    for parent in ast.walk(tree):
        for field in ('body', 'orelse', 'finalbody'):
            block = getattr(parent, field, [])
            for call_statement, denial in zip(block, block[1:]):
                if not _same_ast(denial, expected_denial):
                    continue
                if (not isinstance(call_statement, ast.Assign)
                        or len(call_statement.targets) != 1
                        or not _same_ast(call_statement.targets[0], ast.Name(id='prepared', ctx=ast.Store()))):
                    continue
                call = call_statement.value
                if (isinstance(call, ast.Call)
                        and _same_ast(call.func, ast.parse('self._prepare_host_claim', mode='eval').body)
                        and len(call.args) == 2
                        and all(k.arg is not None for k in call.keywords)
                        and all(_same_ast(a, b) for a, b in zip(call.args, ast.parse('qualified, order', mode='eval').body.elts))
                        and any(k.arg == 'total' and _same_ast(k.value, ast.parse('scan_pass.total', mode='eval').body)
                                for k in call.keywords)):
                    return True
    return False


def _observe_delegation(queue_cls, decision_source, runtime_manifest, root):
    """Bind helper source to the loaded functions and their publication files."""
    receipts = {}
    try:
        preparation = queue_cls._prepare_host_claim
        policy = preparation.__globals__['candidate_policy']
        if queue_cls._claim_pass.__globals__['candidate_policy'] is not policy:
            return False, False, receipts
        helpers = (preparation, policy.never_fits, policy.CandidateDenial)
        paths = [Path(inspect.getsourcefile(helper)).resolve(strict=True) for helper in helpers]
        expected_paths = [root / 'src/prismabuild/pool.py',
                          root / 'src/prismabuild/candidate_policy.py',
                          root / 'src/prismabuild/candidate_policy.py']
        before = {path: path.read_bytes() for path in paths}
        sources = [inspect.getsource(helper) for helper in helpers]
        after = {path: path.read_bytes() for path in paths}
        identity = paths == expected_paths and before == after
        for path in paths:
            key = path.relative_to(root).as_posix()
            digest = hashlib.sha256(after[path]).hexdigest()
            receipts[key] = digest
            identity = identity and digest == runtime_manifest.get('files', {}).get(key)
        identity = identity and all(helper.__module__ == module for helper, module in zip(
            helpers, ('prismabuild.pool', 'prismabuild.candidate_policy', 'prismabuild.candidate_policy')))
        # CandidateDenial must carry the returned reason and evidence unchanged.
        denial_tree = _statements(sources[2])[0]
        fields = [node.target.id for node in denial_tree.body
                  if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)]
        decorator = ast.parse('dataclass(frozen=True)', mode='eval').body
        denial_valid = (fields[:3] == ['reason', 'evidence', 'order_before']
                        and len(denial_tree.decorator_list) == 1
                        and _same_ast(denial_tree.decorator_list[0], decorator)
                        and not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    for node in denial_tree.body))
        semantic = denial_valid and delegated_positive_kind_refusal(decision_source, sources[0], sources[1])
        return bool(identity), semantic, receipts
    except (AttributeError, KeyError, TypeError, ValueError, OSError, SyntaxError):
        return False, False, receipts


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
    helper_identity = True
    helper_receipts = {}
    if not predicate_verified:
        helper_identity, predicate_verified, helper_receipts = _observe_delegation(
            PoolQueue, decision_source, runtime_manifest, Path(PB_ROOT).resolve(strict=True))
    api_problems = required_api_problems(client)
    api_verified = api_problems == []
    identity_verified = (api_verified and helper_identity and source_file == expected and before == after
                         and digest == manifest_digest
                         and all(method.__module__ == "prismabuild.pool" for method in
                                 (PoolQueue.claim, PoolQueue._claim, PoolQueue._claim_pass)))
    return {"owner": OWNER, "chain": CHAIN, "source_file": str(source_file),
            "pool_sha256": digest, "published_pool_sha256": manifest_digest,
            "generation": runtime_manifest.get("generation"),
            "sdk_version": SDK_VERSION, "api_verified": api_verified, "api_problems": api_problems,
            "helper_source_sha256": helper_receipts,
            "identity_verified": identity_verified,
            "chain_verified": chain_verified,
            "positive_reservation_refusal_verified": predicate_verified,
            "verified": identity_verified and chain_verified and predicate_verified,
            "decision_source_sha256": hashlib.sha256(decision_source.encode()).hexdigest(),
            "predicate": "any(total.get(kind,0)<need for kind,need in reservation_demand.items()) ->never_fits_capacity +continue",
            "observation_only": True, "claim_invoked": False, "denial_synthesized": False}
