"""Exact native apply boundaries for an entire canonical engine census.

These observers preserve the installed implementation and its arguments. A
canonical routed group observes RoutedExperts.quant_method.apply after routing,
the same boundary used by the native whole-MoE receipt. Module forward is wider.
"""
from contextlib import contextmanager
import importlib
import inspect


def resolve_apply_boundaries(model, roster):
    modules = dict(model.named_modules())
    result, units, owners = [], set(), set()
    for row in roster:
        unit, name = row["unit_id"], row["module"]
        if unit in units or name not in modules:
            raise ValueError("duplicate or missing canonical timing unit: " + unit)
        units.add(unit)
        owner = modules[name]
        boundary_name = name
        if unit.startswith("s:"):
            owner = getattr(owner, "routed_experts", None)
            boundary_name += ".routed_experts"
            if owner is None:
                raise ValueError("canonical routed group has no RoutedExperts owner: " + unit)
        method = getattr(owner, "quant_method", None)
        apply = getattr(method, "apply", None)
        if not callable(apply) or (unit.startswith("s:") and getattr(method, "is_monolithic", None) is not False):
            raise ValueError("canonical timing requires a nonmonolithic native apply: " + unit)
        if id(owner) in owners:
            raise ValueError("canonical timing units alias one native owner")
        owners.add(id(owner))
        parameters = inspect.signature(apply).parameters
        if "layer" not in parameters:
            raise ValueError("native apply does not expose its layer argument: " + unit)
        result.append({"unit_id": unit, "module": name, "boundary": boundary_name + ".quant_method.apply",
                       "owner": owner, "method": method, "apply": apply,
                       "runner": modules[name] if unit.startswith("s:") else None,
                       "signature": inspect.signature(apply),
                       "includes_router": False if unit.startswith("s:") else None})
    if not result:
        raise ValueError("canonical timing census is empty")
    return result


@contextmanager
def observe_apply_boundaries(boundaries, observe):
    """Patch each method instance once; dispatch on the actual layer identity.

Methods may be shared between layers, so a bound method object is not a unit.
Unlisted owners sharing a method still execute their original implementation.
    """
    methods = {}
    for row in boundaries:
        methods.setdefault(id(row["method"]), []).append(row)
    restores = []
    try:
        for rows in methods.values():
            method = rows[0]["method"]
            original = rows[0]["apply"]
            signature = rows[0]["signature"]
            by_owner = {id(row["owner"]): row for row in rows}
            previous = vars(method).get("apply")
            had_override = "apply" in vars(method)

            def wrapped(*args, _original=original, _signature=signature,
                        _by_owner=by_owner, **kwargs):
                arguments = _signature.bind(*args, **kwargs).arguments
                owner = arguments["layer"]
                row = _by_owner.get(id(owner))
                if row is None:
                    return _original(*args, **kwargs)
                call = {"arguments": arguments, "result": None}
                with observe(row, call):
                    result = _original(*args, **kwargs)
                    call["result"] = result
                    return result

            method.apply = wrapped
            restores.append((method, had_override, previous))
        yield
    finally:
        for method, had_override, previous in reversed(restores):
            if had_override:
                method.apply = previous
            else:
                del method.apply


@contextmanager
def observe_tp2_owner_boundaries(boundaries, observe_apply, observe_reduce, observe_collective):
    """Observe the two disjoint spans of a TP routed native owner.

    The stock runner executes shared output/transform work between the routed
    apply and ``_maybe_reduce_final_output``.  That work belongs in measured
    fixed gaps, so wrapping its whole forward would overprice the native owner.
    The recorder checks one late-reduction call for each routed apply and the
    profile checks the actual CUDA stream and launches.
    """
    restores = []
    moe_runner = importlib.import_module("vllm.model_executor.layers.fused_moe.runner.moe_runner")
    collective_name = "tensor_model_parallel_all_reduce"
    collective = getattr(moe_runner, collective_name, None)
    if not callable(collective):
        raise ValueError("stock MoE runner no longer publishes its final all-reduce callsite")

    def wrapped_collective(states, *args, **kwargs):
        with observe_collective(states):
            return collective(states, *args, **kwargs)

    moe_runner.tensor_model_parallel_all_reduce = wrapped_collective
    try:
        with observe_apply_boundaries(boundaries, observe_apply):
            try:
                for row in boundaries:
                    runner = row["runner"]
                    if runner is None:
                        continue
                    original = getattr(runner, "_maybe_reduce_final_output", None)
                    if not callable(original):
                        raise ValueError("routed owner has no stock final-reduction boundary: " + row["unit_id"])
                    had_override = "_maybe_reduce_final_output" in vars(runner)
                    previous = vars(runner).get("_maybe_reduce_final_output")

                    def wrapped(*args, _original=original, _row=row, **kwargs):
                        call = {"arguments": args, "keywords": kwargs, "result": None}
                        with observe_reduce(_row, call):
                            result = _original(*args, **kwargs)
                            call["result"] = result
                            return result

                    runner._maybe_reduce_final_output = wrapped
                    restores.append((runner, had_override, previous))
                yield
            finally:
                for runner, had_override, previous in reversed(restores):
                    if had_override:
                        runner._maybe_reduce_final_output = previous
                    else:
                        del runner._maybe_reduce_final_output
    finally:
        moe_runner.tensor_model_parallel_all_reduce = collective
