"""Census metadata must match the real dense dispatch in both compile modes.

Execute the route's actual Python forward with a CPU prepared-bundle double.
This proves the dispatch/metadata seam, not a GPU kernel or graph replay.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tessera.serving import bf16_route, fp8_gemv, fp8_route
from tessera.serving.scheme import EAGER_ONLY_LAUNCHES


def forward(module):
    path = Path(module.__file__)
    tree = ast.parse(path.read_text())
    methods = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "apply"]
    assert len(methods) == 1
    namespace = dict(vars(module))
    records = []
    namespace.update(prefix="fixture", emit_route=lambda layer, **record: records.append(record))
    # The FP8 route quantizes before dispatch; arithmetic is outside this test.
    namespace["native_ops"] = SimpleNamespace(native_fp8_quant=lambda x: (x, None))
    exec(compile(ast.Module(body=[*methods], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["apply"], records


@pytest.mark.parametrize("route, census", [(fp8_route, fp8_gemv), (bf16_route, bf16_route)])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("mode", ["resident", "streamed"])
def test_census_is_exactly_the_pairs_emitted_by_dense_forward(route, census, compiled, mode):
    apply, records = forward(route)
    observed = {"decode": set(), "batch": set()}
    for m, regime in [(1, "decode"), (3, "batch")]:
        for pair in route.DENSE_LAUNCHES:
            if compiled and pair in EAGER_ONLY_LAUNCHES:
                continue    # its owner refuses under compile (contract v56)
            layer = SimpleNamespace(
                tessera_native=SimpleNamespace(
                    launch_pair=pair, decoded=None,
                    apply=lambda *args: torch.zeros(m, 2, dtype=torch.bfloat16)),
                tessera_rows=2, tessera_columns=4, tessera_mode=mode,
                tessera_activation_contract=route.ACTIVATION_CONTRACT,
            )
            output = apply(None, layer, torch.zeros(m, 4, dtype=torch.bfloat16))
            assert output.shape == (m, 2)
            record = records.pop()
            assert record["state"] == "served"
            observed[regime].add((record["symbol"], record["decoder"]))
    assert census.census_expected(compiled=compiled) == observed
