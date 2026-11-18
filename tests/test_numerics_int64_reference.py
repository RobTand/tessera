"""Check the dense E4M3 contract against the unchanged object contract."""
from fractions import Fraction
import math

import pytest
import torch

from experiments.numerics_rows.common import Invalid
from experiments.numerics_rows.reference import FiniteBound, outward
from experiments.numerics_rows.row import Dense, dot_reference


@pytest.fixture
def e4m3_sum():
    def evaluate(values, axis):
        function = getattr(FiniteBound, "exact_sum_e4m3", None)
        assert callable(function), "E4M3 int64 exact reference is absent"
        return function(values, axis)
    return evaluate


def tensor_bytes(tensor):
    return tensor.contiguous().view(torch.uint8).numpy().tobytes()


def require_equal(function, values, axis):
    actual, expected = function(values, axis), FiniteBound.exact_sum(values, axis)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.device == expected.device
    assert tensor_bytes(actual.reshape(-1)) == tensor_bytes(expected.reshape(-1))


@pytest.mark.parametrize("width", (0, 1, 7, 1024, 4096, 12288))
def test_e4m3_random_products(e4m3_sum, width):
    alphabet = torch.arange(256, dtype=torch.uint8).view(torch.float8_e4m3fn).double()
    alphabet = alphabet[torch.isfinite(alphabet)]
    generator = torch.Generator().manual_seed(1717 + width)
    operands = alphabet[torch.randint(len(alphabet), (2, 3, width), generator=generator)]
    products = operands[0] * operands[1]
    require_equal(e4m3_sum, products, -1)
    require_equal(e4m3_sum, products.T, 0)


@pytest.mark.parametrize("sign_pattern", ("zero", "positive", "negative", "alternating", "subnormal"))
def test_e4m3_product_edges(e4m3_sum, sign_pattern):
    operands = torch.tensor([0.0, -0.0, 2**-9, -2**-9, 7 * 2**-9, -448.0, 448.0], dtype=torch.float64)
    products = (operands[:, None] * operands[None, :]).flatten()
    if sign_pattern == "zero":
        products.zero_()
    elif sign_pattern in ("positive", "negative", "alternating"):
        products.fill_(448.0**2 * (-1 if sign_pattern == "negative" else 1))
        if sign_pattern == "alternating":
            products[1::2].neg_()
    require_equal(e4m3_sum, products, 0)
    require_equal(e4m3_sum, products.reshape(7, 7).T, 1)


@pytest.mark.parametrize("sign", (-1, 1))
def test_e4m3_outward_rounding_boundaries(e4m3_sum, sign):
    # The first terms sum to 2^35. Its binary64 spacing is 2^-17.
    # The final term reaches the midpoint and both adjacent grid points.
    values = torch.full((4, 2**18 + 1), sign * 2.0**17, dtype=torch.float64)
    values[:, -1] = sign * torch.arange(4, dtype=torch.float64) * 2**-18
    require_equal(e4m3_sum, values, -1)
    expected = torch.tensor([outward(sign * (Fraction(2**35) + Fraction(delta, 2**18)))
                             for delta in range(4)], dtype=torch.float64)
    assert tensor_bytes(e4m3_sum(values, -1)) == tensor_bytes(expected)


@pytest.mark.parametrize("bad,reason", (
    (2**-19, "2\\^-18 grid"),
    (200704.0 + 2**-18, "finite terms"),
    (-200704.0 - 2**-18, "finite terms"),
    (math.inf, "finite terms"),
    (-math.inf, "finite terms"),
    (math.nan, "finite terms"),
))
def test_e4m3_refuses_one_invalid_term(e4m3_sum, bad, reason):
    values = torch.ones((2, 8), dtype=torch.float64)
    values[1, 3] = bad
    with pytest.raises(Invalid, match=reason):
        e4m3_sum(values, -1)


def test_e4m3_refuses_a_scaled_product(e4m3_sum):
    values = torch.tensor([[2**-9 * 2**-9, 1.0]], dtype=torch.float64)
    values[0, 0] *= 0.5
    with pytest.raises(Invalid, match="2\\^-18 grid"):
        e4m3_sum(values, 1)


@pytest.mark.parametrize("width", (2**27, 2**27 + 1))
def test_e4m3_refuses_width_before_reading_values(e4m3_sum, width):
    values = torch.empty((0, width), device="meta", dtype=torch.float64)
    with pytest.raises(Invalid, match="K < 2\\^27"):
        e4m3_sum(values, -1)


@pytest.mark.parametrize("family", ("e4m3", "e2m1", "value"))
@pytest.mark.parametrize("caller", ("dot", "dense"))
def test_dot_family_selects_one_exact_contract(e4m3_sum, monkeypatch, family, caller):
    e4m3_sum(torch.zeros(1, dtype=torch.float64), 0)
    name = "exact_sum_e4m3" if family == "e4m3" else "exact_sum"
    function = getattr(FiniteBound, name)
    calls = []

    def selected(values, axis):
        calls.append(values.clone())
        return function(values, axis)

    def forbidden(values, axis):
        raise AssertionError("dot called the wrong exact contract")

    monkeypatch.setattr(FiniteBound, name, staticmethod(selected))
    other = "exact_sum" if family == "e4m3" else "exact_sum_e4m3"
    monkeypatch.setattr(FiniteBound, other, staticmethod(forbidden))
    activation = torch.tensor([[2**-9, -448.0]], dtype=torch.float64)
    weight = torch.tensor([[2**-9, 448.0]], dtype=torch.float64)
    if caller == "dot":
        result, mask, bound = dot_reference(activation, weight, None, torch.tensor(1.0),
                                           family=family, with_bounds=True, exact=False)
    else:
        adapter = Dense.__new__(Dense)
        adapter.family, adapter.gs = family, 1.0
        adapter.roles = [("dense", weight, torch.tensor(1.0), None)]
        result, mask, bound = adapter.reference(activation, None, exact=False)
    assert len(calls) == 1
    assert torch.isfinite(result).all() and mask.all() and bound.held.all()
