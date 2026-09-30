"""Derived error bounds for the window kernels' tests (tessera#693 follow-up 3).

Every limit here is derived from the dtypes and the operation counts of the
arithmetic under test; nothing is fitted.  The model (stated once, cited by
the tests that use it):

* The kernels sum ``K`` EXACT fp32 products -- bf16 x bf16 and e4m3 x e4m3
  (widened to f16) both are -- as ``S`` fp32 partials summed in a fixed order
  when K is split.  One fp32 ulp (``2^-23``) is charged per accumulation step
  so a truncating tensor-core adder is covered: with ``Sigma = sum_k |a_k
  w_k|`` and ``gamma(n, u) = n u / (1 - n u)``, ``E_acc = gamma(K + S + 2,
  2^-23) Sigma + gamma(K, 2^-53) Sigma`` -- the second term is the fp64
  reference's own dot.
* Round-to-nearest fp32 multiplies in the epilogue (``(acc * a_scale) *
  w_scale`` on E4M3, ``* routing_weight`` on a weighted route) each cost
  ``2^-24`` relative: ``gamma(n_mul, 2^-24) (|r| + E_acc)``.
* One bf16 rounding costs half a bf16 ulp, taken at ``|r| + E_pre`` -- the
  largest magnitude the pre-rounding value can have -- so a straddled binade
  cannot undershoot it.
* Two lanes computing the same function (fused vs Triton, fused vs the
  compact adapter) each sit inside the bound, so they are within TWICE it of
  each other.

The routed reduction adds: per route the bound above (S = 1, one more
multiply for the routing weight, one bf16 rounding), then a fixed-order fp32
sum of ``top_k`` bf16 routes (``gamma(top_k, 2^-24)`` of the absolute sum)
and one more bf16 rounding.

Three more facts of the arithmetic, each read off the kernel it states:

* The BF16 value family has two weight contracts.  FOLDED (the fused lane,
  and the Triton lanes under ``arithmetic="folded"``): the weight is
  ``bf16(fp32(value * row_scale))``, exact in the reference because the
  reference decodes it the same way, and no epilogue multiply.  EPILOGUE (the
  grouped Triton GEMM's default): the weight is the bf16 table value and the
  row scale multiplies the fp32 accumulator -- one more round-to-nearest fp32
  multiply, the reference weight being ``value * row_scale`` in fp64 (exact:
  8 x 24 significant bits).
* The Triton GEMMs accumulate ``acc += tl.dot(x_chunk, w_chunk)`` over
  ``BK``-column chunks of every run.  Fused into the accumulator, a product
  passes through at most the ``K - 1`` adds of the products after it; summed
  from zero and then added, through at most the other real products of its
  own chunk plus one add per later chunk -- again at most ``K``, since every
  chunk holds at least one real column (masked columns add exact zeros).
  Either way its depth is at most ``K``, so the ``S = 1`` charge
  ``gamma(K + 3, 2^-23)`` covers them.
* The grouped Triton reduction (``preserve=False``) does NOT round a route
  to bf16 unless ``round_routes=True``: it adds each weighted fp32 route into
  a zeroed fp32 output with ``tl.atomic_add`` (round-to-nearest, in whatever
  order the programs reach it) and rounds once to bf16.  Its per-route bound
  is then :func:`dense_bound` with ``rounded=False`` (no half bf16 ulp), and
  :func:`route_sum_bound`'s ``gamma(top_k, 2^-24)`` holds for ANY summation
  order (``top_k - 1`` rounded adds suffice for any order).
"""

import torch

U_ACC = 2.0 ** -23          # one fp32 ulp per accumulation step (a truncating adder)
U32 = 2.0 ** -24            # fp32 unit roundoff (round-to-nearest multiply / add)
U64 = 2.0 ** -53            # fp64 unit roundoff (the reference's own dot)


def gamma(n, u):
    """The classical Higham ``gamma_n``: ``n u / (1 - n u)``."""
    return n * u / (1.0 - n * u)


def bf16_ulp(v):
    """The bf16 ulp at magnitude ``v`` (fp64, >= 0): ``2^(e - 7)`` for ``v`` in
    ``[2^e, 2^(e+1))``, floored at the smallest normal's ulp."""
    return torch.exp2(torch.floor(torch.log2(v.clamp(min=2.0 ** -126))) - 7)


def decoded_weight(expert, family, *, folded=True):
    """The role's decoded weight as the kernel multiplies it, fp32 ``[rows, cols]``.

    E4M3: the e4m3 byte's value, UNSCALED (the row scale is applied in the
    epilogue).  Value, ``folded``: the table value folded once to bf16 with
    the row scale, as the folded contract does.  Value, not ``folded`` (the
    epilogue contract): the bf16 table value, UNSCALED.
    """
    states = expert.states.cuda()
    if family == "e4m3":
        byte = expert.unit.native[expert.unit.codes_of_state[states].long()]
        return byte.view(torch.float8_e4m3fn).float()
    values = expert.values.float().cuda()[states]
    if not folded:
        return values
    return (values * expert.scale[:, None]).bfloat16().float()


def fp64_weight(expert, family, *, folded=True):
    """The role's decoded weight in fp64 ``[rows, cols]``, row scale applied."""
    w = decoded_weight(expert, family, folded=folded).double()
    if family == "e4m3" or not folded:
        return w * expert.scale.double()[:, None]
    return w


def epilogue_multiplies(family, *, folded=True):
    """The round-to-nearest fp32 multiplies the epilogue applies before any
    routing weight: ``(acc * a_scale) * w_scale`` on E4M3, ``acc * w_scale``
    on the value family's epilogue contract, none when the scale is folded."""
    if family == "e4m3":
        return 2
    return 0 if folded else 1


def dense_bound(family, a64, w64, k, s=1, *, weight=None, folded=True, rounded=True):
    """``(r, bound)``: the fp64 reference ``(a64 @ w64.T) * weight`` and the
    per-element bound on ``|kernel - r|`` for one dense (or one route's) GEMM.

    ``a64`` is the scaled A operand in fp64 (``xq * a_scale`` for E4M3, the
    bf16 ``x`` for value), ``w64`` the scaled weight (:func:`fp64_weight`,
    with the same ``folded``), ``k`` the reduction length, ``s`` the K split,
    ``weight`` an optional per-row routing weight (fp64, broadcastable) the
    epilogue multiplies in as one more round-to-nearest fp32 multiply.
    ``folded=False`` is the value family's epilogue contract (one multiply by
    the row scale, :func:`epilogue_multiplies`).  ``rounded=False`` returns
    the bound on the fp32 value BEFORE any bf16 rounding (a route the grouped
    reduction sums unrounded).
    """
    r = a64 @ w64.t()
    sigma = a64.abs() @ w64.abs().t()
    n_mul = epilogue_multiplies(family, folded=folded)
    if weight is not None:
        r = r * weight
        sigma = sigma * weight.abs()
        n_mul += 1
    e_acc = (gamma(k + s + 2, U_ACC) + gamma(k, U64)) * sigma
    e_pre = e_acc + (gamma(n_mul, U32) * (r.abs() + e_acc) if n_mul else 0.0)
    if not rounded:
        return r, e_pre
    return r, e_pre + 0.5 * bf16_ulp(r.abs() + e_pre)


def one_hot_inputs(family, cols, quant):
    """``x = I`` (one hot per row), the family's A operand and scale, and the
    fp32 value each hot element carries into the kernel's product (448 for
    E4M3: 1.0 quantised at scale 1/448; 1.0 for value).  ``quant`` is the
    route's own per-token E4M3 quantiser."""
    x = torch.eye(cols, device="cuda").bfloat16()
    if family == "e4m3":
        xq, a = quant(x)
        return x, xq.contiguous(), a.reshape(-1).contiguous().float(), xq.float().diagonal()
    return x, x.contiguous(), None, x.float().diagonal()


def one_hot_expected(expert, family, hot, a, *, weight=None):
    """The kernel's own arithmetic on one-hot rows, elementwise in torch.

    Row ``k`` of ``x`` selects column ``k`` of the weight: every accumulator
    is ONE exact fp32 product (``hot_k * w[n, k]``: 3 x 4 significant bits on
    E4M3, a bf16 on value) plus zeros, so the E4M3 epilogue ``(acc * a_scale)
    * w_scale`` (then ``* weight`` on a weighted route) is a chain of
    round-to-nearest fp32 multiplies torch reproduces bitwise, and the value
    family's folded ``bf16(table * scale)`` comes back as itself.  Output
    ``[cols, rows]``: ``y[k, n]``."""
    w = decoded_weight(expert, family)                      # [rows, cols] fp32
    acc = w.t() * hot[:, None]                              # exact
    if family == "e4m3":
        acc = (acc * a[:, None]) * expert.scale[None, :]
    if weight is not None:
        acc = acc * weight
    return acc.bfloat16()


def route_sum_bound(r_routes, e_routes, top_k_dim=1):
    """The fp32 sum of ``top_k`` routes, rounded once to bf16.

    ``r_routes`` are the routes' fp64 values (already weighted), ``e_routes``
    their per-element bounds -- each including its own bf16 rounding when the
    kernel rounds a route before the sum (the fused lane's fixed-order
    ``token_sum``, the grouped GEMM's ``round_routes=True``), or not
    (``dense_bound(..., rounded=False)``: the grouped GEMM's fp32 atomics).
    ``gamma(top_k, 2^-24)`` of the absolute sum covers ANY order of the
    ``top_k - 1`` rounded adds.  Returns ``(r, bound)`` for the token sum
    along ``top_k_dim``.
    """
    top_k = int(r_routes.shape[top_k_dim])
    r = r_routes.sum(top_k_dim)
    e_in = e_routes.sum(top_k_dim)
    abs_sum = (r_routes.abs() + e_routes).sum(top_k_dim)
    e_pre = e_in + gamma(top_k, U32) * abs_sum
    return r, e_pre + 0.5 * bf16_ulp(r.abs() + e_pre)


def check_within(out, r, bound, what, *, scale=1.0):
    """Assert ``|out - r| <= scale * bound`` per element; returns the worst ratio."""
    d = (out.double() - r).abs()
    ratio = float((d / (scale * bound)).max())
    n_bad = int((d > scale * bound).sum())
    assert n_bad == 0, f"{what}: {n_bad} element(s) outside the derived bound (worst ratio {ratio:.4f})"
    return ratio
