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


def decoded_weight(expert, family):
    """The role's decoded weight as the kernel multiplies it, fp32 ``[rows, cols]``.

    E4M3: the e4m3 byte's value, UNSCALED (the row scale is applied in the
    epilogue).  Value: the table value folded once to bf16 with the row scale,
    as the folded contract does.
    """
    states = expert.states.cuda()
    if family == "e4m3":
        byte = expert.unit.native[expert.unit.codes_of_state[states].long()]
        return byte.view(torch.float8_e4m3fn).float()
    values = expert.values.float().cuda()[states]
    return (values * expert.scale[:, None]).bfloat16().float()


def fp64_weight(expert, family):
    """The role's decoded weight in fp64 ``[rows, cols]``, row scale applied."""
    w = decoded_weight(expert, family).double()
    if family == "e4m3":
        return w * expert.scale.double()[:, None]
    return w


def dense_bound(family, a64, w64, k, s=1, *, weight=None):
    """``(r, bound)``: the fp64 reference ``(a64 @ w64.T) * weight`` and the
    per-element bound on ``|kernel - r|`` for one dense (or one route's) GEMM.

    ``a64`` is the scaled A operand in fp64 (``xq * a_scale`` for E4M3, the
    bf16 ``x`` for value), ``w64`` the scaled weight (:func:`fp64_weight`),
    ``k`` the reduction length, ``s`` the K split, ``weight`` an optional
    per-row routing weight (fp64, broadcastable) the epilogue multiplies in as
    one more round-to-nearest fp32 multiply.
    """
    r = a64 @ w64.t()
    sigma = a64.abs() @ w64.abs().t()
    n_mul = 2 if family == "e4m3" else 0
    if weight is not None:
        r = r * weight
        sigma = sigma * weight.abs()
        n_mul += 1
    e_acc = (gamma(k + s + 2, U_ACC) + gamma(k, U64)) * sigma
    e_pre = e_acc + (gamma(n_mul, U32) * (r.abs() + e_acc) if n_mul else 0.0)
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
    """The fixed-order fp32 sum of ``top_k`` bf16 routes, rounded once to bf16.

    ``r_routes`` are the routes' fp64 values (already weighted), ``e_routes``
    their per-element bounds (each already including its own bf16 rounding);
    returns ``(r, bound)`` for the token sum along ``top_k_dim``.
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
