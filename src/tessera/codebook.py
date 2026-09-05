"""Learned reconstruction grids: a codebook fitted to the tensor it encodes.

``tuple_grid(E2M1_GRID, 2)`` is a *tensor product* -- sixteen scalar levels
crossed with themselves -- so it spends codes on the corners of a square while
the weight density it has to cover is a round blob.  That shape costs more than
the trellis earns: measured at the rate cap, the trellis is worth 1.134x over
nearest-neighbour on the same grid, and replacing the product grid with a
codebook fitted to the data is worth 1.106x on top (33 tensors, full width, no
losses).  Below the cap it is worth far more -- 1.22x to 1.26x -- because there
the product grid is not merely the wrong shape, its *anchors* are chosen by
k-d bisection of a lattice rather than by where the data is.

Re-spacing the scalar ladder does not fix this and has been tried: a learned
sixteen-level E2M1 measured 1.006x and then inverted to 0.990x on replication.
It learned the levels and kept the cross.

**Why a tree and not free Lloyd.**  Tessera's rate axis is embedded: one deep
encode serves every lower rate, because ``build_forest`` lays anchors out in
contiguous dyadic blocks of the value order so that truncating completion bits
lands on an *ancestor*.  A flat Lloyd codebook has no ancestors, so adopting one
would buy 1.106x and pay the ladder for it.  Tree-structured VQ buys both: split
recursively to ``depth``, and the ``2^l`` nodes at level ``l`` are exactly the
anchor set a rate-``(l-1)`` trellis wants.

The construction is arranged so that the *existing* forest machinery is what
enforces the nesting, rather than a second mechanism that has to agree with it.
Leaves are emitted in tree order, so leaf ``i``'s ancestor at level ``l`` is
``i >> (depth - l)`` -- and a contiguous dyadic block of ``2^(depth-l)`` leaves
starting at ``k * 2^(depth-l)`` is precisely the set of leaves under node ``k``.
"Contiguous in the value order" and "under the same subtree" become the same
statement, so `build_forest` needs no special case beyond declining to k-d
bisect a grid that is already a tree.

TSVQ is *constrained* Lloyd -- a centroid must live inside its parent's cell --
so it cannot beat a free fit and does not claim to.  Measured, it keeps 76-80%
of the free gain at the rungs below the cap and 49% at the cap itself.

**Determinism.**  The codebook travels in the artifact, so a reader never
rebuilds it and no cross-device float agreement is required for correctness.
Fitting is nonetheless free of RNG -- splits seed from the principal axis, which
is a function of the data -- so the same tensor and depth reproduce the same
grid, which is what makes an encoder profile id meaningful.
"""

from __future__ import annotations

import torch

from .alphabet import TREE_PARTITION, PayloadGrid
from .errors import GrammarError

__all__ = ["learn_tree_codebook", "TREE_PARTITION"]

#: ``PayloadGrid.partition`` marker for a grid whose code order is a tree
#: traversal.  ``build_forest`` reads it to skip k-d bisection: the blocks it
#: would compute are already the code order.
TREE_PARTITION = "tree"


def _two_means(points: torch.Tensor, iterations: int = 12) -> torch.Tensor:
    """Split one cell in two.  Seeded on the principal axis, so no RNG.

    Distances are direct squared differences -- subtract first, then square.
    The GEMM expansion ``||x||^2 - 2<x,c> + ||c||^2`` (``torch.cdist``'s
    matmul path) subtracts large nearly-equal terms on uncentred data, and at
    float32 the small squared separations cancel to zero: a cloud of 64
    distinct values translated to 10000 fitted two centroids that tied at
    zero distance for half its points and erased that half (tessera#227).
    The direct form's error scales with the *separation*, not the magnitude,
    which is what an assignment rule needs.
    """
    mean = points.mean(0)
    centred = points - mean
    axis = torch.linalg.eigh((centred.T @ centred).double())[1][:, -1].float()
    spread = (centred @ axis).std().clamp(min=1e-12)
    centroids = torch.stack([mean - axis * spread, mean + axis * spread])
    for _ in range(iterations):
        assign = (points.unsqueeze(1) - centroids.unsqueeze(0)).square().sum(2).argmin(1)
        for side in (0, 1):
            if int((assign == side).sum()):
                centroids[side] = points[assign == side].mean(0)
    return centroids


def _hoist(
    points: torch.Tensor, assign: torch.Tensor, leaves: torch.Tensor, depth: int
) -> torch.Tensor:
    """Leaf order placing every node's representative first in its own block.

    Bottom-up: a leaf represents itself; an internal node is represented by
    whichever child's representative costs less **over that node's own points**,
    and that child's subtree is emitted first.  Scoring against the node's
    points rather than the child's is the whole point -- the representative is
    what a reader gets when it truncates to this level, so it is answering for
    its sibling's mass too.

    **The tie-break is left**, deliberately and not incidentally: ``<=`` keeps
    the left child's representative and emits its subtree first when the two
    costs are equal, which happens on any symmetric cell -- exactly the cells a
    principal-axis split produces most often.  A tie resolved by whichever
    float comparison happens to win, rather than by a stated rule, would make
    the leaf order and therefore the emitted grid depend on summation order.
    An empty cell takes the same branch for the same reason: no points, no
    preference, left.
    """
    def walk(level: int, node: int) -> "tuple[list[int], int]":
        if level == depth:
            return [node], node
        left, left_rep = walk(level + 1, 2 * node)
        right, right_rep = walk(level + 1, 2 * node + 1)
        mine = points[(assign >> (depth - level)) == node]
        if not len(mine):
            return left + right, left_rep
        cost = lambda rep: float(((mine - leaves[rep]) ** 2).sum())
        if cost(left_rep) <= cost(right_rep):
            return left + right, left_rep
        return right + left, right_rep

    order, _ = walk(0, 0)
    return torch.tensor(order, dtype=torch.long, device=leaves.device)


def learn_tree_codebook(
    samples: torch.Tensor,
    depth: int = 8,
    name: str = "TESSERA_TREE",
) -> PayloadGrid:
    """Fit a ``2^depth``-code grid to ``samples`` (``n x arity``).

    Returns a ``PayloadGrid`` whose codes are ordered by their path through the
    tree, which is what makes ``build_forest``'s dyadic blocks its subtrees.
    """
    if samples.ndim != 2:
        raise GrammarError(f"samples must be (n, arity), got {tuple(samples.shape)}")
    arity = samples.shape[1]
    if len(samples) < (1 << depth):
        raise GrammarError(
            f"{len(samples)} samples cannot fit {1 << depth} codes; a cell with "
            "no points has no centroid and the grid would carry a duplicate"
        )
    points = samples.float()
    level = points.mean(0, keepdim=True)
    assign = torch.zeros(len(points), dtype=torch.long, device=points.device)
    for _ in range(depth):
        nxt = torch.zeros(2 * len(level), arity, device=points.device)
        for node in range(len(level)):
            cell = points[assign == node]
            if len(cell) < 2:
                # A cell too small to split keeps its parent's value in both
                # children.  The duplicate is legal -- ties break to the lower
                # code -- and it is strictly better than refusing to build a
                # grid because one tail cell went empty.
                nxt[2 * node] = nxt[2 * node + 1] = level[node]
                continue
            nxt[2 * node:2 * node + 2] = _two_means(cell)
        # A point may only descend into its own parent's two children, which is
        # what keeps the tree a tree: re-assigning globally would let a point
        # cross cells and break the ancestor property this grid exists for.
        # Distances against each point's own pair only, by direct subtraction:
        # the all-node cdist both materialised an n x 2^level matrix nothing
        # read and cancelled on uncentred data (tessera#227, as in _two_means).
        # The tie stays with the left child, as before.
        left = (points - nxt[0::2][assign]).square().sum(1)
        right = (points - nxt[1::2][assign]).square().sum(1)
        assign = 2 * assign + (right < left).long()
        level = nxt

    # --- hoist each node's representative to its block's index 0 -----------
    # ``_order_block`` does this for scalar grids by recursive swap, scoring
    # candidates against samples.  It cannot run here: it reads
    # ``grid.values[code]`` as a scalar and a k-tuple code has no scalar.  But
    # the swap does not need to happen at forest-build time -- it needs to have
    # happened.  Doing it during the fit, where the points are still in hand, is
    # strictly better information than a forest builder would have, and it
    # leaves ``build_forest`` with nothing to decide: a block is a contiguous
    # slice whose first element is already its representative, at every level at
    # once, because a node's representative is its better child's.
    #
    # Swapping two subtrees keeps every block a contiguous dyadic run, so the
    # nesting property survives the reordering that establishes it.
    order = _hoist(points, assign, level, depth)
    level = level[order]

    values = tuple(level.reshape(-1).tolist())
    # ``keys`` drives ``value_order``, which sorts by (sum, vector, code).  One
    # entry per code makes that sort the identity, so the value order IS the
    # tree order -- the whole construction rests on this line.
    keys = tuple((code,) for code in range(1 << depth))
    return PayloadGrid(name=name, values=values, arity=arity,
                       keys=keys, partition=TREE_PARTITION)
