"""Barnes-Hut tree-code gravity -- O(N log N) in 2D (quadtree) or 3D (octree).

The direct sum in :mod:`nbody.forces` costs O(N^2), which becomes the
bottleneck well before "galaxy" sizes.  Barnes-Hut recursively partitions space
into a tree -- a quadtree in 2D, an octree in 3D -- and when a group of distant
bodies subtends a small enough angle (cell width / distance < ``theta``) it is
replaced by its center of mass.  That drops the per-step cost to O(N log N) at
the price of a controlled, ``theta``-dependent force error -- which
``tests/test_barnes_hut.py`` checks against the exact direct sum in both 2D and
3D.

Plug it into the integrators with::

    from nbody.barnes_hut import barnes_hut_accelerations
    traj = simulate(system, dt, n, acc_func=barnes_hut_accelerations(theta=0.5))

The implementation is dimension-agnostic: a cell has ``2**D`` children, indexed
by one bit per axis.  The API mirrors :func:`nbody.forces.accelerations`.
"""

from __future__ import annotations

import numpy as np


class _Node:
    """A (hyper)cubic cell holding the aggregate mass of its subtree.

    ``center`` and ``com`` are length-``D`` lists of floats; ``half`` is the
    half-width of the cell.  A node is empty, a leaf (one ``body``), or an
    internal node with ``2**D`` ``children``.
    """

    __slots__ = ("center", "half", "mass", "com", "body", "children")

    def __init__(self, center, half):
        self.center = center  # list of D floats
        self.half = half
        self.mass = 0.0
        self.com = [0.0] * len(center)
        self.body = -1
        self.children = None


def _accumulate(node, point, m):
    """Fold a body of mass ``m`` at ``point`` into a node's running COM."""
    total = node.mass + m
    inv = 1.0 / total
    com = node.com
    old = node.mass
    for k in range(len(point)):
        com[k] = (com[k] * old + point[k] * m) * inv
    node.mass = total


def _child_index(node, point):
    idx = 0
    center = node.center
    for ax in range(len(point)):
        if point[ax] >= center[ax]:
            idx |= 1 << ax
    return idx


def _insert_into_child(node, k, points, masses, depth, max_depth):
    point = points[k]
    child_half = node.half * 0.5
    idx = _child_index(node, point)
    child = node.children[idx]
    if child is None:
        center = [
            node.center[ax] + (child_half if (idx >> ax) & 1 else -child_half)
            for ax in range(len(point))
        ]
        child = _Node(center, child_half)
        node.children[idx] = child
    _insert(child, k, points, masses, depth + 1, max_depth)


def _insert(node, i, points, masses, depth, max_depth):
    point = points[i]
    m = masses[i]

    # Empty cell -> becomes a leaf holding body i.
    if node.body == -1 and node.children is None and node.mass == 0.0:
        node.body = i
        node.mass = m
        node.com = list(point)
        return

    # Leaf already holding one body -> subdivide and push both down.
    if node.children is None:
        if depth >= max_depth:
            # Coincident/near-coincident bodies: stop splitting, bucket them.
            _accumulate(node, point, m)
            node.body = -1
            return
        node.children = [None] * (1 << len(point))
        j = node.body
        node.body = -1
        _insert_into_child(node, j, points, masses, depth, max_depth)
        _accumulate(node, point, m)
        _insert_into_child(node, i, points, masses, depth, max_depth)
        return

    # Internal cell -> update aggregate and descend.
    _accumulate(node, point, m)
    _insert_into_child(node, i, points, masses, depth, max_depth)


def _bounds(positions):
    """Center and half-width of a cube enclosing all bodies."""
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = list(0.5 * (mins + maxs))
    half = 0.5 * float(np.max(maxs - mins))
    half = half * 1.0000001 + 1e-12  # pad so all points are strictly inside
    return center, half


def build_tree(positions, masses, max_depth=64):
    """Build the tree root over all bodies (2D or 3D)."""
    positions = np.asarray(positions, dtype=float)
    points = [tuple(row) for row in positions]
    masses = [float(x) for x in masses]
    center, half = _bounds(positions)
    root = _Node(center, half)
    for i in range(len(masses)):
        _insert(root, i, points, masses, 0, max_depth)
    return root


def _force_on(root, point, G, softening, theta):
    """Acceleration at ``point`` (length-D tuple) by walking the tree."""
    d = len(point)
    soft2 = softening * softening
    theta2 = theta * theta
    acc = [0.0] * d
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or node.mass == 0.0:
            continue

        com = node.com
        dx = [com[k] - point[k] for k in range(d)]
        d2 = soft2
        for v in dx:
            d2 += v * v

        if node.children is None:
            # Leaf: skip the target body itself (all components zero).
            if d2 == soft2:
                continue
            inv = G * node.mass * d2 ** -1.5
            for k in range(d):
                acc[k] += dx[k] * inv
        else:
            width = 2.0 * node.half
            # Opening criterion: cell far enough to treat as a single mass.
            if width * width < theta2 * d2:
                inv = G * node.mass * d2 ** -1.5
                for k in range(d):
                    acc[k] += dx[k] * inv
            else:
                for c in node.children:
                    if c is not None:
                        stack.append(c)
    return acc


def accelerations_bh(positions, masses, G=1.0, softening=0.0, theta=0.5,
                     max_depth=64):
    """Barnes-Hut acceleration of every body (2D quadtree or 3D octree)."""
    positions = np.asarray(positions, dtype=float)
    d = positions.shape[1]
    if d not in (2, 3):
        raise ValueError("Barnes-Hut supports 2D or 3D positions only")

    points = [tuple(row) for row in positions]
    masses_list = [float(x) for x in masses]
    center, half = _bounds(positions)
    root = _Node(center, half)
    for i in range(len(masses_list)):
        _insert(root, i, points, masses_list, 0, max_depth)

    acc = np.empty((len(points), d))
    for i, p in enumerate(points):
        acc[i] = _force_on(root, p, G, softening, theta)
    return acc


def barnes_hut_accelerations(theta=0.5, max_depth=64):
    """Return an ``acc_func`` closure for :func:`nbody.integrators.simulate`."""

    def acc(positions, masses, G, softening):
        return accelerations_bh(positions, masses, G, softening, theta, max_depth)

    return acc
