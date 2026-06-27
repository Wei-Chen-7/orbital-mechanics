"""Barnes-Hut quadtree force evaluation -- O(N log N) gravity in 2D.

The direct sum in :mod:`nbody.forces` costs O(N^2), which becomes the
bottleneck well before "galaxy" sizes.  Barnes-Hut recursively partitions space
into a quadtree; when a group of distant bodies subtends a small enough angle
(cell width / distance < ``theta``) it is replaced by its center of mass.  That
drops the per-step cost to O(N log N) at the price of a controlled,
``theta``-dependent force error -- which ``tests/test_barnes_hut.py`` checks
against the exact direct sum.

Plug it into the integrators with::

    from nbody.barnes_hut import barnes_hut_accelerations
    traj = simulate(system, dt, n, acc_func=barnes_hut_accelerations(theta=0.5))

This implementation is 2D (a quadtree); the API mirrors
:func:`nbody.forces.accelerations`.
"""

from __future__ import annotations

import numpy as np


class _Node:
    """A square quadtree cell holding the aggregate mass of its subtree."""

    __slots__ = ("cx", "cy", "h", "mass", "comx", "comy", "body", "children")

    def __init__(self, cx, cy, h):
        self.cx = cx
        self.cy = cy
        self.h = h  # half-width of the square cell
        self.mass = 0.0
        self.comx = 0.0
        self.comy = 0.0
        self.body = -1  # index of the single body in a leaf, else -1
        self.children = None  # list of 4 child cells once subdivided


def _accumulate(node, x, y, m):
    """Fold a body of mass ``m`` at ``(x, y)`` into a node's running COM."""
    total = node.mass + m
    node.comx = (node.comx * node.mass + x * m) / total
    node.comy = (node.comy * node.mass + y * m) / total
    node.mass = total


def _insert_into_child(node, k, positions, masses, depth, max_depth):
    x, y = positions[k, 0], positions[k, 1]
    half = node.h * 0.5
    idx = (1 if x >= node.cx else 0) + (2 if y >= node.cy else 0)
    child = node.children[idx]
    if child is None:
        ox = node.cx + (half if (idx & 1) else -half)
        oy = node.cy + (half if (idx & 2) else -half)
        child = _Node(ox, oy, half)
        node.children[idx] = child
    _insert(child, k, positions, masses, depth + 1, max_depth)


def _insert(node, i, positions, masses, depth, max_depth):
    x, y, m = positions[i, 0], positions[i, 1], masses[i]

    # Empty cell -> becomes a leaf holding body i.
    if node.body == -1 and node.children is None and node.mass == 0.0:
        node.body = i
        node.mass = m
        node.comx, node.comy = x, y
        return

    # Leaf already holding one body -> subdivide and push both down.
    if node.children is None:
        if depth >= max_depth:
            # Coincident/near-coincident bodies: stop splitting, bucket them.
            _accumulate(node, x, y, m)
            node.body = -1
            return
        node.children = [None, None, None, None]
        j = node.body
        node.body = -1
        _insert_into_child(node, j, positions, masses, depth, max_depth)
        _accumulate(node, x, y, m)
        _insert_into_child(node, i, positions, masses, depth, max_depth)
        return

    # Internal cell -> update aggregate and descend.
    _accumulate(node, x, y, m)
    _insert_into_child(node, i, positions, masses, depth, max_depth)


def build_tree(positions, masses, max_depth=64):
    """Build the quadtree root over all bodies."""
    xmin, ymin = positions[:, 0].min(), positions[:, 1].min()
    xmax, ymax = positions[:, 0].max(), positions[:, 1].max()
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    half = half * 1.0000001 + 1e-12  # pad so all points are strictly inside

    root = _Node(cx, cy, half)
    for i in range(len(masses)):
        _insert(root, i, positions, masses, 0, max_depth)
    return root


def _force_on(root, x, y, G, softening, theta):
    """Acceleration at point ``(x, y)`` by walking the tree (iterative)."""
    soft2 = softening * softening
    theta2 = theta * theta
    ax = ay = 0.0
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None or node.mass == 0.0:
            continue
        dx = node.comx - x
        dy = node.comy - y
        d2 = dx * dx + dy * dy + soft2

        if node.children is None:
            # Leaf: skip the target body itself (dx = dy = 0).
            if dx == 0.0 and dy == 0.0:
                continue
            inv = d2 ** -1.5
            ax += G * node.mass * dx * inv
            ay += G * node.mass * dy * inv
        else:
            width = 2.0 * node.h
            # Opening criterion: cell is far enough to treat as a point mass.
            if width * width < theta2 * d2:
                inv = d2 ** -1.5
                ax += G * node.mass * dx * inv
                ay += G * node.mass * dy * inv
            else:
                for c in node.children:
                    if c is not None:
                        stack.append(c)
    return ax, ay


def accelerations_bh(positions, masses, G=1.0, softening=0.0, theta=0.5,
                     max_depth=64):
    """Barnes-Hut acceleration of every body (2D)."""
    positions = np.asarray(positions, dtype=float)
    masses = np.asarray(masses, dtype=float)
    if positions.shape[1] != 2:
        raise ValueError("Barnes-Hut implementation is 2D only")

    n = len(masses)
    root = build_tree(positions, masses, max_depth)
    acc = np.empty((n, 2))
    for i in range(n):
        ax, ay = _force_on(root, positions[i, 0], positions[i, 1],
                           G, softening, theta)
        acc[i, 0] = ax
        acc[i, 1] = ay
    return acc


def barnes_hut_accelerations(theta=0.5, max_depth=64):
    """Return an ``acc_func`` closure for :func:`nbody.integrators.simulate`."""

    def acc(positions, masses, G, softening):
        return accelerations_bh(positions, masses, G, softening, theta, max_depth)

    return acc
