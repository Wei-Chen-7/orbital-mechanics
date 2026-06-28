"""Shared helpers for the example scripts: output paths and a plain plot style."""

import os
import sys

# Make the package importable when running a script directly (no install needed).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless: render straight to files
import matplotlib.pyplot as plt  # noqa: E402

FIGURES = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
os.makedirs(FIGURES, exist_ok=True)


def fig_path(name):
    return os.path.join(FIGURES, name)


def use_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.autolayout": True,
    })
