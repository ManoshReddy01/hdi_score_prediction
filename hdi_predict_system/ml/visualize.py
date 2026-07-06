# ml/visualize.py
# Turns a numeric prediction into the two PNGs referenced by
# VISUALIZATION_REPORT.graph_path. Kept separate from app.py so the
# charting code can be unit-tested or reused (e.g. from a notebook)
# without booting the Flask app.
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAVY = "#12233D"
PAPER = "#FAF7F0"
AMBER = "#D98E3B"
TEAL = "#2F6F63"
TEAL_LIGHT = "#6FA694"
ROSE = "#B5535D"
SLATE = "#2A2E35"

BAND_COLORS = [ROSE, AMBER, TEAL_LIGHT, TEAL]
BAND_EDGES = [0.0, 0.550, 0.700, 0.800, 1.0]
BAND_LABELS = ["Low", "Medium", "High", "Very High"]


def _style_axes(ax):
    ax.set_facecolor(PAPER)
    for spine in ax.spines.values():
        spine.set_visible(False)


def generate_gauge_chart(score, category, out_path):
    """Semi-circular dial with tier bands and a needle at the predicted score.

    Built with plain Cartesian axes + matplotlib.patches.Wedge (rather
    than a polar projection) so every text label sits at an ordinary
    (x, y) point - simpler to reason about and avoids the edge cases a
    polar projection introduces when text needs to sit at "negative
    radius" positions (e.g. below the dial's center).
    """
    from matplotlib.patches import Wedge

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.55, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")

    # Tier bands, each drawn as a ring wedge from BAND_EDGES[i]*180deg
    # to BAND_EDGES[i+1]*180deg. angle 0 = right side (score 0), angle
    # 180 = left side (score 1) matches a left-to-right reading dial.
    for i in range(len(BAND_EDGES) - 1):
        theta1 = 180 - BAND_EDGES[i + 1] * 180
        theta2 = 180 - BAND_EDGES[i] * 180
        wedge = Wedge((0, 0), 1.0, theta1, theta2, width=0.32, facecolor=BAND_COLORS[i], edgecolor=PAPER, linewidth=2)
        ax.add_patch(wedge)

        mid_angle = math.radians((theta1 + theta2) / 2)
        ax.text(
            1.12 * math.cos(mid_angle),
            1.12 * math.sin(mid_angle),
            BAND_LABELS[i],
            ha="center",
            va="center",
            fontsize=8.5,
            color=SLATE,
            fontweight="bold",
        )

    # Needle: angle 180deg at score=0 (left... actually right-to-left
    # convention below), sweeping to 0deg at score=1.
    needle_angle = math.radians(180 - score * 180)
    nx, ny = 0.62 * math.cos(needle_angle), 0.62 * math.sin(needle_angle)
    ax.plot([0, nx], [0, ny], color=NAVY, linewidth=3, solid_capstyle="round", zorder=5)
    ax.scatter([0], [0], s=140, color=NAVY, zorder=6)

    ax.text(0, -0.20, f"{score:.3f}", ha="center", va="center", fontsize=26, fontweight="bold", color=NAVY)
    ax.text(0, -0.42, category.upper(), ha="center", va="center", fontsize=12, fontweight="bold", color=SLATE, alpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=PAPER)
    plt.close(fig)


def generate_dimension_chart(life_expectancy, mys, eys, gni, out_path):
    """Bar chart of the three normalized UNDP dimension indices."""
    lei = max(0.0, min(1.0, (life_expectancy - 20) / (85 - 20)))
    mysi = max(0.0, min(1.0, mys / 15))
    eysi = max(0.0, min(1.0, eys / 18))
    ei = (mysi + eysi) / 2
    ii = max(0.0, min(1.0, (math.log(max(gni, 1)) - math.log(100)) / (math.log(75000) - math.log(100))))

    labels = ["Health\n(Life Expectancy)", "Education\n(Schooling)", "Living Standards\n(GNI per capita)"]
    values = [lei, ei, ii]
    colors = [ROSE, AMBER, TEAL]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    fig.patch.set_facecolor(PAPER)
    _style_axes(ax)

    bars = ax.bar(labels, values, color=colors, width=0.55, zorder=3)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(axis="x", labelsize=9, colors=SLATE)
    ax.tick_params(axis="y", labelsize=9, colors=SLATE)
    ax.grid(axis="y", color=SLATE, alpha=0.12, linewidth=1, zorder=0)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.03,
            f"{v:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=NAVY,
        )

    ax.set_title("Normalized Dimension Indices", fontsize=12, fontweight="bold", color=NAVY, pad=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=PAPER)
    plt.close(fig)
