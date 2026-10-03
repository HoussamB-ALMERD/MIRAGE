"""Diagram figures 1-3 (overview, system and threat model, attack pipeline), same style as make_figures.py.

Usage (from the project root): venv/bin/python paper/figures/make_diagrams.py
Writes paper/figures/fig{1,2,3}_*.pdf (+ .png previews). No data needed.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4
plt.rcParams.update({"font.size": 7, "legend.fontsize": 6, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
HONEST, VICTIM, SYBIL, POISON = "#ffffff", "#0072b2", "#d55e00", "#000000"


def letter(ax, i, x=0.0, y=1.0):
    ax.text(x, y, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)


def node(ax, xy, kind, r=0.055, label=None):
    face = {"honest": HONEST, "victim": VICTIM, "sybil": SYBIL}[kind]
    hatch = "////" if kind == "sybil" else None
    ax.add_patch(Circle(xy, r, facecolor=face, edgecolor="black", lw=0.7, hatch=hatch, zorder=3))
    if label:
        ax.text(xy[0], xy[1], label, ha="center", va="center", fontsize=6, zorder=4,
                color="white" if kind == "victim" else "black", fontweight="bold")


def link(ax, a, b, style="-", color="#555555", lw=0.7, z=1):
    ax.plot([a[0], b[0]], [a[1], b[1]], ls=style, color=color, lw=lw, zorder=z)


def arrow(ax, a, b, color="black", lw=0.9, style="-|>", ls="-", ms=6, z=5, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, ls=ls, zorder=z,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def ring(n, r, c=(0, 0), start=90):
    ang = np.deg2rad(start + 360 * np.arange(n) / n)
    return [(c[0] + r * np.cos(a), c[1] + r * np.sin(a)) for a in ang]


def legend(ax, loc="lower center", ncol=3, bbox=None):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", mfc=HONEST, mec="black", ms=6, label="Honest node"),
         Line2D([], [], marker="o", ls="", mfc=VICTIM, mec="black", ms=6, label="Victim"),
         Line2D([], [], marker="o", ls="", mfc=SYBIL, mec="black", ms=6, label="Sybil")]
    ax.legend(handles=h, loc=loc, ncol=ncol, frameon=False, bbox_to_anchor=bbox, handletextpad=0.2, columnspacing=1.0)


# ---------------------------------------------------------------------------------------------- Fig. 1
def fig1():
    """Overview: (a) normal neighbourhood, (b) eclipsed victim whose Sybil updates look unrelated to each other
    but average to the poisoned direction."""
    fig, axes = plt.subplots(1, 3, figsize=(170 * MM, 58 * MM), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)

    # (a) normal neighbourhood inside a larger network
    ax = axes[0]
    inner = ring(6, 0.5)
    outer = ring(10, 0.92, start=72)
    for i, p in enumerate(inner):
        link(ax, (0, 0), p)
        link(ax, p, outer[(i * 10) // 6])
        link(ax, p, outer[((i * 10) // 6 + 1) % 10])
    for i in range(10):
        link(ax, outer[i], outer[(i + 1) % 10], color="#bbbbbb")
    for p in inner + outer:
        node(ax, p, "honest", r=0.07)
    node(ax, (0, 0), "victim", r=0.09, label="v")
    letter(ax, 0, -0.02, 1.02)

    # (b) eclipse: honest links cut, Sybils take every slot
    ax = axes[1]
    for i, p in enumerate(inner):
        link(ax, p, outer[(i * 10) // 6], color="#bbbbbb")
        link(ax, p, outer[((i * 10) // 6 + 1) % 10], color="#bbbbbb")
        cut = (0.8 * p[0], 0.8 * p[1])
        link(ax, (0, 0), p, style=(0, (2, 2)), color="#bbbbbb")
        ax.text(*cut, "×", ha="center", va="center", fontsize=8, color="#666666", zorder=4)
    for i in range(10):
        link(ax, outer[i], outer[(i + 1) % 10], color="#dddddd")
    syb = ring(5, 0.27, start=54)
    for p in syb:
        link(ax, (0, 0), p, color=SYBIL, lw=1.0, z=2)
    for p in inner + outer:
        node(ax, p, "honest", r=0.07)
    for p in syb:
        node(ax, p, "sybil", r=0.065)
    node(ax, (0, 0), "victim", r=0.09, label="v")
    letter(ax, 1, -0.02, 1.02)

    # (c) the updates the victim receives
    ax = axes[2]
    ax.set_xlim(-0.25, 1.45)
    ax.set_ylim(-1.05, 1.05)
    o = np.array([0.0, 0.0])
    t = np.array([0.75, 0.0])
    # Sybil updates: the poisoned update plus mutually different orthogonal parts that sum to zero (schematic: in
    # the model's high-dimensional space the orthogonal parts are also mutually orthogonal, giving cosine beta^2)
    spread = np.array([-0.9, -0.42, 0.06, 0.48, 0.78])
    spread -= spread.mean()
    for a in spread:
        arrow(ax, o, o + t + np.array([0.0, a]), color=SYBIL, lw=0.9)
    arrow(ax, o, o + t, color=POISON, lw=1.6, ms=8, z=6)
    node(ax, o, "victim", r=0.06)
    ax.text(*(o + t + np.array([0.06, 0.0])), "shared part:\npoisoned\nupdate $s\\,t$", fontsize=6, va="center")
    ax.text(0.95, 0.8, "Sybil updates,\npairwise cosine $\\beta^2$", fontsize=6, va="center", color=SYBIL)
    letter(ax, 2, -0.02, 1.02)

    legend(axes[1], loc="upper center", ncol=3, bbox=(0.5, -0.02))
    fig.subplots_adjust(wspace=0.05)
    save(fig, "fig1_overview")


# ---------------------------------------------------------------------------------------------- Fig. 2
def fig2():
    """System and threat model: decentralized FL graph, the victim, the attacker's Sybils, knowledge and capabilities."""
    fig, ax = plt.subplots(figsize=(170 * MM, 80 * MM))
    ax.axis("off")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8.0)

    g = nx.barabasi_albert_graph(22, 2, seed=4)
    pos = nx.kamada_kawai_layout(g)
    P = np.array([pos[n] for n in g.nodes])
    P = (P - P.min(0)) / (P.max(0) - P.min(0))
    pos = {n: (0.3 + 4.9 * P[i, 0], 1.4 + 4.6 * P[i, 1]) for i, n in enumerate(g.nodes)}
    victim = max((n for n in g.nodes if g.degree(n) == 3), key=lambda n: pos[n][0] + pos[n][1])
    vnb = list(g.neighbors(victim))
    # the victim is drawn in the free upper-right corner so its Sybils have room
    cx, cy = 2.75, 3.7
    vx, vy = 5.6, 6.3
    pos[victim] = (vx, vy)
    for a, b in g.edges:
        if victim in (a, b):
            continue
        link(ax, pos[a], pos[b], color="#999999", lw=0.6)
    for j in vnb:
        link(ax, pos[victim], pos[j], style=(0, (2, 2)), color="#bbbbbb", lw=0.7)
    for n in g.nodes:
        if n != victim:
            node(ax, pos[n], "honest", r=0.17)
    away = np.arctan2(vy - cy, vx - cx)
    angs = away + np.deg2rad([-110, -55, 0, 55, 110])
    for a in angs:
        p = (vx + 0.8 * np.cos(a), vy + 0.8 * np.sin(a))
        link(ax, (vx, vy), p, color=SYBIL, lw=1.0, z=2)
        node(ax, p, "sybil", r=0.16)
    node(ax, (vx, vy), "victim", r=0.22, label="v")

    def box(x, y, w, h, title, lines):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white", ec="black",
                                    lw=0.8))
        ax.text(x + 0.2, y + h - 0.2, title, fontsize=7, fontweight="bold", va="top")
        ax.text(x + 0.2, y + h - 0.7, "\n".join(lines), fontsize=6, va="top", linespacing=1.5)

    W, H = 4.95, 3.6
    box(6.95, 4.2, W, H, "Attacker's goal", [
        "• v labels inputs of every class",
        "   as the attacker's target class",
        "• Other nodes stay unaffected",
        "• Stealth: Sybil updates unlike",
        "   each other (similarity checks)"])
    box(12.0, 4.2, W, H, "Knowledge", [
        "• v's current model (shared with",
        "   its neighbours every round)",
        "• A small auxiliary dataset (2%)",
        "• No honest data or updates",
        "• No knowledge of the defense's",
        "   settings"])
    box(6.95, 0.3, W, H, "Capabilities", [
        "• Run k Sybil identities",
        "• Occupy all, or a fraction, of",
        "   v's neighbour slots (eclipse)",
        "• Send each Sybil model after",
        "   seeing v's fresh local model"])
    box(12.0, 0.3, W, H, "Defender (the victim)", [
        "• Aggregates the received models",
        "   with a robust rule: median,",
        "   trimmed mean, Multi-Krum,",
        "   FoolsGold, norm clipping,",
        "   clipping + FoolsGold, or an",
        "   own-update anchor"])
    legend(ax, loc="upper left", ncol=3, bbox=(0.0, 1.02))
    save(fig, "fig2_threat_model")


# ---------------------------------------------------------------------------------------------- Fig. 3
def fig3():
    """Attack pipeline, stages A-C, one round."""
    fig, ax = plt.subplots(figsize=(170 * MM, 54 * MM))
    ax.axis("off")
    ax.set_xlim(0, 17)
    ax.set_ylim(0.8, 6.2)

    Y, H = 1.0, 4.7  # stage boxes: drawing in the upper part (y 3.0-5.0), text in the lower part (y 1.15-2.6)

    def stage(x, w, letter_, title, body):
        ax.add_patch(FancyBboxPatch((x, Y), w, H, boxstyle="round,pad=0.02,rounding_size=0.15", fc="white", ec="black",
                                    lw=0.8))
        ax.add_patch(Circle((x + 0.35, Y + H - 0.35), 0.22, fc="black", ec="black", zorder=4))
        ax.text(x + 0.35, Y + H - 0.35, letter_, color="white", ha="center", va="center", fontsize=7, fontweight="bold",
                zorder=5)
        ax.text(x + 0.7, Y + H - 0.35, title, fontsize=7, fontweight="bold", va="center")
        ax.text(x + 0.2, Y + 0.15, body, fontsize=6, va="bottom", linespacing=1.6)

    stage(0.1, 4.6, "A", "Eclipse the victim",
          "Sybils take all of v's neighbour\nslots, or a fraction of them.\n"
          "Done once, before the attack.")
    stage(5.45, 4.6, "B", "Poison direction",
          "Every round, train v's fresh model\n$h_v$ on poisoned auxiliary data:\n"
          "$t = \\mathrm{Train}_{\\mathrm{poison}}(h_v) - h_v$")
    stage(10.8, 6.1, "C", "Decorrelated Sybil updates",
          "$\\Delta_i = (s/\\beta)\\,\\|t\\|\\,(\\beta\\,\\hat t + \\sqrt{1-\\beta^2}\\,e_i)$,  $e_i \\perp \\hat t$,  "
          "$e_i \\perp e_j$\n"
          "$\\cos(\\Delta_i,\\Delta_j) = \\beta^2$;  component along $\\hat t$: $s\\,t$\n"
          "Every round, Sybil $i$ sends $h_v + \\Delta_i$")
    for x0 in (4.72, 10.07):
        arrow(ax, (x0, 3.9), (x0 + 0.68, 3.9), lw=1.2, ms=9)

    # A: small eclipse sketch
    c = (2.4, 4.0)
    for p in ring(5, 0.55, c, start=54):
        link(ax, c, p, color=SYBIL, lw=0.9)
    for p in ring(6, 1.05, c, start=90):
        link(ax, c, p, style=(0, (2, 2)), color="#bbbbbb")
        node(ax, p, "honest", r=0.15)
    for p in ring(5, 0.55, c, start=54):
        node(ax, p, "sybil", r=0.15)
    node(ax, c, "victim", r=0.2, label="v")

    # B: model space sketch, xv -> hv (local step) and hv -> hv + t (poison step)
    xv, hv = np.array([6.3, 3.05]), np.array([7.6, 3.55])
    ht = hv + np.array([1.7, 1.0])
    arrow(ax, xv, hv, color=VICTIM, lw=1.0)
    arrow(ax, hv, ht, color=POISON, lw=1.4, ms=8)
    ax.plot(*xv, "o", color="#999999", ms=4, zorder=6)
    ax.plot(*hv, "o", color=VICTIM, ms=5, zorder=6)
    ax.text(*(xv + np.array([0.0, -0.18])), "$x_v$", fontsize=7, ha="center", va="top")
    ax.text(*(hv + np.array([0.1, -0.15])), "$h_v$", fontsize=7, ha="left", va="top")
    ax.text(*((xv + hv) / 2 + np.array([0.25, -0.12])), "local\ntraining", fontsize=5.5, ha="left", va="top",
            color=VICTIM)
    ax.text(*((hv + ht) / 2 + np.array([-0.1, 0.2])), "$t$", fontsize=8, ha="right")

    # C: fan of Sybil updates around s t
    o = np.array([12.2, 3.9])
    t = np.array([2.0, 0.0])
    offs = np.array([-0.95, -0.5, 0.05, 0.45, 0.95])
    offs -= offs.mean()
    for a in offs:
        arrow(ax, o, o + t + np.array([0.0, a]), color=SYBIL, lw=0.8)
    arrow(ax, o, o + t, color=POISON, lw=1.6, ms=8, z=6)
    ax.plot(*o, "o", color=VICTIM, ms=5, zorder=7)
    ax.text(*(o + np.array([-0.15, 0.0])), "$h_v$", fontsize=7, ha="right", va="center")
    ax.text(*(o + t + np.array([0.15, 0.0])), "$s\\,t$ (shared)", fontsize=7, va="center")
    ax.text(*(o + t + np.array([0.15, offs[-1]])), "$\\Delta_i$", fontsize=7, va="center", color=SYBIL)
    legend(ax, loc="upper left", ncol=3, bbox=(0.0, 1.03))
    save(fig, "fig3_attack_pipeline")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("diagrams written to", OUT)
