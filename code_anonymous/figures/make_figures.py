"""Journal-style result figures (Cybersecurity: 85 or 170 mm wide, no in-plot titles, panel letters, PDF).

Usage (from the project root): venv/bin/python paper/figures/make_figures.py
Reads results/dfl/{mnist,fmnist,cifar10}_seed*.json and writes paper/figures/fig{4..8}_*.pdf (+ .png previews).
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
MM = 1 / 25.4
DATASETS = [("mnist", "MNIST"), ("fmnist", "Fashion-MNIST"), ("cifar10", "CIFAR-10")]
LAST = 10
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6, "xtick.labelsize": 6,
                     "ytick.labelsize": 6, "lines.linewidth": 1.0, "lines.markersize": 3, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "font.family": "DejaVu Sans"})
ATTACK_STYLE = {  # label, colour, marker, linestyle (distinguishable in greyscale)
    "eclipse": ("Epistemic Eclipse", "#000000", "o", "-"),
    "identical": ("Identical Sybils", "#d55e00", "s", "--"),
    "correlated": ("Correlated Sybils", "#e69f00", "^", ":"),
    "minmax-t": ("Min-Max, targeted (oracle)", "#0072b2", "v", "-."),
    "lie": ("LIE (oracle)", "#009e73", "D", (0, (1, 1))),
    "none": ("No attack", "#999999", "", "-"),
}


def load(ds):
    """Load runs; variants rerun with the finiteness check (`*_fixnan_seed*.json`) replace the original ones."""
    runs = []
    for f in sorted(glob.glob(f"results/dfl/{ds}_seed*.json")):
        r = json.load(open(f))
        fix = f.replace("_seed", "_fixnan_seed")
        if os.path.exists(fix):
            r["variants"].update(json.load(open(fix))["variants"])
        runs.append(r)
    return runs


def series(runs, name, metric="asr"):
    a = np.array([r["variants"][name][metric] for r in runs if name in r["variants"]], dtype=float)
    return a.mean(0), a.std(0)


def final(runs, name, metric="asr"):
    # non-finite entries count as missing, and a seed without any finite value is left out (as in tab4_stealth)
    v = [x[np.isfinite(x)].mean() for r in runs if name in r["variants"]
         for x in [np.asarray(r["variants"][name][metric][-LAST:], float)] if np.isfinite(x).any()]
    return (np.mean(v), np.std(v)) if v else (np.nan, np.nan)


def smooth(y, w=3):
    """Centred moving average over w rounds (edges use the available rounds)."""
    k = np.ones(w)
    return np.convolve(y, k, "same") / np.convolve(np.ones_like(y), k, "same")


def letter(ax, i):
    ax.text(-0.18, 1.04, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4_asr_rounds(data):
    """ASR over rounds: rows = mean / FoolsGold, columns = datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(170 * MM, 95 * MM), sharey=True)
    for c, (ds, label) in enumerate(DATASETS):
        for r, agg in enumerate(("mean", "foolsgold")):
            ax = axes[r, c]
            for att in ("eclipse", "identical", "correlated", "minmax-t", "lie", "none"):
                if not data[ds]:
                    continue
                m, s = series(data[ds], f"{att}|{agg}")
                m, s = smooth(m), smooth(s)
                x = np.arange(1, len(m) + 1)
                lab, col, mk, ls = ATTACK_STYLE[att]
                ax.plot(x, 100 * m, color=col, ls=ls, marker=mk, markevery=max(1, len(m) // 8), label=lab)
                ax.fill_between(x, 100 * (m - s), 100 * (m + s), color=col, alpha=0.12, lw=0)
            ax.set_ylim(-3, 103)
            if r == 1:
                ax.set_xlabel("Round")
            if c == 0:
                ax.set_ylabel(f"ASR (%), {'mean' if agg == 'mean' else 'FoolsGold'}")
            if r == 0:
                ax.text(0.5, 1.06, label, transform=ax.transAxes, ha="center", fontsize=7)
            letter(ax, r * 3 + c)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig4_asr_rounds")


def fig5_stealth(data):
    """Pairwise Sybil cosine against norm ratio (mean aggregation), one point per attack and dataset."""
    fig, ax = plt.subplots(figsize=(85 * MM, 60 * MM))
    markers = {"mnist": "o", "fmnist": "s", "cifar10": "^"}
    for ds, label in DATASETS:
        for att in ("eclipse", "identical", "correlated", "minmax-t", "lie"):
            if not data[ds]:
                continue
            cs, _ = final(data[ds], f"{att}|mean", "cos_sybil_ref_prev")
            nr, _ = final(data[ds], f"{att}|mean", "norm_ratio")
            ax.scatter(nr, cs, marker=markers[ds], color=ATTACK_STYLE[att][1], s=14, edgecolor="k", lw=0.3)
    for att in ("eclipse", "identical", "correlated", "minmax-t", "lie"):
        ax.scatter([], [], color=ATTACK_STYLE[att][1], marker="o", s=14, label=ATTACK_STYLE[att][0])
    for ds, label in DATASETS:
        ax.scatter([], [], color="white", edgecolor="k", marker=markers[ds], s=14, label=label)
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xticks([0.5, 1, 2, 4, 6])
    ax.set_xticklabels(["0.5", "1", "2", "4", "6"])
    ax.set_xlabel("Norm ratio (Sybil / honest update)")
    ax.set_ylabel("Mean pairwise cosine between Sybils")
    ax.legend(frameon=False, fontsize=5, loc="center right")
    save(fig, "fig5_stealth")


def fig6_sensitivity(data):
    """Sensitivity of Epistemic Eclipse (mean and FoolsGold) on MNIST and Fashion-MNIST."""
    params = [("k", (1, 3, 5, 8, 12), "Number of Sybils k"), ("beta", (0.05, 0.1, 0.2, 0.3, 0.5), "Shared component β"),
              ("frac", (0.5, 0.6, 0.75, 0.9, 1.0), "Eclipse fraction"), ("scale", (0.5, 1.0, 2.0, 3.0), "Poison scale")]
    default = {"k": 5, "beta": 0.2, "frac": 1.0, "scale": 1.0}
    fig, axes = plt.subplots(1, 5, figsize=(170 * MM, 45 * MM), sharey=True)
    styles = {("mnist", "mean"): ("#000000", "o", "-"), ("mnist", "foolsgold"): ("#000000", "o", "--"),
              ("fmnist", "mean"): ("#0072b2", "s", "-"), ("fmnist", "foolsgold"): ("#0072b2", "s", "--")}
    for i, (p, vals, xl) in enumerate(params):
        ax = axes[i]
        for (ds, agg), (col, mk, ls) in styles.items():
            if not data[ds]:
                continue
            ys, es = [], []
            for v in vals:
                name = f"eclipse|{agg}" if v == default[p] else f"eclipse|{agg}|{p}={v}"
                m, s = final(data[ds], name)
                ys.append(100 * m)
                es.append(100 * s)
            lab = f"{dict(DATASETS)[ds]}, {'mean' if agg == 'mean' else 'FoolsGold'}"
            ax.errorbar(vals, ys, yerr=es, color=col, marker=mk, ls=ls, capsize=1.5, label=lab)
        ax.set_xlabel(xl)
        ax.set_ylim(0, 103)
        if p == "beta":
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xticks(vals)
        ax.set_xticklabels([str(v) for v in vals])
        letter(ax, i)
    # (e) data heterogeneity on MNIST: Dirichlet alpha (IID plotted at the right end)
    ax = axes[4]
    tags = [("0.1", "_a0.1"), ("0.2", ""), ("0.5", "_a0.5"), ("1.0", "_a1.0"), ("IID", "_a0")]
    het = [[json.load(open(f)) for f in sorted(glob.glob(f"results/dfl/mnist{t}_seed*.json"))] for _, t in tags]
    for agg, col, mk, ls, lab in (("mean", "#000000", "o", "-", "MNIST, mean"),
                                  ("foolsgold", "#000000", "o", "--", "MNIST, FoolsGold"),
                                  ("clipfg", "#d55e00", "D", ":", "MNIST, clip + FoolsGold")):
        stats = [final(runs, f"eclipse|{agg}") if runs else (np.nan, np.nan) for runs in het]
        ax.errorbar(range(len(tags)), [100 * m for m, _ in stats], yerr=[100 * s for _, s in stats], color=col,
                    marker=mk, ls=ls, capsize=1.5, label=lab if agg == "clipfg" else None)
    ax.set_xticks(range(len(tags)))
    ax.set_xticklabels([t for t, _ in tags])
    ax.set_xlabel("Dirichlet α (MNIST)")
    letter(ax, 4)
    axes[0].set_ylabel("ASR (%)")
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[4].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig6_sensitivity")


def fig7_ablation(data):
    """Grouped bars: full attack, no eclipse (Sybils added), noise only, identical Sybils; mean and FoolsGold."""
    variants = [("eclipse|{g}", "Full attack"), ("ablation-no-eclipse|{g}", "Without eclipse"),
                ("identical|{g}", "Without decorrelation"), ("ablation-noise-only|{g}", "Without target component")]
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 50 * MM), sharey=True)
    hatches = ["", "///", "xxx", "..."]
    for i, g in enumerate(("mean", "foolsgold")):
        ax = axes[i]
        width = 0.2
        for j, (tpl, lab) in enumerate(variants):
            ys, es = [], []
            for ds, _ in DATASETS[:2]:
                m, s = final(data[ds], tpl.format(g=g)) if data[ds] else (np.nan, np.nan)
                ys.append(100 * m)
                es.append(100 * s)
            ax.bar(np.arange(2) + (j - 1.5) * width, ys, width, yerr=es, capsize=1.5, label=lab,
                   color=["#000000", "#777777", "#bbbbbb", "#ffffff"][j], edgecolor="k", lw=0.5, hatch=hatches[j])
        ax.set_xticks(range(2))
        ax.set_xticklabels([l for _, l in DATASETS[:2]])
        ax.set_ylim(0, 100)
        ax.text(0.02, 0.97, "mean" if g == "mean" else "FoolsGold", transform=ax.transAxes, va="top", fontsize=6)
        letter(ax, i)
    axes[0].set_ylabel("ASR (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig7_ablation")


def fig8_adaptive(data):
    """Norm-constrained attacker: ASR over (cap, beta) against FoolsGold and clip + FoolsGold, per dataset."""
    caps, betas = (1.0, 2.0), (0.2, 0.5, 0.7)
    fig, axes = plt.subplots(2, 3, figsize=(170 * MM, 78 * MM), gridspec_kw={"hspace": 0.45, "wspace": 0.3})
    for c, (ds, label) in enumerate(DATASETS):
        for r, g in enumerate(("foolsgold", "clipfg")):
            ax = axes[r, c]
            grid = np.full((len(caps), len(betas)), np.nan)
            for a, cap in enumerate(caps):
                for b, beta in enumerate(betas):
                    if data[ds]:
                        grid[a, b] = 100 * final(data[ds], f"eclipse-nc|{g}|cap={cap}|beta={beta}")[0]
            im = ax.imshow(grid, cmap="Greys", vmin=0, vmax=100, aspect="auto")
            for a in range(len(caps)):
                for b in range(len(betas)):
                    if not np.isnan(grid[a, b]):
                        ax.text(b, a, f"{grid[a, b]:.0f}", ha="center", va="center", fontsize=6,
                                color="white" if grid[a, b] > 55 else "black")
            ax.set_xticks(range(len(betas)))
            ax.set_xticklabels([str(b) for b in betas])
            ax.set_yticks(range(len(caps)))
            ax.set_yticklabels([str(cp) for cp in caps])
            if r == 1:
                ax.set_xlabel("β")
            if c == 0:
                ax.set_ylabel("Norm cap\n(" + ("FoolsGold" if g == "foolsgold" else "clip + FoolsGold") + ")")
            if r == 0:
                ax.text(0.5, 1.08, label, transform=ax.transAxes, ha="center", fontsize=7)
            letter(ax, r * 3 + c)
    fig.colorbar(im, ax=axes, shrink=0.8, label="ASR (%)")
    save(fig, "fig8_adaptive")


if __name__ == "__main__":
    data = {ds: load(ds) for ds, _ in DATASETS}
    print({ds: len(v) for ds, v in data.items()})
    fig4_asr_rounds(data)
    fig5_stealth(data)
    fig6_sensitivity(data)
    fig7_ablation(data)
    fig8_adaptive(data)
    print("figures written to", OUT)
