"""Statistical analysis for the revision (paper/09_revision.md, c4).

For every dataset and aggregation rule of the main grid:
  * 95% confidence interval of the ASR of every attack (Student t over seeds);
  * paired comparison of Epistemic Eclipse with every baseline attack, paired by seed (the seed fixes the data split,
    graph and honest trajectory, so the two attacks face the same network): paired t-test and exact Wilcoxon
    signed-rank test, two-sided, Holm-corrected within each dataset and rule (seven comparisons per family).
Writes paper/manuscript/tables/tabA_stats.tex (appendix) and refs-free summary lines to stdout for the text.

Usage (from the project root): venv/bin/python paper/figures/stats.py [--ext]
"""
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_tables import AGGS, ATTACKS, DATASETS, EXT_AGGS, EXT_SETS, LAST, NEW_AGGS, load, write  # noqa: E402


def per_seed(runs, name, metric="asr"):
    """Last-10-round mean per seed (NaN if the variant or every value is missing), in %."""
    out = []
    for r in runs:
        if name not in r["variants"]:
            out.append(np.nan)
            continue
        x = np.asarray(r["variants"][name][metric][-LAST:], float)
        x = np.where(np.isfinite(x), x, np.nan)
        out.append(100 * np.nanmean(x) if np.isfinite(x).any() else np.nan)
    return np.asarray(out)


def ci95(x):
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return max(0.0, x.mean() - h), min(100.0, x.mean() + h)  # ASR is a percentage


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    if len(d) < 3 or np.allclose(d, 0):
        return d.mean() if len(d) else np.nan, 1.0, 1.0, len(d)
    pt = stats.ttest_rel(a, b).pvalue
    pw = stats.wilcoxon(a, b, method="exact" if len(d) <= 25 else "auto").pvalue
    return d.mean(), pt, pw, len(d)


def analyse(groups):
    """groups: list of (dataset label, [(runs, aggs), ...]). One test family per dataset and rule."""
    rows = []
    for dl, sources in groups:
        for runs, aggs in sources:
            if not runs:
                continue
            for g, gl in aggs:  # EE against the seven baselines
                ee = per_seed(runs, f"eclipse|{g}")
                if not np.isfinite(ee).any():  # rule not in this set's grid (reduced ResNet-20/Shakespeare grid)
                    continue
                tests = [(al, *compare(ee, per_seed(runs, f"{att}|{g}"))) for att, al in ATTACKS[:-1]]
                ok = [i for i, t in enumerate(tests) if np.isfinite(t[1])]  # baselines run in this grid
                adj_t, adj_w = np.full(len(tests), np.nan), np.full(len(tests), np.nan)
                adj_t[ok], adj_w[ok] = holm([tests[i][2] for i in ok]), holm([tests[i][3] for i in ok])
                lo, hi = ci95(ee)
                cells = [(diff, at, aw) for (al, diff, pt, pw, n), at, aw in zip(tests, adj_t, adj_w)]
                rows.append((dl, gl, np.nanmean(ee), lo, hi, len(runs), cells))
    return rows


def table(rows, name, label, scope):
    lines = ["\\begin{table*}[t]",
             f"\\caption{{Paired comparison of Epistemic Eclipse with each baseline attack ({scope}). "
             "EE gives the mean ASR of Epistemic Eclipse and its 95\\% confidence interval (Student $t$ over seeds, "
             "truncated to [0, 100]). "
             "The other cells give the mean ASR difference in percentage points, Epistemic Eclipse minus the baseline, "
             "and in parentheses the Holm-adjusted $p$-value of the paired $t$-test, corrected over the "
             "comparisons of each row (seven, or four in the reduced ResNet-20 and Shakespeare grid); -- , baseline not "
             "run; $^\\ast$ marks differences that are also significant at 0.05 under the exact "
             f"Wilcoxon signed-rank test with the same correction}}\\label{{{label}}}",
             "\\centering\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%",
             "\\begin{tabular}{lll" + "r" * (len(ATTACKS) - 1) + "}", "\\toprule",
             "Dataset & Rule & EE [95\\% CI] & " + " & ".join(al for _, al in ATTACKS[:-1]) + " \\\\", "\\midrule"]
    prev = None
    for dl, gl, m, lo, hi, n, cells in rows:
        if prev is not None and dl != prev:
            lines.append("\\midrule")
        txt = []
        for diff, at, aw in cells:
            if not np.isfinite(diff):  # baseline not in this set's grid
                txt.append("--")
                continue
            star = "$^\\ast$" if aw < 0.05 else ""
            ptxt = "<0.001" if at < 0.001 else f"{at:.3f}"
            txt.append(f"{diff:+.1f} ({ptxt}){star}")
        lines.append(" & ".join([dl if dl != prev else "", gl, f"{m:.1f} [{lo:.1f}, {hi:.1f}]"] + txt) + " \\\\")
        prev = dl
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]
    write(name, "\n".join(lines))


def main():
    # Main datasets: the nine original rules (main grid) and the five added rules (defense grid, same seeds).
    groups = [(dl, [(load(f"{ds}_seed*.json"), AGGS), (load(f"{ds}_defense_seed*.json"), NEW_AGGS[1:])])
              for ds, dl in DATASETS]
    rows = analyse(groups)
    for dl, gl, m, lo, hi, n, _ in rows:
        print(f"{dl:14s} {gl:8s} EE ASR {m:5.1f}  95% CI [{lo:5.1f}, {hi:5.1f}]  (n={n})")
    table(rows, "tabA_stats.tex", "tab:stats", "MNIST, Fashion-MNIST and CIFAR-10")
    if "--ext" in sys.argv:  # added datasets and architectures (five seeds)
        ext = analyse([(dl, [(load(f"{pre}_seed*.json"), EXT_AGGS)]) for pre, dl in EXT_SETS])
        for dl, gl, m, lo, hi, n, _ in ext:
            print(f"{dl:24s} {gl:8s} EE ASR {m:5.1f}  95% CI [{lo:5.1f}, {hi:5.1f}]  (n={n})")
        table(ext, "tabA_stats_ext.tex", "tab:stats_ext", "added datasets and architectures")


if __name__ == "__main__":
    main()
