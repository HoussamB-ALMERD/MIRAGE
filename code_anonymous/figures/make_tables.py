"""LaTeX tables (booktabs, no colour or shading, as the journal requires) from results/dfl/*.json.

Usage (from the project root): venv/bin/python paper/figures/make_tables.py
Writes paper/manuscript/tables/tab{2..5}_*.tex. Values: mean ± std over seeds of the last 10 rounds, in %.
"""
import glob
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "manuscript", "tables")
DATASETS = [("mnist", "MNIST"), ("fmnist", "Fashion-MNIST"), ("cifar10", "CIFAR-10")]
ATTACKS = [("none", "No attack"), ("signflip", "Sign flip"), ("identical", "Identical Sybils"),
           ("correlated", "Correlated Sybils"), ("lie", "LIE$^\\dagger$"), ("minmax", "Min-Max$^\\dagger$"),
           ("minmax-t", "Min-Max-T$^\\dagger$"), ("eclipse", "\\textbf{Epistemic Eclipse}")]
AGGS = [("mean", "Mean"), ("trimmed", "TrMean"), ("median", "Median"), ("multikrum", "M-Krum"),
        ("foolsgold", "FG"), ("foolsgold_delta", "FG-$\\Delta$"), ("normclip", "Clip"), ("clipfg", "Clip+FG"),
        ("selfanchor", "Anchor")]
LAST = 10


def load(pattern):
    """Load runs; variants rerun with the finiteness check (`*_fixnan_seed*.json`) replace the original ones."""
    runs = []
    for f in sorted(glob.glob(f"results/dfl/{pattern}")):
        r = json.load(open(f))
        fix = f.replace("_seed", "_fixnan_seed")
        if os.path.exists(fix):
            r["variants"].update(json.load(open(fix))["variants"])
        runs.append(r)
    return runs


def stat(runs, name, metric="asr"):
    # non-finite entries (e.g. a norm ratio over a zero victim update after a reset) count as missing
    v = [np.nanmean(np.where(np.isfinite(x := np.asarray(r["variants"][name][metric][-LAST:], float)), x, np.nan))
         for r in runs if name in r["variants"]]
    return (100 * np.mean(v), 100 * np.std(v)) if v else (np.nan, np.nan)


def cell(ms, digits=1, std=True):
    m, s = ms
    if np.isnan(m):
        return "--"
    return f"{m:.{digits}f}" + (f"\\,$\\pm$\\,{s:.{digits}f}" if std else "")


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n").write(text)


def table_main(metric, fname, caption, label):
    """Rows: dataset x attack; columns: aggregation rules. Mean only (std in the supplementary version)."""
    lines = ["\\begin{table*}[t]", f"\\caption{{{caption}}}\\label{{{label}}}", "\\centering\\small",
             "\\begin{tabular}{ll" + "r" * len(AGGS) + "}", "\\toprule",
             "Dataset & Attack & " + " & ".join(a for _, a in AGGS) + " \\\\", "\\midrule"]
    for d, (ds, dl) in enumerate(DATASETS):
        runs = load(f"{ds}_seed*.json")
        for i, (att, al) in enumerate(ATTACKS):
            row = [dl if i == 0 else "", al] + [cell(stat(runs, f"{att}|{g}", metric), std=False) for g, _ in AGGS]
            lines.append(" & ".join(row) + " \\\\")
        if d < len(DATASETS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write(fname, "\n".join(lines))


def table_stealth():
    lines = ["\\begin{table}[t]",
             "\\caption{Stealth statistics under mean aggregation: mean pairwise cosine between Sybil models "
             "(measured from the victim's previous model) and ratio of Sybil to honest update norm}\\label{tab:stealth}",
             "\\centering\\small", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{Fashion-MNIST} & \\multicolumn{2}{c}{CIFAR-10} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Attack & cos & norm & cos & norm & cos & norm \\\\", "\\midrule"]
    data = {ds: load(f"{ds}_seed*.json") for ds, _ in DATASETS}
    for att, al in ATTACKS[2:]:
        row = [al]
        for ds, _ in DATASETS:
            c = stat(data[ds], f"{att}|mean", "cos_sybil_ref_prev")
            n = stat(data[ds], f"{att}|mean", "norm_ratio")
            row += ["--" if np.isnan(x[0]) else f"{x[0] / 100:.2f}" for x in (c, n)]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    write("tab4_stealth.tex", "\n".join(lines))


def table_defense():
    """Clip + FoolsGold against the attacks, the best norm-constrained attacker, and partial eclipse."""
    rows = [("Epistemic Eclipse, full eclipse", "eclipse|{g}"),
            ("Norm-constrained, cap 1, $\\beta$=0.2", "eclipse-nc|{g}|cap=1.0|beta=0.2"),
            ("Norm-constrained, cap 1, $\\beta$=0.5", "eclipse-nc|{g}|cap=1.0|beta=0.5"),
            ("Norm-constrained, cap 2, $\\beta$=0.5", "eclipse-nc|{g}|cap=2.0|beta=0.5"),
            ("Norm-constrained, cap 2, $\\beta$=0.7", "eclipse-nc|{g}|cap=2.0|beta=0.7"),
            ("Identical Sybils, cap 1", "identical-nc|{g}|cap=1.0|beta=1.0")]
    lines = ["\\begin{table*}[t]",
             "\\caption{ASR (\\%) of the norm-constrained attacker under FoolsGold and Clip+FG. FG, FoolsGold; "
             "Clip+FG, clipping followed by FoolsGold; mean\\,$\\pm$\\,std over five seeds}\\label{tab:defense}",
             "\\centering\\small", "\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{Fashion-MNIST} & \\multicolumn{2}{c}{CIFAR-10} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Attacker & FG & Clip+FG & FG & Clip+FG & FG & Clip+FG \\\\", "\\midrule"]
    data = {ds: load(f"{ds}_seed*.json") for ds, _ in DATASETS}
    for lab, tpl in rows:
        row = [lab]
        for ds, _ in DATASETS:
            row += [cell(stat(data[ds], tpl.format(g=g))) for g in ("foolsgold", "clipfg")]
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write("tab5_defense.tex", "\n".join(lines))


def table_anchor():
    """Sybil models built on h_v (original) vs on x_v (anchored), by eclipse level: ASR (victim accuracy)."""
    levels = [("Full eclipse", ""), ("Fraction 0.75", "|frac=0.75"), ("Half eclipse", "|frac=0.5"),
              ("No eclipse", "|keep")]
    rules = [("mean", "Mean"), ("foolsgold", "FG"), ("clipfg", "Clip+FG")]
    lines = ["\\begin{table*}[t]",
             "\\caption{ASR and victim accuracy (\\%) with original and anchored Sybil models. Cells give ASR "
             "(accuracy). Sybil models are built on the victim's fresh model $h_v$ (orig.) or on its previous model "
             "$x_v$ (anch.). ``No eclipse'': "
             "five Sybils are added and all honest neighbours are kept. Mean over five seeds}\\label{tab:anchor}",
             "\\centering\\small", "\\begin{tabular}{ll" + "r" * 6 + "}", "\\toprule",
             " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{rl}}}" for _, rl in rules) + " \\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}",
             "Dataset & Eclipse & " + " & ".join(["orig.", "anch."] * 3) + " \\\\", "\\midrule"]
    for d, (ds, dl) in enumerate(DATASETS):
        runs = load(f"{ds}_anchor_seed*.json")
        for i, (ll, suf) in enumerate(levels):
            row = [dl if i == 0 else "", ll]
            for g, _ in rules:
                for pre in ("eclipse", "eclipse-xv"):
                    a, c = stat(runs, f"{pre}|{g}{suf}"), stat(runs, f"{pre}|{g}{suf}", "acc")
                    row.append("--" if np.isnan(a[0]) else f"{a[0]:.1f} ({c[0]:.0f})")
            lines.append(" & ".join(row) + " \\\\")
        if d < len(DATASETS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    write("tab6_anchor.tex", "\n".join(lines))


if __name__ == "__main__":
    table_main("asr", "tab2_asr.tex",
               "ASR (\\%) on the victim, mean over five seeds (last 10 rounds). $^\\dagger$Oracle knowledge of the "
               "victim's honest neighbours' updates", "tab:asr")
    table_main("acc", "tab3_acc.tex", "Victim test accuracy (\\%), mean over five seeds (last 10 rounds)", "tab:acc")
    table_stealth()
    table_defense()
    table_anchor()
    print("tables written to", OUT)
