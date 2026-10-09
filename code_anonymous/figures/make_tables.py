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


# Wide (table*) tables: footnotesize, tighter columns, and scaled down to the text width only when still too wide
# (the referee layout is 372pt wide and the line numbers sit in the margin, so nothing may overhang).
WIDE_OPEN = ["\\centering\\footnotesize", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{%"]
WIDE_CLOSE = ["\\bottomrule", "\\end{tabular}}", "\\end{table*}", ""]


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n").write(text)


def table_main(metric, fname, caption, label):
    """Rows: dataset x attack; columns: aggregation rules. Mean only (std in the supplementary version)."""
    lines = ["\\begin{table*}[t]", f"\\caption{{{caption}}}\\label{{{label}}}", *WIDE_OPEN,
             "\\begin{tabular}{ll" + "r" * len(AGGS) + "}", "\\toprule",
             "Dataset & Attack & " + " & ".join(a for _, a in AGGS) + " \\\\", "\\midrule"]
    for d, (ds, dl) in enumerate(DATASETS):
        runs = load(f"{ds}_seed*.json")
        for i, (att, al) in enumerate(ATTACKS):
            row = [dl if i == 0 else "", al] + [cell(stat(runs, f"{att}|{g}", metric), std=False) for g, _ in AGGS]
            lines.append(" & ".join(row) + " \\\\")
        if d < len(DATASETS) - 1:
            lines.append("\\midrule")
    lines += WIDE_CLOSE
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
    partial = False
    for att, al in ATTACKS[2:]:
        row = [al]
        for ds, _ in DATASETS:
            for metric in ("cos_sybil_ref_prev", "norm_ratio"):
                # a seed whose last 10 rounds hold no finite value (victim reset every round) is left out and marked
                v = [np.nanmean(x[np.isfinite(x)]) for r in data[ds] if f"{att}|mean" in r["variants"]
                     for x in [np.asarray(r["variants"][f"{att}|mean"][metric][-LAST:], float)] if np.isfinite(x).any()]
                n_all = sum(f"{att}|mean" in r["variants"] for r in data[ds])
                partial |= 0 < len(v) < n_all
                row.append("--" if not v else f"{np.mean(v):.2f}" + ("$^\\ast$" if len(v) < n_all else ""))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    if partial:
        lines.append("\\par\\smallskip\\raggedright\\footnotesize $^\\ast$Nine of ten seeds; on the tenth the victim's "
                     "local training diverged throughout the last 10 rounds, which leaves the ratio undefined.")
    lines += ["\\end{table}", ""]
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
             "Clip+FG, clipping followed by FoolsGold; mean\\,$\\pm$\\,std over "
             + WORDS.get(len(load("mnist_seed*.json")), "five") + " seeds}\\label{tab:defense}",
             *WIDE_OPEN, "\\begin{tabular}{lrrrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{MNIST} & \\multicolumn{2}{c}{Fashion-MNIST} & \\multicolumn{2}{c}{CIFAR-10} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Attacker & FG & Clip+FG & FG & Clip+FG & FG & Clip+FG \\\\", "\\midrule"]
    data = {ds: load(f"{ds}_seed*.json") for ds, _ in DATASETS}
    for lab, tpl in rows:
        row = [lab]
        for ds, _ in DATASETS:
            row += [cell(stat(data[ds], tpl.format(g=g))) for g in ("foolsgold", "clipfg")]
        lines.append(" & ".join(row) + " \\\\")
    lines += WIDE_CLOSE
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
             "five Sybils are added and all honest neighbours are kept. Mean over "
             + WORDS.get(len(load("mnist_anchor_seed*.json")), "five") + " seeds}\\label{tab:anchor}",
             *WIDE_OPEN, "\\begin{tabular}{ll" + "r" * 6 + "}", "\\toprule",
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
    lines += WIDE_CLOSE
    write("tab6_anchor.tex", "\n".join(lines))


NEW_AGGS = [("mean", "Mean"), ("bulyan", "Bulyan"), ("rfa", "RFA"), ("signsgd", "signSGD"),
            ("cc_bucket", "CC+B"), ("fltrust", "FLTrust")]
EXT_AGGS = [("mean", "Mean"), ("trimmed", "TrMean"), ("median", "Median"), ("multikrum", "M-Krum"),
            ("bulyan", "Bulyan"), ("rfa", "RFA"), ("foolsgold", "FG"), ("clipfg", "Clip+FG"), ("signsgd", "signSGD"),
            ("cc_bucket", "CC+B"), ("fltrust", "FLTrust"), ("selfanchor", "Anchor")]
# Added datasets and architectures: (file prefix, label)
EXT_SETS = [("cifar100_ext", "CIFAR-100, CNN"), ("cifar100_r20_ext", "CIFAR-100, ResNet-20"),
            ("cifar10_r20_ext", "CIFAR-10, ResNet-20"), ("tinyimagenet_ext", "Tiny ImageNet, ResNet-20"),
            ("femnist_ext", "FEMNIST, CNN"), ("shakespeare_ext", "Shakespeare, LSTM")]
WORDS = {5: "five", 10: "ten"}


def table_grid(fname, caption, label, datasets, aggs, attacks, metric="asr"):
    """Generic attacks x rules table, one block per dataset, mean over seeds."""
    lines = ["\\begin{table*}[t]", f"\\caption{{{caption}}}\\label{{{label}}}", *WIDE_OPEN,
             "\\begin{tabular}{ll" + "r" * len(aggs) + "}", "\\toprule",
             "Dataset & Attack & " + " & ".join(a for _, a in aggs) + " \\\\", "\\midrule"]
    blocks = []
    for pattern, dl in datasets:
        runs = load(pattern)
        rows = []
        for att, al in attacks:
            cells = [cell(stat(runs, f"{att}|{g}", metric), std=False) for g, _ in aggs]
            if all(c == "--" for c in cells):  # attack not in this set's grid (reduced ResNet-20 grid)
                continue
            rows.append(" & ".join([dl if not rows else "", al] + cells) + " \\\\")
        if rows:  # a set without any result file yet gets no block (and no empty rule)
            blocks.append(rows)
    for i, rows in enumerate(blocks):
        lines += rows + (["\\midrule"] if i < len(blocks) - 1 else [])
    lines += WIDE_CLOSE
    write(fname, "\n".join(lines))


def table_newrules():
    n = len(load("mnist_defense_seed*.json"))
    table_grid("tab7_newrules.tex",
               f"ASR (\\%) under the five added aggregation rules, mean over {WORDS.get(n, n)} seeds. CC+B, centered "
               "clipping with bucketing; FLTrust uses a root dataset of 100 samples; $^\\dagger$oracle knowledge",
               "tab:newrules", [(f"{ds}_defense_seed*.json", dl) for ds, dl in DATASETS], NEW_AGGS, ATTACKS)


def table_ext():
    key = [a for a in ATTACKS if a[0] in ("none", "identical", "minmax-t", "eclipse")]
    sets = [(f"{p}_seed*.json", l) for p, l in EXT_SETS]
    r20 = ("ResNet-20 runs (150 rounds) and Shakespeare runs cover five attacks against seven rules; "
           "--, configuration not run")
    table_grid("tab8_ext.tex", "ASR (\\%) on the added datasets and architectures, mean over five seeds. "
               "Key attacks shown; all attacks in Appendix~\\ref{app:settings}. " + r20, "tab:ext", sets, EXT_AGGS, key)
    table_grid("tabA_ext_asr.tex", "ASR (\\%) of every attack on the added datasets and architectures, mean over "
               "five seeds. " + r20, "tab:ext_full", sets, EXT_AGGS, ATTACKS)
    table_grid("tabA_ext_acc.tex", "Victim accuracy (\\%) on the added datasets and architectures, mean over five "
               "seeds. " + r20, "tab:ext_acc", sets, EXT_AGGS, ATTACKS, metric="acc")


def table_topo():
    """Topologies and network sizes: ASR (victim accuracy) of key attack/rule pairs, mean over seeds."""
    topos = [("seed*", "BA, 50 nodes"), ("topo_er_seed*", "Erd\\H{o}s--R\\'enyi, 50"),
             ("topo_ws_seed*", "Watts--Strogatz, 50"), ("topo_ring_seed*", "Ring lattice, 50"),
             ("topo_n100_seed*", "BA, 100 nodes"), ("topo_n200_seed*", "BA, 200 nodes")]
    cols = [("eclipse|mean", "EE, Mean"), ("eclipse|median", "EE, Median"), ("eclipse|foolsgold", "EE, FG"),
            ("eclipse|rfa", "EE, RFA"), ("eclipse|clipfg", "EE, Clip+FG"), ("identical|foolsgold", "Ident., FG"),
            ("eclipse|mean|frac=0.5", "EE, Mean, $\\rho$=0.5")]
    lines = ["\\begin{table*}[t]",
             "\\caption{ASR (\\%) on other topologies and network sizes. Mean over five seeds (ten for the "
             "Barab\\'asi--Albert graph of 50 nodes, from the main runs); victim accuracy (\\%) in parentheses. EE, "
             "Epistemic Eclipse; BA, Barab\\'asi--Albert; $\\rho$, eclipse fraction}\\label{tab:topo}",
             *WIDE_OPEN, "\\begin{tabular}{ll" + "r" * len(cols) + "}", "\\toprule",
             "Dataset & Topology & " + " & ".join(c for _, c in cols) + " \\\\", "\\midrule"]
    for d, (ds, dl) in enumerate(DATASETS[:2]):
        for i, (suffix, tl) in enumerate(topos):
            runs = load(f"{ds}_{suffix}.json")
            if suffix == "seed*":  # BA-50 reference from the main runs (rfa from the defense runs)
                runs_rfa = load(f"{ds}_defense_seed*.json")
            row = [dl if i == 0 else "", tl]
            for name, _ in cols:
                src = runs_rfa if (suffix == "seed*" and "rfa" in name) else runs
                a, c = stat(src, name), stat(src, name, "acc")
                row.append("--" if np.isnan(a[0]) else f"{a[0]:.1f} ({c[0]:.0f})")
            lines.append(" & ".join(row) + " \\\\")
        if d == 0:
            lines.append("\\midrule")
    lines += WIDE_CLOSE
    write("tab9_topo.tex", "\n".join(lines))


if __name__ == "__main__":
    n = WORDS.get(len(load("mnist_seed*.json")), "five")
    table_main("asr", "tab2_asr.tex",
               f"ASR (\\%) on the victim, mean over {n} seeds (last 10 rounds). $^\\dagger$Oracle knowledge of the "
               "victim's honest neighbours' updates", "tab:asr")
    table_main("acc", "tab3_acc.tex", f"Victim test accuracy (\\%), mean over {n} seeds (last 10 rounds)", "tab:acc")
    table_stealth()
    table_defense()
    table_anchor()
    import sys
    if "--revision" in sys.argv:  # tables of the revision runs (paper/09_revision.md)
        table_newrules()
        table_topo()
        table_ext()
    print("tables written to", OUT)
