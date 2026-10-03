"""Aggregates results/dfl/*.json over seeds into mean +/- std tables (final `last` rounds averaged per run).

Usage: python -m src.dfl.summarize "results/dfl/mnist_pilot_seed*.json"
"""
import glob
import json
import sys

import numpy as np

from src.dfl.engine import AGG_NAMES, ATTACKS

METRICS = ["asr", "acc", "cos_sybil", "cos_sybil_ref_prev", "cos_honest_ref_prev", "norm_ratio", "sybil_weight",
           "honest_weight", "self_weight"]


def load(pattern, last=10):
    runs = [json.load(open(f)) for f in sorted(glob.glob(pattern))]
    table = {}
    for name in runs[0]["variants"]:
        table[name] = {}
        for m in METRICS:
            per_run = [np.nanmean(r["variants"][name][m][-last:]) if not np.all(np.isnan(r["variants"][name][m][-last:]))
                       else np.nan for r in runs]
            table[name][m] = (float(np.mean(per_run)), float(np.std(per_run)))
    net = [np.mean(r["network"]["node_acc_mean"][-last:]) for r in runs]
    return runs, table, (float(np.mean(net)), float(np.std(net)))


def fmt(ms, pct=True):
    m, s = ms
    if np.isnan(m):
        return "  –  "
    return f"{100 * m:5.1f}±{100 * s:4.1f}" if pct else f"{m:5.2f}±{s:4.2f}"


if __name__ == "__main__":
    runs, table, net = load(sys.argv[1])
    print(f"runs: {len(runs)} | honest node accuracy (last 10 rounds): {fmt(net)} | seconds per run: "
          f"{np.mean([r['seconds'] for r in runs]):.0f}")
    for metric, title in (("asr", "ASR on victim (%)"), ("acc", "Victim accuracy (%)"),
                          ("sybil_weight", "Share of aggregation weight held by Sybils (%)")):
        print(f"\n{title}\n" + "attack".ljust(12) + "".join(g[:11].rjust(12) for g in AGG_NAMES))
        for a in ATTACKS:
            print(a.ljust(12) + "".join(fmt(table.get(f"{a}|{g}", {metric: (np.nan, 0)})[metric]).rjust(12)
                                        for g in AGG_NAMES))
    print("\nStealth statistics under mean aggregation (last 10 rounds)")
    print("attack".ljust(12) + "cos(Sybil,Sybil)".rjust(18) + "same, ref x_prev".rjust(18) + "norm ratio".rjust(14))
    for a in ATTACKS[1:]:
        t = table[f"{a}|mean"]
        print(a.ljust(12) + fmt(t["cos_sybil"], False).rjust(18) + fmt(t["cos_sybil_ref_prev"], False).rjust(18)
              + fmt(t["norm_ratio"], False).rjust(14))
    print("honest neighbours' pairwise cosine (no attack):", fmt(table["none|mean"]["cos_honest_ref_prev"], False))
    extra = [n for n in table if n.count("|") >= 2 or n.startswith("ablation")]
    if extra:
        print("\nSensitivity and ablation (Epistemic Eclipse)\n" + "variant".ljust(34) + "ASR".rjust(12) + "acc".rjust(12)
              + "Sybil w".rjust(12) + "honest w".rjust(12) + "norm ratio".rjust(12) + "cos prev".rjust(12))
        for n in extra:
            t = table[n]
            print(n.ljust(40) + fmt(t["asr"]).rjust(12) + fmt(t["acc"]).rjust(12) + fmt(t["sybil_weight"]).rjust(12)
                  + fmt(t["honest_weight"]).rjust(12) + fmt(t["norm_ratio"], False).rjust(12)
                  + fmt(t["cos_sybil_ref_prev"], False).rjust(12))
