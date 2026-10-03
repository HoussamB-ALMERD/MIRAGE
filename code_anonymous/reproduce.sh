#!/usr/bin/env bash
# Reproduces every result in the paper from scratch, then regenerates the tables and figures.
# One NVIDIA RTX 3060 (12 GB): about 19-31 minutes per seed and grid; the whole script takes about two days.
# The per-seed result files are already included in results/dfl/, so the last two lines can be run alone.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
SEEDS="0 1 2 3 4"
CIFAR="--rounds 60 --lr 0.03 --local-steps 50"

# Main grid, sensitivity, partial eclipse and adaptive attacker (MNIST, Fashion-MNIST; Tables 2-5, Figs. 4-8)
for d in mnist fmnist; do
  $PY -m src.dfl.engine --dataset $d --seeds $SEEDS --rounds 40 --grid full
done
# CIFAR-10: main grid, partial eclipse, adaptive attacker against the FoolsGold-based rules
$PY -m src.dfl.engine --dataset cifar10 --seeds $SEEDS $CIFAR --grid reduced
$PY -m src.dfl.engine --dataset cifar10 --seeds $SEEDS $CIFAR --grid partial --tag _partial
# Heterogeneity sweep on MNIST (alpha 0 = IID; Fig. 6e)
for a in 0.1 0.5 1.0 0; do
  $PY -m src.dfl.engine --dataset mnist --seeds $SEEDS --rounds 40 --grid hetero --alpha $a --tag _a$a
done
# Anchored Sybil models (Table 6)
for d in mnist fmnist; do
  $PY -m src.dfl.engine --dataset $d --seeds $SEEDS --rounds 40 --grid anchor --tag _anchor
done
$PY -m src.dfl.engine --dataset cifar10 --seeds $SEEDS $CIFAR --grid anchor --tag _anchor
# Variants rerun with the finiteness check (sign flip, targeted Min-Max, poison scale); they replace the
# same variants of the main files in the tables and figures
for d in mnist fmnist; do
  $PY -m src.dfl.engine --dataset $d --seeds $SEEDS --rounds 40 --grid fixnan --tag _fixnan
done
$PY -m src.dfl.engine --dataset cifar10 --seeds $SEEDS $CIFAR --grid fixnan --tag _fixnan

# Tables (written to manuscript/tables/) and figures (written to figures/)
$PY figures/make_tables.py
$PY figures/make_figures.py
$PY figures/make_diagrams.py
