#!/usr/bin/env bash
# Reproduces every result in the paper from scratch, then regenerates the tables, statistics and figures.
# One NVIDIA RTX 3060 (12 GB), one job at a time: about one week (the ResNet-20 runs take about 7 hours per seed).
# The per-seed result files are already included in results/dfl/, so the last block can be run alone.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
E="$PY -m src.dfl.engine"
S5="0 1 2 3 4"; S10="5 6 7 8 9"
CIFAR="--rounds 60 --lr 0.03 --local-steps 50"

# ---- Main benchmarks (MNIST, Fashion-MNIST, CIFAR-10), ten seeds: Tables 2-7, Figs. 4-8
# Seeds 0-4 were run before the finiteness check was added; the variants it affects (sign flip, targeted Min-Max,
# poison scale) were rerun with the check (*_fixnan_*) and replace the same variants of the main files.
# Seeds 5-9 were run with the check, so they need no rerun.
for S in "$S5" "$S10"; do
  for d in mnist fmnist; do
    $E --dataset $d --seeds $S --rounds 40 --grid full
    $E --dataset $d --seeds $S --rounds 40 --grid anchor --tag _anchor
  done
  $E --dataset cifar10 --seeds $S $CIFAR --grid reduced
  $E --dataset cifar10 --seeds $S $CIFAR --grid partial --tag _partial
  $E --dataset cifar10 --seeds $S $CIFAR --grid anchor --tag _anchor
  for a in 0.1 0.5 1.0 0; do  # heterogeneity sweep on MNIST (alpha 0 = IID; Fig. 6e)
    $E --dataset mnist --seeds $S --rounds 40 --grid hetero --alpha $a --tag _a$a
  done
done
for d in mnist fmnist; do
  $E --dataset $d --seeds $S5 --rounds 40 --grid fixnan --tag _fixnan
done
$E --dataset cifar10 --seeds $S5 $CIFAR --grid fixnan --tag _fixnan

# ---- Added aggregation rules against every attack, ten seeds (Table 7)
for d in mnist fmnist; do
  $E --dataset $d --seeds $S5 $S10 --rounds 40 --grid defense --tag _defense
done
$E --dataset cifar10 --seeds $S5 $S10 $CIFAR --grid defense --tag _defense

# ---- Other topologies and network sizes, five seeds (Table 9)
for d in mnist fmnist; do
  for t in er ws ring; do
    $E --dataset $d --seeds $S5 --rounds 40 --grid topo --topology $t --tag _topo_$t
  done
  for n in 100 200; do
    $E --dataset $d --seeds $S5 --rounds 40 --grid topo --nodes $n --tag _topo_n$n
  done
done

# ---- Added datasets and architectures, five seeds (Table 8)
bash scripts/download_datasets.sh            # FEMNIST, Shakespeare, Tiny ImageNet into data/raw/; CIFAR-100
$PY scripts/prepare_datasets.py              # writes data/processed/
$E --grid ext --seeds $S5 --tag _ext --dataset cifar100 $CIFAR
$E --grid ext --seeds $S5 --tag _ext --dataset femnist --partition natural --rounds 40 --lr 0.01 --local-steps 30
$E --grid ext_small --seeds $S5 --tag _ext --dataset shakespeare --partition natural --target 57 \
  --rounds 40 --lr 0.3 --local-steps 30
R20="--grid ext_small --seeds $S5 --rounds 150 --lr 0.03 --local-steps 50"
$E $R20 --dataset cifar10 --arch resnet20 --tag _r20_ext
$E $R20 --dataset cifar100 --arch resnet20 --tag _r20_ext
$E $R20 --dataset tinyimagenet --tag _ext     # Tiny ImageNet always uses ResNet-20

# ---- Tables (written to manuscript/tables/), statistics (appendix tables) and figures (written to figures/)
$PY figures/make_tables.py --revision
$PY figures/stats.py --ext
$PY figures/make_figures.py
$PY figures/make_diagrams.py
