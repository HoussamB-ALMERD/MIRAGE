#!/usr/bin/env bash
# Heterogeneity sweep on MNIST (alpha 0.2 is covered by the main runs). Usage: bash scripts/run_hetero.sh 0.1 0.5
# alpha 0 means IID.
cd "$(dirname "$0")/.." || exit 1
for a in "$@"; do
  venv/bin/python -u -m src.dfl.engine --dataset mnist --seeds 0 1 2 3 4 --rounds 40 --grid hetero --alpha "$a" --tag "_a$a"
done
