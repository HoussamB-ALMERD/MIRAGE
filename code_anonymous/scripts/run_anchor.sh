#!/usr/bin/env bash
# Anchored-Sybil grid (outline 12.2) on MNIST and Fashion-MNIST, same settings as the main runs.
cd "$(dirname "$0")/.."
for d in mnist fmnist; do
  venv/bin/python -m src.dfl.engine --dataset "$d" --seeds 0 1 2 3 4 --rounds 40 --grid anchor --tag _anchor \
    > "results/dfl/${d}_anchor.log" 2>&1
done
echo ANCHOR_DONE
