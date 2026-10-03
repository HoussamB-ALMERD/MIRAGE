#!/usr/bin/env bash
# Rerun the variants whose victim diverged to NaN (sign flip, targeted Min-Max, poison scale 0.5-3) with the
# finiteness check in engine.py. Results go to *_fixnan_seed*.json and override the same variants in the tables.
cd "$(dirname "$0")/.."
while pgrep -f "grid anchor" > /dev/null; do sleep 30; done
for d in mnist fmnist; do
  venv/bin/python -m src.dfl.engine --dataset "$d" --seeds 0 1 2 3 4 --rounds 40 --grid fixnan --tag _fixnan \
    > "results/dfl/${d}_fixnan.log" 2>&1
done
venv/bin/python -m src.dfl.engine --dataset cifar10 --seeds 0 1 2 3 4 --rounds 60 --lr 0.03 --local-steps 50 \
  --grid fixnan --tag _fixnan > results/dfl/cifar10_fixnan.log 2>&1
echo FIXNAN_DONE
