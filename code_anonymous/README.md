# Epistemic Eclipse — code and data (anonymised for review)

Code and per-seed results for the manuscript "Epistemic Eclipse: decorrelated Sybil collusion against isolated
nodes in decentralized federated learning".

## Contents

| Path | What it is |
|---|---|
| `src/dfl/engine.py` | Decentralized FL simulator: honest network, eclipse, attacks, aggregation rules at the victim, experiment grids |
| `src/dfl/core.py` | Data loading and Dirichlet partition, models, local training, evaluation, aggregation rules |
| `src/models/mnist_cnn.py` | CNN used for MNIST and Fashion-MNIST |
| `scripts/download_datasets.sh`, `scripts/prepare_datasets.py` | Download FEMNIST, Shakespeare, Tiny ImageNet and CIFAR-100, and convert them to tensors |
| `results/dfl/*_seed*.json` | Per-round logs of every victim configuration: ten seeds for the main benchmarks, five for the other settings |
| `figures/make_tables.py`, `figures/make_figures.py` | Produce the result tables and Figs. 4-8 from the result files |
| `figures/stats.py` | Paired tests (t-test, Wilcoxon, Holm correction) and 95% confidence intervals (appendix tables) |
| `figures/make_diagrams.py` | Produces Figs. 1-3 |
| `reproduce.sh` | Reruns every experiment and regenerates all tables and figures |

## Requirements

Python 3.10 or later, the packages in `requirements.txt` (a CUDA build of PyTorch is recommended), and
internet access on the first run to download MNIST, Fashion-MNIST and CIFAR-10 through torchvision and the other
datasets through `scripts/download_datasets.sh`.

## Regenerate the tables and figures from the included results

```bash
pip install -r requirements.txt
python3 figures/make_tables.py --revision   # writes manuscript/tables/*.tex
python3 figures/stats.py --ext              # writes the statistical tables
python3 figures/make_figures.py             # writes figures/fig4-fig8 (.pdf, .png)
```

## Rerun the experiments

```bash
bash reproduce.sh
```

A single run, for example the main MNIST grid for one seed:

```bash
python3 -m src.dfl.engine --dataset mnist --seeds 0 --rounds 40 --grid full
```

Each result file holds, for every variant (named `attack|rule[|parameter]`), per-round lists of the victim's
attack success rate (`asr`), clean accuracy (`acc`), the weight its rule gave the Sybils, the mean pairwise
cosine between Sybil updates, the Sybil-to-honest norm ratio, and whether the finiteness check fired
(`diverged`).

## Licence

Released for review. A licence and a persistent identifier will be added with the public release.
