#!/usr/bin/env bash
# Revision datasets (paper/09_revision.md, G2-G3): raw files into data/raw/. CIFAR-100 is fetched by torchvision.
set -e
cd "$(dirname "$0")/.."
mkdir -p data/raw
HF=https://huggingface.co/datasets
get() { [ -s "$2" ] || curl -sSL --retry 3 -o "$2" "$1"; ls -la "$2"; }
get "$HF/flwrlabs/femnist/resolve/main/data/train-00000-of-00001.parquet" data/raw/femnist.parquet
get "$HF/flwrlabs/shakespeare/resolve/main/shakespeare.csv" data/raw/shakespeare.csv
get "$HF/zh-plus/tiny-imagenet/resolve/main/data/train-00000-of-00001-1359597a978bc4fa.parquet" data/raw/tinyimagenet_train.parquet
get "$HF/zh-plus/tiny-imagenet/resolve/main/data/valid-00000-of-00001-70d52db3c749a935.parquet" data/raw/tinyimagenet_valid.parquet
${PYTHON:-python3} -c "from torchvision import datasets; [datasets.CIFAR100('./data', train=t, download=True) for t in (True, False)]"
