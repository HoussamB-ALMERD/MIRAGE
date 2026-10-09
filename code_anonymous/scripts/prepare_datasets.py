"""Convert the revision datasets (scripts/download_datasets.sh) into tensors in data/processed/.

FEMNIST and Shakespeare keep their natural client structure (LEAF; Caldas et al., 2018): one DFL node per writer or
per speaking role, selected once with a fixed generator, so every seed sees the same 50 clients. The attacker's
auxiliary data (2% of the nodes' pooled training data) comes from writers or roles that are not in the network.
TinyImageNet is stored as uint8 and split among nodes by the engine with the usual Dirichlet label skew.

Usage (from the project root): venv/bin/python scripts/prepare_datasets.py [femnist] [shakespeare] [tinyimagenet]
"""
import collections
import csv
import io
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

RAW, OUT = "data/raw", "data/processed"
NODES, AUX_FRAC, SEED = 50, 0.02, 2026
# LEAF's 80-character vocabulary for Shakespeare (leaf/models/utils/language_utils.py).
LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"


def decode(rows, mode, size=None):
    out = []
    for r in rows:
        im = Image.open(io.BytesIO(r["bytes"])).convert(mode)
        if size and im.size != size:
            im = im.resize(size)
        out.append(np.asarray(im, dtype=np.uint8))
    return np.stack(out)


def femnist():
    t = pq.read_table(f"{RAW}/femnist.parquet", columns=["writer_id", "character"])
    writers = np.asarray(t.column("writer_id").to_pylist())
    labels = np.asarray(t.column("character").to_pylist())
    rng = np.random.default_rng(SEED)
    counts = collections.Counter(writers)
    eligible = sorted(w for w, c in counts.items() if c >= 200)
    chosen = list(rng.choice(eligible, NODES, replace=False))
    rows_tr, rows_te, parts = [], [], []
    for w in chosen:
        idx = rng.permutation(np.flatnonzero(writers == w))
        cut = int(0.8 * len(idx))
        parts.append(np.arange(len(rows_tr), len(rows_tr) + cut))
        rows_tr += idx[:cut].tolist()
        rows_te += idx[cut:].tolist()
    n_aux = int(AUX_FRAC * len(rows_tr))
    others = rng.permutation(np.flatnonzero(~np.isin(writers, chosen)))[:n_aux]
    aux = np.arange(len(rows_tr), len(rows_tr) + n_aux)
    rows_tr += others.tolist()
    img = pq.read_table(f"{RAW}/femnist.parquet", columns=["image"]).column("image")

    def take(rows):
        return torch.as_tensor(decode(img.take(rows).to_pylist(), "L", (28, 28)))
    x_tr, x_te = take(rows_tr), take(rows_te)
    xf = x_tr.float() / 255.0
    save("femnist", {"x_train": x_tr, "y_train": torch.as_tensor(labels[rows_tr]), "x_test": x_te,
                     "y_test": torch.as_tensor(labels[rows_te]), "parts": parts, "aux": aux,
                     "stats": (xf.mean().item(), xf.std().item()), "writers": chosen})


def shakespeare(cap_train=3000, cap_test=5000):
    by_role = collections.defaultdict(list)
    with open(f"{RAW}/shakespeare.csv", newline="") as f:
        for role, x, y in csv.reader(f):
            if role != "character_id":
                by_role[role].append((x, y))
    idx = {c: i for i, c in enumerate(LETTERS)}
    enc = lambda s: [idx.get(c, 1) for c in s]  # unknown characters map to the space (index 1)
    rng = np.random.default_rng(SEED)
    eligible = sorted(r for r, v in by_role.items() if len(v) >= 2000)
    chosen = list(rng.choice(eligible, NODES, replace=False))
    xtr, ytr, xte, yte, parts = [], [], [], [], []
    for r in chosen:
        s = by_role[r]
        order = rng.permutation(len(s))
        cut = int(0.9 * len(s))
        tr, te = order[:cut][:cap_train], order[cut:]
        parts.append(np.arange(len(xtr), len(xtr) + len(tr)))
        xtr += [enc(s[i][0]) for i in tr]
        ytr += [idx.get(s[i][1], 1) for i in tr]
        xte += [enc(s[i][0]) for i in te]
        yte += [idx.get(s[i][1], 1) for i in te]
    keep = rng.permutation(len(xte))[:cap_test]
    xte, yte = [xte[i] for i in keep], [yte[i] for i in keep]
    n_aux = int(AUX_FRAC * len(xtr))
    pool = [(r, i) for r in sorted(set(eligible) - set(chosen)) for i in range(len(by_role[r]))]
    aux = np.arange(len(xtr), len(xtr) + n_aux)
    for j in rng.choice(len(pool), n_aux, replace=False):
        r, i = pool[j]
        xtr.append(enc(by_role[r][i][0]))
        ytr.append(idx.get(by_role[r][i][1], 1))
    save("shakespeare", {"x_train": torch.tensor(xtr), "y_train": torch.tensor(ytr), "x_test": torch.tensor(xte),
                         "y_test": torch.tensor(yte), "parts": parts, "aux": aux, "roles": chosen,
                         "vocab": LETTERS})


def tinyimagenet():
    out = {}
    for split, key in (("train", "train"), ("valid", "test")):
        t = pq.read_table(f"{RAW}/tinyimagenet_{split}.parquet")
        x = decode(t.column("image").to_pylist(), "RGB", (64, 64))
        out[f"x_{key}"] = torch.as_tensor(x).permute(0, 3, 1, 2).contiguous()
        out[f"y_{key}"] = torch.as_tensor(np.asarray(t.column("label").to_pylist()))
    xf = out["x_train"][::10].float() / 255.0
    out["stats"] = (xf.mean((0, 2, 3)), xf.std((0, 2, 3)))
    save("tinyimagenet", out)


def save(name, d):
    os.makedirs(OUT, exist_ok=True)
    torch.save(d, f"{OUT}/{name}.pt")
    if "stats" in d:
        torch.save(d["stats"], f"{OUT}/{name}_stats.pt")
    sizes = [len(p) for p in d.get("parts", [])]
    print(name, "train", tuple(d["x_train"].shape), "test", tuple(d["x_test"].shape),
          "classes", int(d["y_train"].max()) + 1, "aux", len(d.get("aux", [])),
          "node sizes min/median/max", (min(sizes), int(np.median(sizes)), max(sizes)) if sizes else "-")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["femnist", "shakespeare", "tinyimagenet"]:
        globals()[name]()
