"""Building blocks for the decentralized FL experiments: data, models, aggregation rules, attacks.

Everything works on flat parameter vectors held on one device, so that many victim variants
can be evaluated against a single honest training trajectory.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision import datasets

from src.models.mnist_cnn import MnistCNN

STATS = {
    "mnist": ((0.1307,), (0.3081,)),
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
}
NUM_CLASSES = {"mnist": 10, "fmnist": 10, "cifar10": 10, "cifar100": 100, "femnist": 62, "tinyimagenet": 200,
               "shakespeare": 80}
# Datasets prepared by scripts/prepare_datasets.py into data/processed/<name>.pt (natural splits for FEMNIST and
# Shakespeare; TinyImageNet kept as uint8 on the device and normalized per batch to save memory).
PROCESSED = ("femnist", "tinyimagenet", "shakespeare")


# ----------------------------------------------------------------------------- data
def load_processed(name: str, root: str = "./data") -> dict:
    return torch.load(f"{root}/processed/{name}.pt", weights_only=False)


def load_dataset(name: str, device: str, root: str = "./data"):
    """Returns normalized train/test tensors on `device` (no per-batch CPU transforms)."""
    if name in PROCESSED:
        p = load_processed(name, root)
        if name == "shakespeare":  # character indices, no normalization
            return [p[key].to(device) for key in ("x_train", "y_train", "x_test", "y_test")]
        if name == "tinyimagenet":  # uint8 NCHW; Trainer normalizes per batch with p["stats"]
            return [p[key].to(device) for key in ("x_train", "y_train", "x_test", "y_test")]
        mean, std = p["stats"]
        out = []
        for key in ("train", "test"):
            x = p[f"x_{key}"].float().unsqueeze(1) / 255.0
            out += [((x - mean) / std).to(device), p[f"y_{key}"].to(device)]
        return out
    cls = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10,
           "cifar100": datasets.CIFAR100}[name]
    out = []
    for train in (True, False):
        ds = cls(root=root, train=train, download=True)
        x = torch.as_tensor(np.asarray(ds.data), dtype=torch.float32) / 255.0
        x = x.unsqueeze(1) if x.dim() == 3 else x.permute(0, 3, 1, 2)
        mean, std = STATS[name]
        x = (x - torch.tensor(mean).view(1, -1, 1, 1)) / torch.tensor(std).view(1, -1, 1, 1)
        y = torch.as_tensor(np.asarray(ds.targets), dtype=torch.long)
        out += [x.to(device), y.to(device)]
    return out


def dirichlet_partition(labels: np.ndarray, num_nodes: int, alpha: float, rng: np.random.Generator,
                        aux_frac: float, min_size: int = 64):
    """Label-skew Dirichlet split. Returns per-node index arrays and the attacker's auxiliary indices."""
    idx = rng.permutation(len(labels))
    n_aux = int(aux_frac * len(labels))
    aux, rest = idx[:n_aux], idx[n_aux:]
    nodes = [[] for _ in range(num_nodes)]
    for c in np.unique(labels):
        c_idx = rest[labels[rest] == c]
        if alpha is None:  # IID
            props = np.full(num_nodes, 1.0 / num_nodes)
        else:
            props = rng.dirichlet(np.full(num_nodes, alpha))
        cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
        for node, part in zip(nodes, np.split(c_idx, cuts)):
            node.extend(part.tolist())
    for node in nodes:  # no node may be too small to draw a batch from
        if len(node) < min_size:
            node.extend(rng.choice(rest, size=min_size - len(node), replace=False).tolist())
    return [np.array(n) for n in nodes], aux


# --------------------------------------------------------------------------- models
class CifarCNN(nn.Module):
    """Small CNN without normalization buffers, so the parameter vector is the full model state."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


class FemnistCNN(MnistCNN):
    """The two-layer MNIST network with 62 outputs (digits, upper- and lower-case letters)."""

    def __init__(self, num_classes=62):
        super().__init__()
        self.fc2 = nn.Linear(128, num_classes)


class BasicBlockGN(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.n1 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.n2 = nn.GroupNorm(8, cout)
        self.sc = None if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.GroupNorm(8, cout))

    def forward(self, x):
        out = F.relu(self.n1(self.c1(x)))
        out = self.n2(self.c2(out))
        return F.relu(out + (x if self.sc is None else self.sc(x)))


class ResNet20GN(nn.Module):
    """ResNet-20 (He et al., 2016, CIFAR variant) with GroupNorm instead of BatchNorm, so that the model has no
    running statistics and the parameter vector is the full model state. stem_stride=2 for 64x64 inputs."""

    def __init__(self, num_classes=10, stem_stride=1):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, stem_stride, 1, bias=False), nn.GroupNorm(8, 16), nn.ReLU())
        layers, cin = [], 16
        for cout, stride in ((16, 1), (32, 2), (64, 2)):
            for b in range(3):
                layers.append(BasicBlockGN(cin, cout, stride if b == 0 else 1))
                cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


class CharLSTM(nn.Module):
    """Next-character model of the LEAF Shakespeare benchmark: 8-dim embedding, two LSTM layers of 256 units."""

    def __init__(self, num_classes=80):
        super().__init__()
        self.emb = nn.Embedding(num_classes, 8)
        self.lstm = nn.LSTM(8, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(out[:, -1])


def make_model(dataset: str, arch: str = "default") -> nn.Module:
    if arch == "resnet20" or dataset == "tinyimagenet":  # 64x64 inputs: the small CNN expects 32x32
        return ResNet20GN(NUM_CLASSES[dataset], stem_stride=2 if dataset == "tinyimagenet" else 1)
    if dataset == "shakespeare":
        return CharLSTM()
    if dataset == "femnist":
        return FemnistCNN()
    if dataset == "cifar100":
        return CifarCNN(NUM_CLASSES[dataset])
    return CifarCNN() if dataset == "cifar10" else MnistCNN()


class Trainer:
    """Runs local SGD and evaluation for flat parameter vectors with one shared module instance."""

    def __init__(self, dataset, x_train, y_train, x_test, y_test, device, lr, batch_size, target_class, seed,
                 arch="default"):
        self.dataset, self.arch = dataset, arch
        self.net = make_model(dataset, arch).to(device)
        self.x, self.y, self.xt, self.yt = x_train, y_train, x_test, y_test
        self.lr, self.bs, self.target, self.device = lr, batch_size, target_class, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.dim = parameters_to_vector(self.net.parameters()).numel()
        self.norm = None  # uint8 images (TinyImageNet) are normalized per batch
        if x_train.dtype == torch.uint8:
            mean, std = torch.load(f"./data/processed/{dataset}_stats.pt")
            self.norm = (mean.view(1, -1, 1, 1).to(device), std.view(1, -1, 1, 1).to(device))

    def _x(self, x):
        return x if self.norm is None else (x.float() / 255.0 - self.norm[0]) / self.norm[1]

    def init_flat(self) -> torch.Tensor:
        return parameters_to_vector(make_model(self.dataset, self.arch).to(self.device).parameters()).detach().clone()

    def _load(self, flat):
        """Loads a flat vector into the shared module. The LSTM is loaded in place: re-pointing its weights (as
        vector_to_parameters does) breaks cuDNN's contiguous weight buffer and makes every call very slow."""
        if isinstance(self.net, CharLSTM):
            with torch.no_grad():
                i = 0
                for p in self.net.parameters():
                    p.copy_(flat[i:i + p.numel()].view_as(p))
                    i += p.numel()
            self.net.lstm.flatten_parameters()
        else:
            vector_to_parameters(flat.clone(), self.net.parameters())

    def train(self, flat, idx, steps, lr=None, poison=False, momentum=0.9) -> torch.Tensor:
        """`steps` SGD steps from `flat` on samples `idx`; with poison=True every label is set to the target."""
        self._load(flat)
        self.net.train()
        opt = torch.optim.SGD(self.net.parameters(), lr=lr or self.lr, momentum=momentum)
        for _ in range(steps):
            b = idx[torch.randint(len(idx), (self.bs,), device=self.device, generator=self.gen)]
            y = torch.full_like(self.y[b], self.target) if poison else self.y[b]
            opt.zero_grad()
            F.cross_entropy(self.net(self._x(self.x[b])), y).backward()
            opt.step()
        return parameters_to_vector(self.net.parameters()).detach().clone()

    @torch.no_grad()
    def evaluate(self, flat):
        """Returns (accuracy, attack success rate). ASR = share of non-target test samples predicted as target.
        A non-finite model returns NaN for both: argmax over NaN logits would return class 0 and fake an ASR of 1."""
        if not bool(torch.isfinite(flat).all()):
            return float("nan"), float("nan")
        self._load(flat)
        self.net.eval()
        pred = torch.cat([self.net(self._x(self.xt[i:i + 5000])).argmax(1) for i in range(0, len(self.xt), 5000)])
        other = self.yt != self.target
        return (pred == self.yt).float().mean().item(), (pred[other] == self.target).float().mean().item()


# ---------------------------------------------------------------------- aggregation
def pairwise_cos(u: torch.Tensor) -> torch.Tensor:
    un = u / (u.norm(dim=1, keepdim=True) + 1e-12)
    return un @ un.T


def mean_offdiag(m: torch.Tensor) -> float:
    n = m.shape[0]
    return float("nan") if n < 2 else ((m.sum() - m.diag().sum()) / (n * (n - 1))).item()


def agg_mean(models):
    return models.mean(0), torch.ones(len(models), device=models.device)


def agg_trimmed_mean(models, trim=0.2):
    n = len(models)
    k = min(max(1, int(trim * n)), (n - 1) // 2) if n >= 3 else 0
    s = models.sort(0).values
    return s[k:n - k].mean(0), torch.ones(n, device=models.device)


def agg_median(models):
    return models.median(0).values, torch.ones(len(models), device=models.device)


def agg_multikrum(models):
    """Multi-Krum (Blanchard et al., 2017) with f set to the largest value the rule tolerates, f = (n - 3) // 2."""
    n = len(models)
    if n < 5:
        return agg_mean(models)
    f = (n - 3) // 2
    d = torch.cdist(models, models) ** 2
    d.fill_diagonal_(float("inf"))
    scores = d.sort(1).values[:, :n - f - 2].sum(1)
    keep = scores.argsort()[:n - f]
    w = torch.zeros(n, device=models.device)
    w[keep] = 1.0
    return models[keep].mean(0), w


def _krum_scores(models, f):
    n = len(models)
    d = torch.cdist(models, models) ** 2
    d.fill_diagonal_(float("inf"))
    return d.sort(1).values[:, :max(1, n - f - 2)].sum(1)


def agg_bulyan(models):
    """Bulyan (El Mhamdi et al., 2018) with f set to the largest value the rule tolerates, f = (n - 3) // 4:
    select theta = n - 2f models by repeated Krum, then average, per coordinate, the beta = theta - 2f selected
    values closest to the coordinate-wise median."""
    n = len(models)
    f = max(0, (n - 3) // 4)
    remaining, chosen = list(range(n)), []
    for _ in range(n - 2 * f):
        scores = _krum_scores(models[remaining], f) if len(remaining) > 1 else torch.zeros(1, device=models.device)
        chosen.append(remaining.pop(int(scores.argmin())))
    sel = models[chosen]
    beta = len(chosen) - 2 * f
    med = sel.median(0).values
    idx = (sel - med).abs().argsort(0)[:beta]
    w = torch.zeros(n, device=models.device)
    w[chosen] = 1.0
    return sel.gather(0, idx).mean(0), w


def agg_rfa(models, iters=10, nu=1e-6, tol=1e-6):
    """RFA (Pillutla et al., 2022): geometric median by the smoothed Weiszfeld algorithm."""
    z = models.mean(0)
    for _ in range(iters):
        w = 1.0 / (models - z).norm(dim=1).clamp(min=nu)
        z_new = (w.unsqueeze(1) * models).sum(0) / w.sum()
        done = (z_new - z).norm() <= tol * z.norm()
        z = z_new
        if done:
            break
    w = 1.0 / (models - z).norm(dim=1).clamp(min=nu)
    return z, w / w.sum() * len(models)


def agg_signsgd(upd, own_norm):
    """signSGD with majority vote (Bernstein et al., 2019) on updates: the step is the sign of the sum of the
    signs, scaled so that its norm equals the norm of the victim's own update."""
    vote = torch.sign(torch.sign(upd).sum(0))
    nz = vote.abs().sum().clamp(min=1)
    return vote * own_norm / nz.sqrt()


def agg_cc_bucketing(upd, center, tau, gen, bucket=2, iters=1):
    """Centered clipping (Karimireddy et al., 2021) applied to bucket means (Karimireddy et al., 2022):
    inputs are randomly grouped into buckets of `bucket` updates and averaged, then
    v <- v + mean_i clip_tau(b_i - v), starting from the previous aggregate update `center`."""
    perm = torch.randperm(len(upd), device=upd.device, generator=gen)
    b = torch.stack([upd[perm[i:i + bucket]].mean(0) for i in range(0, len(upd), bucket)])
    v = center.clone()
    for _ in range(iters):
        diff = b - v
        diff = diff * (tau / (diff.norm(dim=1, keepdim=True) + 1e-12)).clamp(max=1.0)
        v = v + diff.mean(0)
    return v


def foolsgold_weights(hist: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """FoolsGold (Fung et al., 2020) weights from historical aggregate updates, one row per participant."""
    n = hist.shape[0]
    cs = pairwise_cos(hist) - torch.eye(n, device=hist.device)
    maxcs = cs.max(1).values
    for i in range(n):  # pardoning
        for j in range(n):
            if i != j and maxcs[i] < maxcs[j]:
                cs[i, j] = cs[i, j] * maxcs[i] / maxcs[j]
    wv = (1 - cs.max(1).values).clamp(0, 1)
    wv = wv / (wv.max() + 1e-12)
    wv[wv == 1] = 0.99
    wv = kappa * (torch.log(wv / (1 - wv) + 1e-12) + 0.5)
    return wv.clamp(0, 1)


# -------------------------------------------------------------------------- attacks
def orthonormal_noise(dim, k, exclude: torch.Tensor, gen) -> torch.Tensor:
    """k orthonormal random directions, all orthogonal to the unit vector `exclude`. Shape [k, dim]."""
    g = torch.randn(dim, k, device=exclude.device, generator=gen)
    g = g - exclude.unsqueeze(1) * (exclude @ g).unsqueeze(0)
    q, _ = torch.linalg.qr(g)
    q = q - exclude.unsqueeze(1) * (exclude @ q).unsqueeze(0)
    return (q / q.norm(dim=0, keepdim=True)).T


def eclipse_updates(target: torch.Tensor, k: int, beta: float, scale: float, gen) -> torch.Tensor:
    """Epistemic Eclipse Sybil updates: scale/beta * ||t|| * (beta * t_hat + sqrt(1 - beta^2) * e_i).

    Pairwise cosine between Sybils is beta^2, each has cosine beta with the target direction, and
    their mean keeps the component scale * t regardless of beta.
    """
    t_norm = target.norm()
    t_hat = target / (t_norm + 1e-12)
    if beta >= 1.0:
        return (scale * target).repeat(k, 1)
    e = orthonormal_noise(target.numel(), k, t_hat, gen)
    if beta <= 0.0:  # ablation: noise only, same norm as the beta = 0.2 default
        return (scale / 0.2) * t_norm * e
    return (scale / beta) * t_norm * (beta * t_hat + (1 - beta ** 2) ** 0.5 * e)


def minmax_scale(mu, direction, benign) -> float:
    """Largest g such that mu + g * direction stays within the maximum pairwise distance of the benign set
    (Min-Max, Shejwalkar and Houmansadr, 2021)."""
    bound = torch.cdist(benign, benign).max()
    lo, hi = 0.0, 1.0
    while (benign - (mu + hi * direction)).norm(dim=1).max() <= bound and hi < 1e4:
        hi *= 2
    for _ in range(30):
        mid = (lo + hi) / 2
        if (benign - (mu + mid * direction)).norm(dim=1).max() <= bound:
            lo = mid
        else:
            hi = mid
    return lo
