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
}


# ----------------------------------------------------------------------------- data
def load_dataset(name: str, device: str, root: str = "./data"):
    """Returns normalized train/test tensors on `device` (no per-batch CPU transforms)."""
    cls = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}[name]
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

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def make_model(dataset: str) -> nn.Module:
    return CifarCNN() if dataset == "cifar10" else MnistCNN()


class Trainer:
    """Runs local SGD and evaluation for flat parameter vectors with one shared module instance."""

    def __init__(self, dataset, x_train, y_train, x_test, y_test, device, lr, batch_size, target_class, seed):
        self.net = make_model(dataset).to(device)
        self.x, self.y, self.xt, self.yt = x_train, y_train, x_test, y_test
        self.lr, self.bs, self.target, self.device = lr, batch_size, target_class, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.dim = parameters_to_vector(self.net.parameters()).numel()

    def init_flat(self) -> torch.Tensor:
        return parameters_to_vector(make_model_like(self.net).to(self.device).parameters()).detach().clone()

    def train(self, flat, idx, steps, lr=None, poison=False, momentum=0.9) -> torch.Tensor:
        """`steps` SGD steps from `flat` on samples `idx`; with poison=True every label is set to the target."""
        vector_to_parameters(flat.clone(), self.net.parameters())
        self.net.train()
        opt = torch.optim.SGD(self.net.parameters(), lr=lr or self.lr, momentum=momentum)
        for _ in range(steps):
            b = idx[torch.randint(len(idx), (self.bs,), device=self.device, generator=self.gen)]
            y = torch.full_like(self.y[b], self.target) if poison else self.y[b]
            opt.zero_grad()
            F.cross_entropy(self.net(self.x[b]), y).backward()
            opt.step()
        return parameters_to_vector(self.net.parameters()).detach().clone()

    @torch.no_grad()
    def evaluate(self, flat):
        """Returns (accuracy, attack success rate). ASR = share of non-target test samples predicted as target.
        A non-finite model returns NaN for both: argmax over NaN logits would return class 0 and fake an ASR of 1."""
        if not bool(torch.isfinite(flat).all()):
            return float("nan"), float("nan")
        vector_to_parameters(flat.clone(), self.net.parameters())
        self.net.eval()
        pred = torch.cat([self.net(self.xt[i:i + 5000]).argmax(1) for i in range(0, len(self.xt), 5000)])
        other = self.yt != self.target
        return (pred == self.yt).float().mean().item(), (pred[other] == self.target).float().mean().item()


def make_model_like(net: nn.Module) -> nn.Module:
    return type(net)()


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
