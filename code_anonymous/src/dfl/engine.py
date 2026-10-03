"""Decentralized FL engine: one honest trajectory, many victim variants evaluated in the same pass.

Protocol (per round): every active node runs local SGD, then replaces its model by the average of its own
and its active neighbours' locally trained models. The victim is excluded from the honest nodes'
neighbourhoods, so the honest trajectory does not depend on the attack and every (attack, aggregation
rule, parameter) variant of the victim can be computed against the same honest models.
"""
import dataclasses
import json
import os
import time
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import torch

from src.dfl import core


@dataclasses.dataclass
class RunConfig:
    dataset: str = "mnist"
    seed: int = 0
    num_nodes: int = 50          # honest nodes, including the victim
    ba_m: int = 3
    alpha: Optional[float] = 0.2  # Dirichlet label skew; None = IID
    aux_frac: float = 0.02       # attacker's auxiliary data, disjoint from honest nodes
    churn: float = 0.10
    rounds: int = 40
    local_steps: int = 30
    batch_size: int = 32
    lr: float = 0.01
    target_class: int = 0
    victim_degree: int = 6       # victim = first node with this degree (closest above otherwise)
    poison_steps: int = 10       # attacker's SGD steps on label-flipped auxiliary data
    poison_lr: float = 0.05
    benign_est: int = 4          # auxiliary honest models the attacker trains to estimate benign statistics
    benign_est_steps: int = 10
    eval_nodes: int = 10         # honest nodes sampled for the network accuracy
    eval_every: int = 1


@dataclasses.dataclass
class Variant:
    """One victim configuration. `frac` is the share of Sybils among the victim's neighbours."""
    name: str
    attack: str = "eclipse"      # none | signflip | identical | correlated | lie | minmax | eclipse
    agg: str = "mean"            # mean | trimmed | median | multikrum | foolsgold
    k: int = 5
    beta: float = 0.2
    scale: float = 1.0           # size of the mean poison displacement in units of the poison direction
    frac: float = 1.0
    keep_honest: bool = False    # ablation: Sybils are added without removing honest neighbours
    norm_cap: Optional[float] = None  # adaptive attacker: ||m_i - x_v|| <= norm_cap * ||victim's own update||
    anchor_prev: bool = False    # Sybil models built around x_v instead of h_v, so they do not share the victim's update


AGGS = {"mean": core.agg_mean, "trimmed": core.agg_trimmed_mean, "median": core.agg_median,
        "multikrum": core.agg_multikrum}
LOG_KEYS = ("acc", "asr", "cos_sybil", "cos_sybil_ref_prev", "cos_honest_ref_prev", "norm_ratio", "sybil_weight",
            "honest_weight", "self_weight", "diverged")


class Engine:
    def __init__(self, cfg: RunConfig, variants: List[Variant], device: Optional[str] = None):
        self.cfg, self.variants = cfg, variants
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        xtr, ytr, xte, yte = core.load_dataset(cfg.dataset, self.device)
        parts, aux = core.dirichlet_partition(ytr.cpu().numpy(), cfg.num_nodes, cfg.alpha, self.rng, cfg.aux_frac)
        self.parts = [torch.as_tensor(p, device=self.device) for p in parts]
        self.aux = torch.as_tensor(aux, device=self.device)
        self.tr = core.Trainer(cfg.dataset, xtr, ytr, xte, yte, self.device, cfg.lr, cfg.batch_size,
                               cfg.target_class, cfg.seed)
        self.gen = torch.Generator(device=self.device).manual_seed(cfg.seed + 1)

        graph = nx.barabasi_albert_graph(cfg.num_nodes, cfg.ba_m, seed=cfg.seed)
        degs = dict(graph.degree())
        cands = sorted(degs, key=lambda n: (abs(degs[n] - cfg.victim_degree) + (100 if degs[n] < cfg.victim_degree else 0), n))
        self.victim = cands[0]
        self.victim_nb = sorted(graph.neighbors(self.victim))
        graph.remove_node(self.victim)
        self.nb = {n: sorted(graph.neighbors(n)) for n in graph.nodes}
        self.honest = sorted(graph.nodes)

        init = self.tr.init_flat()
        self.x = {n: init.clone() for n in self.honest}
        self.state: Dict[str, dict] = {}
        for v in variants:
            self.state[v.name] = {"x": init.clone(), "hist": {}, "prev": {}, "log": {key: [] for key in LOG_KEYS}}
        self.net_log = {"round": [], "node_acc_mean": [], "node_acc_std": [], "consensus_acc": []}

    # ------------------------------------------------------------------ victim side
    def _honest_ids(self, v: Variant) -> List[int]:
        if v.attack == "none" or v.keep_honest:
            return self.victim_nb
        n_honest = int(round(v.k * (1 - v.frac) / v.frac)) if v.frac < 1 else 0
        return self.victim_nb[:min(n_honest, len(self.victim_nb))]

    def _sybil_updates(self, v, hv, xv, half, active) -> Optional[torch.Tensor]:
        """Sybil updates relative to the victim's freshly trained model hv (rushing adversary). Shape [k, d]."""
        if v.attack == "none":
            return None
        if v.norm_cap is not None and v.attack in ("eclipse", "identical"):
            # Norm-constrained attacker: every Sybil model lies exactly at distance cap * ||hv - xv|| from the
            # victim's previous model xv, decorrelated around the direction of the poisoned destination hv + t.
            c = self.cfg
            t = self.tr.train(hv, self.aux, c.poison_steps, lr=c.poison_lr, poison=True, momentum=0.0) - hv
            u = hv - xv
            dest = u + v.scale * t
            beta = v.beta if v.attack == "eclipse" else 1.0
            unit = dest / (dest.norm() + 1e-12) * (v.norm_cap * u.norm())
            return core.eclipse_updates(unit, v.k, beta, beta, self.gen) - u
        if v.anchor_prev and v.attack == "eclipse":  # m_i = x_v + eclipse update, i.e. relative to h_v: e_i - u
            return self._raw_sybil_updates(v, hv, half, active) - (hv - xv)
        return self._raw_sybil_updates(v, hv, half, active)

    def _raw_sybil_updates(self, v, hv, half, active) -> torch.Tensor:
        c = self.cfg
        t = self.tr.train(hv, self.aux, c.poison_steps, lr=c.poison_lr, poison=True, momentum=0.0) - hv
        if v.attack == "eclipse":
            return core.eclipse_updates(t, v.k, v.beta, v.scale, self.gen)
        if v.attack == "identical":
            return (v.scale * t).repeat(v.k, 1)
        if v.attack == "correlated":
            noise = torch.randn(v.k, t.numel(), device=t.device, generator=self.gen)
            noise = 0.3 * v.scale * t.norm() * noise / noise.norm(dim=1, keepdim=True)
            return v.scale * t + noise
        # Baselines that need benign statistics get oracle knowledge: the true updates of the victim's
        # original honest neighbours (more than Epistemic Eclipse uses). Fallback: auxiliary honest models.
        ids = [j for j in self.victim_nb if j in active]
        if len(ids) >= 2:
            b = torch.stack([half[j] - hv for j in ids])
        else:
            b = torch.stack([self.tr.train(hv, self.aux, c.benign_est_steps) - hv for _ in range(c.benign_est)])
        mu, sd = b.mean(0), b.std(0)
        if v.attack == "signflip":
            return (-v.scale * t.norm() / (mu.norm() + 1e-12) * mu).repeat(v.k, 1)
        if v.attack == "lie":
            return (mu - 1.5 * sd).repeat(v.k, 1)
        if v.attack in ("minmax", "minmax-t"):
            direction = t / (t.norm() + 1e-12) if v.attack == "minmax-t" else -mu / (mu.norm() + 1e-12)
            return (mu + core.minmax_scale(mu, direction, b) * direction).repeat(v.k, 1)
        raise ValueError(v.attack)

    def _victim_round(self, v: Variant, half: Dict[int, torch.Tensor], active: set):
        st = self.state[v.name]
        xv = st["x"]
        hv = self.tr.train(xv, self.parts[self.victim], self.cfg.local_steps)
        # Sanity check every real client has: never adopt a non-finite model (diverged local training or aggregate).
        diverged = not bool(torch.isfinite(hv).all())
        if diverged:
            hv = xv.clone()
        ids = [j for j in self._honest_ids(v) if j in active]
        honest = [half[j] for j in ids]
        syb = self._sybil_updates(v, hv, xv, half, active)
        n_syb = 0 if syb is None else len(syb)
        models = torch.stack([hv] + honest + ([hv + s for s in syb] if n_syb else []))
        keys = ["self"] + [f"h{j}" for j in ids] + [f"s{i}" for i in range(n_syb)]
        is_syb = torch.tensor([key.startswith("s") and key != "self" for key in keys], device=self.device)

        if v.agg == "foolsgold":
            upd = models - xv
            for key, u in zip(keys, upd):
                st["hist"][key] = st["hist"].get(key, 0) + u
            w = core.foolsgold_weights(torch.stack([st["hist"][key] for key in keys]))
            new = xv + (w.unsqueeze(1) * upd).sum(0) / w.sum() if w.sum() > 0 else hv
        elif v.agg == "foolsgold_delta":  # history of each participant's own model changes
            for key, m in zip(keys, models):
                st["hist"][key] = st["hist"].get(key, 0) + (m - st["prev"].get(key, xv))
                st["prev"][key] = m
            w = core.foolsgold_weights(torch.stack([st["hist"][key] for key in keys]))
            new = (w.unsqueeze(1) * models).sum(0) / w.sum() if w.sum() > 0 else hv
        elif v.agg == "clipfg":  # norm clipping to the victim's own update norm, then FoolsGold on the clipped updates
            upd = models - xv
            upd = upd * (upd[0].norm() / (upd.norm(dim=1, keepdim=True) + 1e-12)).clamp(max=1.0)
            for key, u in zip(keys, upd):
                st["hist"][key] = st["hist"].get(key, 0) + u
            w = core.foolsgold_weights(torch.stack([st["hist"][key] for key in keys]))
            new = xv + (w.unsqueeze(1) * upd).sum(0) / w.sum() if w.sum() > 0 else hv
        elif v.agg in ("normclip", "selfanchor"):
            upd = models - xv
            own = upd[0]
            norms = upd.norm(dim=1, keepdim=True) + 1e-12
            if v.agg == "normclip":  # clip every received update to the norm of the victim's own update
                upd = upd * (own.norm() / norms).clamp(max=1.0)
                w = torch.ones(len(models), device=self.device)
            else:  # FLTrust-style: trust = ReLU(cos(update, own update)), updates rescaled to the own norm
                w = torch.relu((upd @ own) / (norms.squeeze(1) * own.norm() + 1e-12))
                upd = upd * (own.norm() / norms)
            new = xv + (w.unsqueeze(1) * upd).sum(0) / w.sum()
        else:
            new, w = AGGS[v.agg](models)
        if not bool(torch.isfinite(new).all()):
            new, diverged = hv, True
        st["x"] = new

        log = st["log"]
        log["diverged"].append(float(diverged))
        acc, asr = self.tr.evaluate(new)
        log["acc"].append(acc)
        log["asr"].append(asr)
        nan = float("nan")
        ref_prev = models - xv
        log["cos_sybil"].append(core.mean_offdiag(core.pairwise_cos(syb)) if n_syb > 1 else nan)
        log["cos_sybil_ref_prev"].append(core.mean_offdiag(core.pairwise_cos(ref_prev[is_syb])) if n_syb > 1 else nan)
        log["cos_honest_ref_prev"].append(core.mean_offdiag(core.pairwise_cos(ref_prev[1:1 + len(ids)])) if len(ids) > 1 else nan)
        base = ref_prev[1:1 + len(ids)].norm(dim=1).mean() if ids else ref_prev[0].norm()
        log["norm_ratio"].append((ref_prev[is_syb].norm(dim=1).mean() / base).item() if n_syb else nan)
        ok = w.sum() > 0
        log["sybil_weight"].append((w[is_syb].sum() / w.sum()).item() if n_syb and ok else nan)
        log["honest_weight"].append((w[1:1 + len(ids)].sum() / w.sum()).item() if ids and ok else nan)
        log["self_weight"].append((w[0] / w.sum()).item() if ok else nan)

    # ------------------------------------------------------------------- main loop
    def run(self, out_path: Optional[str] = None, verbose: bool = True) -> dict:
        c = self.cfg
        t0 = time.time()
        for r in range(1, c.rounds + 1):
            active = {n for n in self.honest if self.rng.random() >= c.churn}
            half = {n: self.tr.train(self.x[n], self.parts[n], c.local_steps) for n in active}
            for v in self.variants:
                self._victim_round(v, half, active)
            for n in active:
                group = [half[n]] + [half[j] for j in self.nb[n] if j in active]
                self.x[n] = torch.stack(group).mean(0)
            if r % c.eval_every == 0 or r == c.rounds:
                sample = self.rng.choice(self.honest, size=min(c.eval_nodes, len(self.honest)), replace=False)
                accs = [self.tr.evaluate(self.x[int(n)])[0] for n in sample]
                cons = self.tr.evaluate(torch.stack(list(self.x.values())).mean(0))[0]
                self.net_log["round"].append(r)
                self.net_log["node_acc_mean"].append(float(np.mean(accs)))
                self.net_log["node_acc_std"].append(float(np.std(accs)))
                self.net_log["consensus_acc"].append(cons)
            if verbose and (r % 5 == 0 or r == c.rounds):
                print(f"[{c.dataset} seed {c.seed}] round {r:3d}/{c.rounds} | node acc {self.net_log['node_acc_mean'][-1]:.3f}"
                      f" | {time.time() - t0:6.0f}s", flush=True)
        result = {"config": dataclasses.asdict(c), "victim": int(self.victim), "victim_degree": len(self.victim_nb),
                  "seconds": time.time() - t0, "network": self.net_log,
                  "variants": {v.name: {"spec": dataclasses.asdict(v), **self.state[v.name]["log"]} for v in self.variants}}
        if out_path:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(result, f)
        return result


ATTACKS = ["none", "signflip", "identical", "correlated", "lie", "minmax", "minmax-t", "eclipse"]
AGG_NAMES = ["mean", "trimmed", "median", "multikrum", "foolsgold", "foolsgold_delta", "normclip", "clipfg",
             "selfanchor"]


def main_grid() -> List[Variant]:
    return [Variant(name=f"{a}|{g}", attack=a, agg=g) for a in ATTACKS for g in AGG_NAMES]


def adaptive_grid() -> List[Variant]:
    """Norm-constrained (adaptive) attackers against the norm- and similarity-based rules."""
    out = []
    for g in ("mean", "foolsgold", "normclip", "clipfg"):
        for cap in (1.0, 1.5, 2.0):
            out += [Variant(f"eclipse-nc|{g}|cap={cap}|beta={b}", attack="eclipse", agg=g, beta=b, norm_cap=cap)
                    for b in (0.2, 0.5, 0.7, 0.9)]
        out += [Variant(f"identical-nc|{g}|cap={cap}|beta=1.0", attack="identical", agg=g, norm_cap=cap)
                for cap in (1.0, 2.0)]
    return out


def partial_grid() -> List[Variant]:
    """Partial eclipse under the two FoolsGold variants that were not covered by the sensitivity grid."""
    out = []
    for g in ("foolsgold_delta", "clipfg"):
        out += [Variant(f"eclipse|{g}|frac={f}", agg=g, frac=f) for f in (0.5, 0.75)]
        out += [Variant(f"ablation-no-eclipse|{g}", agg=g, keep_honest=True)]
    return out


def sensitivity_grid(aggs=("mean", "foolsgold")) -> List[Variant]:
    out = []
    for g in aggs:
        out += [Variant(f"eclipse|{g}|k={k}", agg=g, k=k) for k in (1, 3, 8, 12)]
        out += [Variant(f"eclipse|{g}|beta={b}", agg=g, beta=b) for b in (0.05, 0.1, 0.3, 0.5)]
        out += [Variant(f"eclipse|{g}|frac={f}", agg=g, frac=f) for f in (0.5, 0.6, 0.75, 0.9)]
        out += [Variant(f"eclipse|{g}|scale={s}", agg=g, scale=s) for s in (0.5, 2.0, 3.0)]
        out += [Variant(f"ablation-noise-only|{g}", agg=g, beta=0.0),
                Variant(f"ablation-no-eclipse|{g}", agg=g, keep_honest=True)]
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--rounds", type=int, default=40)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--grid", default="full",
                   choices=["none", "main", "reduced", "hetero", "partial", "anchor", "fixnan", "full"])
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--local-steps", type=int, default=30)
    p.add_argument("--out", default="results/dfl")
    p.add_argument("--tag", default="")
    a = p.parse_args()
    for seed in a.seeds:
        variants = main_grid() + (sensitivity_grid() + partial_grid() + adaptive_grid() if a.grid == "full" else [])
        if a.grid == "reduced":  # CIFAR-10: main grid, partial eclipse, adaptive attacker against the two FoolsGold-based rules
            variants = main_grid() + partial_grid() + [v for v in adaptive_grid() if v.agg in ("foolsgold", "clipfg")
                                                       and v.norm_cap in (1.0, 2.0) and v.beta in (0.2, 0.5, 0.7, 1.0)]
        if a.grid == "hetero":  # heterogeneity sweep: key attacks against key rules
            variants = [Variant(f"{att}|{g}", attack=att, agg=g) for att in ("none", "identical", "minmax-t", "eclipse")
                        for g in ("mean", "median", "foolsgold", "normclip", "clipfg")]
        if a.grid == "partial":  # partial eclipse under mean and FoolsGold (missing from the reduced CIFAR-10 grid)
            variants = [Variant(f"eclipse|{g}|frac={f}", agg=g, frac=f) for g in ("mean", "foolsgold") for f in (0.5, 0.75)]
            variants += [Variant(f"ablation-no-eclipse|{g}", agg=g, keep_honest=True) for g in ("mean", "foolsgold")]
        if a.grid == "anchor":  # Sybil models anchored at x_v, next to the h_v-anchored originals, full and partial eclipse
            variants = []
            for g in ("mean", "foolsgold", "clipfg"):
                for pre, anc in (("eclipse", False), ("eclipse-xv", True)):
                    variants += [Variant(f"{pre}|{g}", agg=g, anchor_prev=anc)]
                    variants += [Variant(f"{pre}|{g}|frac={f}", agg=g, frac=f, anchor_prev=anc) for f in (0.5, 0.75)]
                    variants += [Variant(f"{pre}|{g}|keep", agg=g, keep_honest=True, anchor_prev=anc)]
        if a.grid == "fixnan":  # variants whose victim diverged to NaN before the finiteness check (rerun with it)
            variants = [Variant(f"{att}|{g}", attack=att, agg=g) for att in ("signflip", "minmax-t") for g in AGG_NAMES]
            if a.dataset != "cifar10":
                variants += [Variant(f"eclipse|{g}|scale={s}", agg=g, scale=s) for g in ("mean", "foolsgold")
                             for s in (0.5, 2.0, 3.0)]
        if a.grid == "none":  # calibration of the honest network only
            variants = [Variant("none|mean", attack="none"), Variant("eclipse|mean"), Variant("identical|mean")]
        cfg = RunConfig(dataset=a.dataset, seed=seed, rounds=a.rounds, alpha=a.alpha if a.alpha > 0 else None,
                        local_steps=a.local_steps, lr=a.lr)
        Engine(cfg, variants).run(os.path.join(a.out, f"{a.dataset}{a.tag}_seed{seed}.json"))
