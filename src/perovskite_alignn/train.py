"""perovskite-alignn fetch|train|predict --config config.yaml [--set section.key=value ...]

  fetch    ABX3 entries from the Materials Project (or offline synthetic data) + perovskite check -> data.table
  train    the angle-aware GNN with a graph-Laplacian SSL term on E_hull; test metrics -> <out>/report.txt
  predict  E_hull and stability verdict of CIF files: perovskite-alignn predict SrTiO3.cif my_cifs/
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pymatgen.core import Structure
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import data, graph
from .model import ALIGNN

STABLE = 0.05  # eV/atom: "stable or metastable" at or below this


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides:  # section.key=value (or key=value), value parsed as YAML
        (*parents, last), value = ov.split("=", 1)[0].split("."), yaml.safe_load(ov.split("=", 1)[1])
        node = cfg
        for p in parents:
            node = node[p]
        node[last] = value
    return cfg


def split(df: pd.DataFrame, seed: int, fractions=(0.8, 0.1)) -> pd.Series:
    """train / val / test, whole reduced formulas at a time (polymorphs never straddle two sets)."""
    formulas = np.random.default_rng(seed).permutation(sorted(df.formula.unique()))
    cum = df.formula.value_counts().reindex(formulas).cumsum() / len(df)
    part = pd.Series(np.where(cum <= fractions[0], "train", np.where(cum <= sum(fractions), "val", "test")), formulas)
    return df.formula.map(part)


def graphs_for(df: pd.DataFrame, cfg: dict) -> dict:
    """material_id -> graph, built once and cached in data.graphs."""
    path, cache = Path(cfg["data"]["graphs"]), {}
    if path.exists():
        cache = torch.load(path, weights_only=True)
    if missing := [m for m in df.material_id if m not in cache]:
        s = dict(zip(df.material_id, df.structure))
        cache |= {m: graph.build(Structure.from_dict(json.loads(s[m])), **cfg["graph"]) for m in missing}
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, path)
    return cache


def predict(model: ALIGNN, gs: list[dict], bs: int = 128) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(graph.collate(gs[i:i + bs]))[0].numpy() for i in range(0, len(gs), bs)] or
                              [np.zeros(0)])


def ssl_loss(model: ALIGNN, pool: list[dict], nbrs: np.ndarray, w: np.ndarray, rng, n: int) -> torch.Tensor:
    """Graph-Laplacian term (Belkin et al. 2006): crystals with similar compositions get similar (L2-normalized)
    embeddings, sum_ij W_ij |z_i - z_j|^2, estimated from n anchors and one random neighbour each. No labels."""
    a = rng.integers(0, len(pool), n)
    j = rng.integers(0, nbrs.shape[1], n)
    z = F.normalize(model(graph.collate([pool[i] for i in a] + [pool[i] for i in nbrs[a, j]]))[1], dim=-1)
    return (torch.as_tensor(w[a, j], dtype=z.dtype) * ((z[:n] - z[n:]) ** 2).sum(-1)).mean()


def train(cfg: dict) -> str:
    t, out = cfg["train"], Path(cfg["out"])
    df = pd.read_parquet(cfg["data"]["table"]).query("is_perovskite & e_hull.notna()").reset_index(drop=True)
    torch.manual_seed(t["seed"])
    rng, gs = np.random.default_rng(t["seed"]), graphs_for(df, cfg)
    df["set"] = split(df, t["seed"])
    tr = df[df.set == "train"].sample(frac=1, random_state=t["seed"])
    lab = tr.head(max(1, round(t["label_fraction"] * len(tr))))  # the rest of the training set: unlabeled
    val, test = df[df.set == "val"], df[df.set == "test"]
    model = ALIGNN(**cfg["model"], cutoff=cfg["graph"]["cutoff"])
    model.y_mean.fill_(lab.e_hull.mean())
    model.y_std.fill_(max(lab.e_hull.std(), 1e-3) if len(lab) > 1 else 1.0)
    X = StandardScaler().fit_transform(np.stack([data.composition_features(f) for f in tr.formula]))
    k = min(t["ssl_k"], len(tr) - 1)
    dist, nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    dist, nbrs = dist[:, 1:], nbrs[:, 1:]  # each crystal's k most similar compositions (itself excluded)
    w = np.exp(-dist ** 2 / max(np.median(dist ** 2), 1e-12))
    pool = [gs[m] for m in tr.material_id]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=1e-5)
    best, best_state, best_epoch, bs = np.inf, None, -1, t["batch_size"]
    for epoch in range(t["epochs"]):
        model.train()
        order = rng.permutation(len(lab))
        for i in range(0, len(lab), bs):
            rows = lab.iloc[order[i:i + bs]]
            pred = model(graph.collate([gs[m] for m in rows.material_id]))[0]
            loss = F.l1_loss(pred, torch.tensor(rows.e_hull.values, dtype=torch.float32)) / model.y_std
            if t["ssl_weight"] > 0 and k > 0:
                loss = loss + t["ssl_weight"] * min(1.0, (epoch + 1) / 10) * ssl_loss(model, pool, nbrs, w, rng,
                                                                                         max(1, bs // 2))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        val_mae = float(np.mean(np.abs(predict(model, [gs[m] for m in val.material_id]) - val.e_hull.values)))
        if val_mae < best:
            best, best_epoch, best_state = val_mae, epoch, {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {epoch:3d}  val MAE {val_mae:.4f}", flush=True)
        if epoch - best_epoch >= t["patience"]:
            break
    model.load_state_dict(best_state)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "model": cfg["model"], "graph": cfg["graph"]}, out / "model.pt")
    p, y = predict(model, [gs[m] for m in test.material_id]), test.e_hull.values
    test.assign(pred_ehull=p)[["material_id", "formula", "e_hull", "pred_ehull"]].to_csv(out / "test_predictions.csv",
                                                                                         index=False)
    stable = y <= STABLE
    lines = [f"ALIGNN-style GNN + graph-Laplacian SSL (weight {t['ssl_weight']}), {len(lab)} labeled / "
             f"{len(tr)} training crystals, {len(val)} val, {len(test)} test; best epoch {best_epoch}",
             f"test MAE   {np.mean(np.abs(p - y)):.4f} eV/atom   (val MAE {best:.4f})",
             f"test RMSE  {np.sqrt(np.mean((p - y) ** 2)):.4f} eV/atom",
             f"test R2    {1 - np.sum((p - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12):.3f}",
             f"Spearman   {spearmanr(y, p).statistic:.3f}" if len(y) > 2 else "",
             f"stable (E_hull <= {STABLE}) vs not, ROC AUC: "
             + (f"{roc_auc_score(stable, -p):.3f}" if 0 < stable.sum() < len(y) else "n/a (one class only)")]
    (out / "report.txt").write_text(text := "\n".join(x for x in lines if x) + "\n")
    return text


def predict_files(cfg: dict, inputs: list[str]) -> str:
    ck = torch.load(Path(cfg["out"]) / "model.pt", weights_only=True)
    model = ALIGNN(**ck["model"], cutoff=ck["graph"]["cutoff"])
    model.load_state_dict(ck["state"])
    files = [f for i in map(Path, inputs) for f in (sorted(i.glob("*.cif")) if i.is_dir() else [i])]
    if not files:
        raise SystemExit("give CIF files or folders of CIF files")
    structs = [Structure.from_file(f) for f in files]
    p = predict(model, [graph.build(s, **ck["graph"]) for s in structs])
    return "\n".join(f"{f.name}: {s.composition.reduced_formula}  E_hull {e:.3f} eV/atom -> "
                     f"{'stable or metastable' if e <= STABLE else 'unstable'}"
                     + ("" if data.is_perovskite(s, cfg["data"]["anions"]) else "  (not a perovskite: extrapolation)")
                     for f, s, e in zip(files, structs, p))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="perovskite-alignn", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["fetch", "train", "predict"])
    ap.add_argument("inputs", nargs="*", help="predict: CIF files or folders")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. train.epochs=20")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.command == "fetch":
        d = cfg["data"]
        df = data.synthetic() if d["source"] == "synthetic" else data.fetch(d["max_materials"], d["anions"],
                                                                             d["max_sites"])
        df["is_perovskite"] = [data.is_perovskite(Structure.from_dict(json.loads(s)), d["anions"])
                               for s in df.structure]
        Path(d["table"]).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d["table"], index=False)
        print(f"{len(df)} materials, {int(df.is_perovskite.sum())} pass the perovskite check -> {d['table']}")
    else:
        print(train(cfg) if a.command == "train" else predict_files(cfg, a.inputs))


if __name__ == "__main__":
    main()
