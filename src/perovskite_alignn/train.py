import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
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

STABLE = 0.05


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides:
        key, raw = ov.split("=", 1)
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            node = node.get(p) if isinstance(node, dict) else None
        if not isinstance(node, dict) or last not in node:
            raise SystemExit(f"unknown setting {key!r} (see {path})")
        node[last] = yaml.safe_load(raw)
    return cfg


def split(df: pd.DataFrame, seed: int, fractions=(0.8, 0.1)) -> pd.Series:
    formulas = np.random.default_rng(seed).permutation(sorted(df.formula.unique()))
    cum = df.formula.value_counts().reindex(formulas).cumsum() / len(df)
    part = pd.Series(np.where(cum <= fractions[0], "train", np.where(cum <= sum(fractions), "val", "test")), formulas)
    return df.formula.map(part)


def _build(args: tuple[str, dict]) -> dict:
    return graph.build(Structure.from_dict(json.loads(args[0])), **args[1])


def graphs_for(df: pd.DataFrame, cfg: dict) -> dict:
    path, settings = Path(cfg["data"]["graphs"]), cfg["graph"]
    cache = torch.load(path, weights_only=True) if path.exists() else {}
    graphs = cache.get("graphs", {}) if cache.get("settings") == settings else {}
    if not (missing := df[~df.material_id.isin(graphs)]).empty:
        items, workers = [(s, settings) for s in missing.structure], cfg["data"].get("workers", 1)
        if workers > 1 and len(items) > 100:
            with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as pool:
                graphs |= dict(zip(missing.material_id, pool.map(_build, items, chunksize=16)))
        else:
            graphs |= dict(zip(missing.material_id, map(_build, items)))
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"settings": settings, "graphs": graphs}, path)
    return graphs


def predict(model: ALIGNN, gs: list[dict], device: str = "cpu", bs: int = 128) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(graph.collate(gs[i:i + bs], device))[0].cpu().numpy()
                               for i in range(0, len(gs), bs)] or [np.zeros(0)])


def similarity(formulas: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
    X = StandardScaler().fit_transform(np.stack([data.composition_features(f) for f in formulas]))
    dist, nbrs = NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)
    dist, nbrs = dist[:, 1:], nbrs[:, 1:]
    return nbrs, np.exp(-dist ** 2 / max(np.median(dist ** 2), 1e-12))


def ssl_loss(model: ALIGNN, pool: list[dict], nbrs: np.ndarray, w: np.ndarray, rng, n: int, device: str):
    a, j = rng.integers(0, len(pool), n), rng.integers(0, nbrs.shape[1], n)
    z = F.normalize(model(graph.collate([pool[i] for i in a] + [pool[i] for i in nbrs[a, j]], device))[1], dim=-1)
    return (torch.as_tensor(w[a, j], dtype=z.dtype, device=device) * ((z[:n] - z[n:]) ** 2).sum(-1)).mean()


def metrics(p: np.ndarray, y: np.ndarray, y_train: np.ndarray) -> dict:
    stable = y <= STABLE
    return {"MAE": np.mean(np.abs(p - y)), "MAE median guess": np.mean(np.abs(np.median(y_train) - y)),
            "RMSE": np.sqrt(np.mean((p - y) ** 2)),
            "R2": 1 - np.sum((p - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12),
            "Spearman": spearmanr(y, p).statistic if len(y) > 2 else np.nan,
            f"AUC stable <= {STABLE}": roc_auc_score(stable, -p) if 0 < stable.sum() < len(y) else np.nan}


def train_seed(cfg: dict, df: pd.DataFrame, gs: dict, seed: int) -> tuple[ALIGNN, pd.DataFrame, dict]:
    t, dev = cfg["train"], cfg["train"].get("device", "cpu")
    torch.manual_seed(seed)
    rng, part = np.random.default_rng(seed), split(df, seed)
    tr = df[part == "train"].sample(frac=1, random_state=seed)
    lab = tr.head(max(1, round(t["label_fraction"] * len(tr))))
    val, test = df[part == "val"], df[part == "test"]
    if val.empty or test.empty:
        raise SystemExit(f"only {len(df)} perovskites with labels: too few for validation and test sets")
    model = ALIGNN(**cfg["model"], cutoff=cfg["graph"]["cutoff"]).to(dev)
    model.y_mean.fill_(lab.e_hull.mean())
    model.y_std.fill_(max(lab.e_hull.std(), 1e-3) if len(lab) > 1 else 1.0)
    k = min(t["ssl_k"], len(tr) - 1)
    use_ssl = t["ssl_weight"] > 0 and k > 0
    nbrs, w = similarity(tr.formula.tolist(), k) if use_ssl else (None, None)
    pool = [gs[m] for m in tr.material_id]
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=t["epochs"])
    best, best_state, best_epoch, bs = np.inf, None, -1, t["batch_size"]
    for epoch in range(t["epochs"]):
        model.train()
        order = rng.permutation(len(lab))
        for i in range(0, len(lab), bs):
            rows = lab.iloc[order[i:i + bs]]
            pred = model(graph.collate([gs[m] for m in rows.material_id], dev))[0]
            loss = F.l1_loss(pred, torch.tensor(rows.e_hull.values, dtype=torch.float32, device=dev)) / model.y_std
            if use_ssl:
                ramp = min(1.0, (epoch + 1) / 10)
                loss = loss + t["ssl_weight"] * ramp * ssl_loss(model, pool, nbrs, w, rng, max(1, bs // 2), dev)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        val_mae = float(np.mean(np.abs(predict(model, [gs[m] for m in val.material_id], dev) - val.e_hull.values)))
        if val_mae < best:
            best, best_epoch = val_mae, epoch
            best_state = {name: v.detach().clone() for name, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"seed {seed}  epoch {epoch:3d}  val MAE {val_mae:.4f}  best {best:.4f}", flush=True)
        if epoch - best_epoch >= t["patience"]:
            break
    model.load_state_dict(best_state)
    p = predict(model, [gs[m] for m in test.material_id], dev)
    info = {"labeled": len(lab), "training": len(tr), "best epoch": best_epoch, "val MAE": best,
            **metrics(p, test.e_hull.values, lab.e_hull.values)}
    return model, test.assign(seed=seed, pred_ehull=p)[["seed", "material_id", "formula", "e_hull", "pred_ehull"]], info


def train(cfg: dict) -> str:
    t, out = cfg["train"], Path(cfg["out"])
    df = pd.read_parquet(cfg["data"]["table"]).query("is_perovskite & e_hull.notna()").reset_index(drop=True)
    gs = graphs_for(df, cfg)
    runs = [train_seed(cfg, df, gs, seed) for seed in t["seeds"]]
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"states": [m.cpu().state_dict() for m, _, _ in runs], "model": cfg["model"], "graph": cfg["graph"]},
               out / "model.pt")
    pd.concat([r[1] for r in runs]).to_csv(out / "test_predictions.csv", index=False)
    info = pd.DataFrame([r[2] for r in runs])
    counts = {"labeled", "training", "best epoch"}
    lines = [f"ALIGNN + graph-Laplacian SSL (weight {t['ssl_weight']}), {len(df)} perovskites, seeds {t['seeds']}, "
             "mean ± sd over seeds (eV/atom for errors)",
             *(f"{name:<24} {v.mean():.{1 if name in counts else 4}f} ± {v.std(ddof=0):.{1 if name in counts else 4}f}"
               for name, v in info.items())]
    (out / "report.txt").write_text(text := "\n".join(lines) + "\n")
    return text


def predict_files(cfg: dict, inputs: list[str]) -> str:
    if not (path := Path(cfg["out"]) / "model.pt").exists():
        raise SystemExit(f"no trained model at {path}: run `perovskite-alignn train` first")
    ck = torch.load(path, weights_only=True)
    models = [ALIGNN(**ck["model"], cutoff=ck["graph"]["cutoff"]) for _ in ck["states"]]
    for m, state in zip(models, ck["states"]):
        m.load_state_dict(state)
    structs, lines = [], []
    files = (f for i in map(Path, inputs) for f in (sorted(i.iterdir()) if i.is_dir() else [i]))
    for f in (f for f in files if f.suffix.lower() == ".cif"):
        try:
            structs.append((f.name, Structure.from_file(f)))
        except Exception as e:
            lines.append(f"{f.name}: skipped ({type(e).__name__}: {e})")
    if not structs:
        raise SystemExit("\n".join(lines + ["no readable CIF files given"]))
    gs = [graph.build(s, **ck["graph"]) for _, s in structs]
    P = np.stack([predict(m, gs) for m in models])
    return "\n".join(lines + [f"{name}: {s.composition.reduced_formula}  E_hull {e:.3f} ± {sd:.3f} eV/atom -> "
                              f"{'stable or metastable' if e <= STABLE else 'unstable'}"
                              + ("" if data.is_perovskite(s, cfg["data"]["anions"]) else "  (not a perovskite)")
                              for (name, s), e, sd in zip(structs, P.mean(0), P.std(0))])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="perovskite-alignn")
    ap.add_argument("command", choices=["fetch", "train", "predict"])
    ap.add_argument("inputs", nargs="*", help="CIF files or folders (predict)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides such as train.epochs=20")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    if a.command == "fetch":
        df = data.prepare(cfg)
        print(f"{len(df)} materials, {int(df.is_perovskite.sum())} perovskites -> {cfg['data']['table']}")
    else:
        print(train(cfg) if a.command == "train" else predict_files(cfg, a.inputs))


if __name__ == "__main__":
    main()
