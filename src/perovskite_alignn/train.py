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

STABLE = 0.05


def load_config(path: str, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides:
        key, raw = ov.split("=", 1)
        *parents, last = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        node[last] = yaml.safe_load(raw)
    return cfg


def split(df: pd.DataFrame, seed: int, fractions=(0.8, 0.1)) -> pd.Series:
    formulas = np.random.default_rng(seed).permutation(sorted(df.formula.unique()))
    cum = df.formula.value_counts().reindex(formulas).cumsum() / len(df)
    part = pd.Series(np.where(cum <= fractions[0], "train", np.where(cum <= sum(fractions), "val", "test")), formulas)
    return df.formula.map(part)


def graphs_for(df: pd.DataFrame, cfg: dict) -> dict:
    path, settings = Path(cfg["data"]["graphs"]), cfg["graph"]
    cache = torch.load(path, weights_only=True) if path.exists() else {}
    graphs = cache.get("graphs", {}) if cache.get("settings") == settings else {}
    if missing := [m for m in df.material_id if m not in graphs]:
        s = dict(zip(df.material_id, df.structure))
        graphs |= {m: graph.build(Structure.from_dict(json.loads(s[m])), **settings) for m in missing}
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


def metrics(p: np.ndarray, y: np.ndarray) -> dict:
    stable = y <= STABLE
    return {"MAE": np.mean(np.abs(p - y)), "RMSE": np.sqrt(np.mean((p - y) ** 2)),
            "R2": 1 - np.sum((p - y) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12),
            "Spearman": spearmanr(y, p).statistic if len(y) > 2 else np.nan,
            f"AUC stable <= {STABLE}": roc_auc_score(stable, -p) if 0 < stable.sum() < len(y) else np.nan}


def train(cfg: dict) -> str:
    t, out, dev = cfg["train"], Path(cfg["out"]), cfg["train"].get("device", "cpu")
    df = pd.read_parquet(cfg["data"]["table"]).query("is_perovskite & e_hull.notna()").reset_index(drop=True)
    torch.manual_seed(t["seed"])
    rng, gs = np.random.default_rng(t["seed"]), graphs_for(df, cfg)
    df["set"] = split(df, t["seed"])
    tr = df[df.set == "train"].sample(frac=1, random_state=t["seed"])
    lab = tr.head(max(1, round(t["label_fraction"] * len(tr))))
    val, test = df[df.set == "val"], df[df.set == "test"]
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
            best, best_epoch, best_state = val_mae, epoch, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  val MAE {val_mae:.4f}  best {best:.4f}", flush=True)
        if epoch - best_epoch >= t["patience"]:
            break
    model.load_state_dict(best_state)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "model": cfg["model"], "graph": cfg["graph"]}, out / "model.pt")
    p = predict(model, [gs[m] for m in test.material_id], dev)
    test.assign(pred_ehull=p)[["material_id", "formula", "e_hull", "pred_ehull"]].to_csv(
        out / "test_predictions.csv", index=False)
    lines = [f"ALIGNN + graph-Laplacian SSL (weight {t['ssl_weight']}): {len(lab)} labeled of {len(tr)} training, "
             f"{len(val)} validation, {len(test)} test materials; best epoch {best_epoch}, validation MAE {best:.4f}",
             *(f"test {name:<18} {v:.4f}" for name, v in metrics(p, test.e_hull.values).items())]
    (out / "report.txt").write_text(text := "\n".join(lines) + "\n")
    return text


def predict_files(cfg: dict, inputs: list[str]) -> str:
    ck = torch.load(Path(cfg["out"]) / "model.pt", weights_only=True)
    model = ALIGNN(**ck["model"], cutoff=ck["graph"]["cutoff"])
    model.load_state_dict(ck["state"])
    structs, lines = [], []
    for f in (f for i in map(Path, inputs) for f in (sorted(i.glob("*.cif")) if i.is_dir() else [i])):
        try:
            structs.append((f.name, Structure.from_file(f)))
        except Exception as e:
            lines.append(f"{f.name}: skipped ({type(e).__name__}: {e})")
    if not structs:
        raise SystemExit("\n".join(lines + ["no readable CIF files given"]))
    p = predict(model, [graph.build(s, **ck["graph"]) for _, s in structs])
    return "\n".join(lines + [f"{name}: {s.composition.reduced_formula}  E_hull {e:.3f} eV/atom -> "
                              f"{'stable or metastable' if e <= STABLE else 'unstable'}"
                              + ("" if data.is_perovskite(s, cfg["data"]["anions"]) else "  (not a perovskite)")
                              for (name, s), e in zip(structs, p)])


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
