import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymatgen.core import Composition, Element, Lattice, Structure

PROPS = ("Z", "X", "atomic_radius", "row", "group")
CUBIC = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]


def fetch(n_max: int, anions: list[str], max_sites: int = 60) -> pd.DataFrame:
    from mp_api.client import MPRester

    load_dotenv()
    if not (key := os.getenv("MP_API_KEY", "").strip()):
        raise SystemExit("no API key: put MP_API_KEY=<your key> in a .env file (see .env.example)")
    with MPRester(key, mute_progress_bars=True) as mpr:
        docs = [d for x in anions for d in mpr.materials.summary.search(
            formula="ABC3", elements=[x], deprecated=False, num_sites=(1, max_sites), include_gnome=False,
            fields=["material_id", "formula_pretty", "structure", "energy_above_hull"])]
    df = pd.DataFrame([{"material_id": str(d.material_id), "formula": d.formula_pretty, "e_hull": d.energy_above_hull,
                        "structure": json.dumps(d.structure.as_dict())}
                       for d in docs if d.energy_above_hull is not None])
    return df.drop_duplicates("material_id").sample(frac=1, random_state=0).head(n_max).reset_index(drop=True)


def is_perovskite(s: Structure, anions: list[str], b_tol: float = 0.25, a_tol: float = 0.4) -> bool:
    comp = s.composition.reduced_composition
    xs = [e.symbol for e in comp if e.symbol in anions]
    if len(comp) != 3 or len(xs) != 1 or comp[xs[0]] != 3:
        return False
    ratios = {}
    for site in s:
        if (el := site.specie.symbol) != xs[0]:
            d = np.array([n.nn_distance for n in s.get_neighbors(site, 6.0) if n.specie.symbol == xs[0]])
            ratios.setdefault(el, []).append(d / d.min() if len(d) else np.array([np.inf]))
    b = [el for el, rs in ratios.items() if all((r <= 1 + b_tol).sum() == 6 for r in rs)]
    a = [el for el in ratios if el not in b]
    return len(b) == 1 and len(a) == 1 and all((r <= 1 + a_tol).sum() >= 8 for r in ratios[a[0]])


def composition_features(formula: str) -> np.ndarray:
    comp = Composition(formula).fractional_composition
    P = np.array([[float(getattr(e, p) or 0) for p in PROPS] for e in comp.elements])
    w = np.array([comp[e] for e in comp.elements])[:, None]
    mean = (w * P).sum(0)
    return np.concatenate([mean, np.sqrt((w * (P - mean) ** 2).sum(0))])


def synthetic(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng, rows = np.random.default_rng(seed), []
    A, B, X = ["Ca", "Sr", "Ba", "K", "Cs", "La"], ["Ti", "Zr", "Nb", "Mg", "Sn", "Fe"], ["O", "F", "Cl"]
    for k in range(n):
        a, b, x = rng.choice(A), rng.choice(B), rng.choice(X)
        ra, rb, rx = (Element(e).atomic_radius for e in (a, b, x))
        s = Structure(Lattice.cubic(2 * (rb + rx)), [a, b, x, x, x], CUBIC)
        for i in range(len(s)):
            s.translate_sites([i], rng.normal(0, 0.05, 3), frac_coords=False)
        t = (ra + rx) / (np.sqrt(2) * (rb + rx))
        rows.append({"material_id": f"syn-{k}", "formula": s.composition.reduced_formula,
                     "e_hull": float(6 * (t - 0.95) ** 2), "structure": json.dumps(s.as_dict())})
    return pd.DataFrame(rows)


def prepare(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    df = synthetic() if d["source"] == "synthetic" else fetch(d["max_materials"], d["anions"], d["max_sites"])
    df["is_perovskite"] = [is_perovskite(Structure.from_dict(json.loads(s)), d["anions"]) for s in df.structure]
    os.makedirs(os.path.dirname(d["table"]) or ".", exist_ok=True)
    df.to_parquet(d["table"], index=False)
    return df

