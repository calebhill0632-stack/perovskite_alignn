"""Data: Materials Project ABX3 entries, a basic perovskite check, composition descriptors, offline test data."""
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymatgen.core import Composition, Element, Lattice, Structure


def fetch(n_max: int, anions: list[str], max_sites: int = 60) -> pd.DataFrame:
    """ABX3 entries (anonymized formula ABC3 containing one of `anions`) with structure and E_hull (eV/atom),
    a fixed random subset of at most n_max. The API key is read from MP_API_KEY (.env) and never printed."""
    from mp_api.client import MPRester

    load_dotenv()
    if not (key := os.getenv("MP_API_KEY", "").strip()):
        raise SystemExit("no API key: put MP_API_KEY=<your key> in a .env file (see .env.example)")
    with MPRester(key, mute_progress_bars=True) as mpr:
        docs = [d for x in anions for d in mpr.materials.summary.search(
            formula="ABC3", elements=[x], deprecated=False, num_sites=(1, max_sites), include_gnome=False,
            fields=["material_id", "formula_pretty", "structure", "energy_above_hull"])]
    df = pd.DataFrame([{"material_id": str(d.material_id), "formula": d.formula_pretty, "e_hull": d.energy_above_hull,
                        "structure": json.dumps(d.structure.as_dict())} for d in docs]).drop_duplicates("material_id")
    return df.sample(frac=1, random_state=0).head(n_max).reset_index(drop=True)


def is_perovskite(s: Structure, anions: list[str], b_tol: float = 0.25, a_tol: float = 0.4) -> bool:
    """Basic ABX3 perovskite check: one cation forms BX6 octahedra (every site has exactly 6 X within
    d_min * (1 + b_tol)) and the other sits in an 8-12 fold X cavity (>= 8 X within d_min * (1 + a_tol))."""
    comp = s.composition.reduced_composition
    xs = [e.symbol for e in comp if e.symbol in anions]
    if len(comp) != 3 or len(xs) != 1 or comp[xs[0]] != 3:
        return False
    cn = {}
    for site in s:
        if (el := site.specie.symbol) != xs[0]:
            d = np.array([n.nn_distance for n in s.get_neighbors(site, 6.0) if n.specie.symbol == xs[0]])
            cn.setdefault(el, []).append(d / d.min() if len(d) else np.array([np.inf]))
    b = [el for el, rs in cn.items() if all((r <= 1 + b_tol).sum() == 6 for r in rs)]
    a = [el for el in cn if el not in b]
    return len(b) == 1 and len(a) == 1 and min(int((r <= 1 + a_tol).sum()) for r in cn[a[0]]) >= 8


PROPS = ("Z", "X", "atomic_radius", "row", "group")


def composition_features(formula: str) -> np.ndarray:
    """Fraction-weighted mean and spread of 5 element properties: the label-free similarity space of the SSL term."""
    comp = Composition(formula).fractional_composition
    P = np.array([[float(getattr(e, p) or 0) for p in PROPS] for e in comp.elements])
    w = np.array([comp[e] for e in comp.elements])[:, None]
    mean = (w * P).sum(0)
    return np.concatenate([mean, np.sqrt((w * (P - mean) ** 2).sum(0))])


def synthetic(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Offline test data (no API key): slightly distorted cubic ABX3 with a made-up, smooth E_hull."""
    rng, rows = np.random.default_rng(seed), []
    A, B, X = ["Ca", "Sr", "Ba", "K", "Cs", "La"], ["Ti", "Zr", "Nb", "Mg", "Sn", "Fe"], ["O", "F", "Cl"]
    for k in range(n):
        a, b, x = rng.choice(A), rng.choice(B), rng.choice(X)
        ra, rb, rx = (Element(e).atomic_radius for e in (a, b, x))
        s = Structure(Lattice.cubic(2 * (rb + rx)), [a, b, x, x, x],
                      [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
        for i in range(len(s)):
            s.translate_sites([i], rng.normal(0, 0.05, 3), frac_coords=False)
        t = (ra + rx) / (np.sqrt(2) * (rb + rx))  # Goldschmidt tolerance factor
        rows.append({"material_id": f"syn-{k}", "formula": s.composition.reduced_formula,
                     "e_hull": float(6 * (t - 0.95) ** 2), "structure": json.dumps(s.as_dict())})
    return pd.DataFrame(rows)
