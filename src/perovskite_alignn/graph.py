import numpy as np
import torch
from pymatgen.core import Structure


def _bounds(c: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.searchsorted(c, np.arange(n), side="left"), np.searchsorted(c, np.arange(n), side="right")


def build(s: Structure, k: int = 12, cutoff: float = 8.0, tie_tol: float = 1e-3) -> dict:
    c, p, img, d = s.get_neighbor_list(r=cutoff, exclude_self=True)
    order = np.lexsort((d, c))
    c, p, img, d = c[order], p[order], img[order], d[order]
    starts, ends = _bounds(c, len(s))
    keep = d <= d[np.maximum(np.minimum(starts + k, ends) - 1, 0)][c] + tie_tol
    c, p, img, d = c[keep], p[keep], img[keep], d[keep]
    vec = s.cart_coords[p] + img @ s.lattice.matrix - s.cart_coords[c]
    starts, ends = _bounds(c, len(s))
    pairs = [np.stack(np.meshgrid(r, r)).reshape(2, -1) for r in (np.arange(a, b) for a, b in zip(starts, ends))]
    ij = np.concatenate(pairs, 1) if pairs else np.zeros((2, 0), int)
    ij = ij[:, ij[0] != ij[1]]
    cos = (vec[ij[0]] * vec[ij[1]]).sum(1) / (d[ij[0]] * d[ij[1]])
    return {"z": torch.tensor(s.atomic_numbers), "edge_index": torch.tensor(np.stack([p, c])),
            "dist": torch.tensor(d, dtype=torch.float32), "line_index": torch.tensor(ij),
            "cos": torch.tensor(np.clip(cos, -1, 1), dtype=torch.float32)}


def collate(graphs: list[dict], device: str = "cpu") -> dict:
    n_atoms = torch.tensor([len(g["z"]) for g in graphs])
    n_bonds = torch.tensor([g["dist"].numel() for g in graphs])
    off_a, off_b = (torch.cumsum(n, 0) - n for n in (n_atoms, n_bonds))
    batch = {"z": torch.cat([g["z"] for g in graphs]), "dist": torch.cat([g["dist"] for g in graphs]),
             "cos": torch.cat([g["cos"] for g in graphs]),
             "edge_index": torch.cat([g["edge_index"] + o for g, o in zip(graphs, off_a)], 1),
             "line_index": torch.cat([g["line_index"] + o for g, o in zip(graphs, off_b)], 1),
             "batch": torch.repeat_interleave(torch.arange(len(graphs)), n_atoms)}
    return {**{k: v.to(device) for k, v in batch.items()}, "n": len(graphs)}
