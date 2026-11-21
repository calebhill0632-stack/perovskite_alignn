import numpy as np
import torch
from pymatgen.core import Structure


def build(s: Structure, k: int = 12, cutoff: float = 8.0) -> dict:
    c, p, img, d = s.get_neighbor_list(r=cutoff, exclude_self=True)
    order = np.lexsort((d, c))
    c, p, img, d = c[order], p[order], img[order], d[order]
    keep = np.arange(len(c)) - np.searchsorted(c, c) < k
    c, p, img, d = c[keep], p[keep], img[keep], d[keep]
    vec = s.cart_coords[p] + img @ s.lattice.matrix - s.cart_coords[c]
    starts, ends = (np.searchsorted(c, np.arange(len(s)), side=side) for side in ("left", "right"))
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
