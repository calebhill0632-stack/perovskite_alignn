"""ALIGNN-style network: edge-gated graph convolutions alternating on the line graph (bonds and angles) and the
atom graph (atoms and bonds), then atom-graph-only layers, mean pooling and a linear head for E_hull."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def rbf(x: torch.Tensor, lo: float, hi: float, n: int) -> torch.Tensor:
    """Gaussian expansion of a scalar feature on n centres in [lo, hi]."""
    centres = torch.linspace(lo, hi, n, device=x.device)
    return torch.exp(-((x[:, None] - centres) / ((hi - lo) / (n - 1))) ** 2)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    return torch.zeros(n, src.size(1), dtype=src.dtype, device=src.device).index_add_(0, index, src)


class EdgeGatedConv(nn.Module):
    """Edge-gated graph convolution (Bresson & Laurent 2017, as in ALIGNN): updates node AND edge features."""

    def __init__(self, h: int):
        super().__init__()
        self.src_gate, self.dst_gate, self.edge_gate, self.src_update, self.dst_update = (nn.Linear(h, h)
                                                                                          for _ in range(5))
        self.norm_x, self.norm_e = nn.LayerNorm(h), nn.LayerNorm(h)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, e: torch.Tensor):
        src, dst = edge_index
        m = self.src_gate(x)[src] + self.dst_gate(x)[dst] + self.edge_gate(e)
        gate = torch.sigmoid(m)
        agg = scatter_sum(gate * self.dst_update(x)[src], dst, len(x)) / (scatter_sum(gate, dst, len(x)) + 1e-6)
        return x + F.silu(self.norm_x(self.src_update(x) + agg)), e + F.silu(self.norm_e(m))


class ALIGNN(nn.Module):
    def __init__(self, hidden: int = 64, n_alignn: int = 2, n_gcn: int = 2, n_rbf: int = 40, cutoff: float = 8.0):
        super().__init__()
        self.n_rbf, self.cutoff = n_rbf, cutoff
        self.atom = nn.Embedding(119, hidden)
        self.bond, self.angle = (nn.Sequential(nn.Linear(n_rbf, hidden), nn.SiLU()) for _ in range(2))
        self.alignn = nn.ModuleList(nn.ModuleList([EdgeGatedConv(hidden), EdgeGatedConv(hidden)])
                                    for _ in range(n_alignn))
        self.gcn = nn.ModuleList(EdgeGatedConv(hidden) for _ in range(n_gcn))
        self.head = nn.Linear(hidden, 1)
        self.register_buffer("y_mean", torch.zeros(()))  # standardization of the training labels
        self.register_buffer("y_std", torch.ones(()))

    def forward(self, g: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """(E_hull in eV/atom, crystal embedding) for a batch from graph.collate."""
        x = self.atom(g["z"])
        y = self.bond(rbf(g["dist"], 0, self.cutoff, self.n_rbf))
        z = self.angle(rbf(g["cos"], -1, 1, self.n_rbf))
        for line_conv, atom_conv in self.alignn:
            y, z = line_conv(y, g["line_index"], z)  # bonds exchange information through the angles between them
            x, y = atom_conv(x, g["edge_index"], y)
        for conv in self.gcn:
            x, y = conv(x, g["edge_index"], y)
        emb = scatter_sum(x, g["batch"], g["n"]) / torch.bincount(g["batch"], minlength=g["n"]).clamp(min=1)[:, None]
        return self.head(emb).squeeze(-1) * self.y_std + self.y_mean, emb
