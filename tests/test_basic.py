"""Offline tests on synthetic perovskites (no API key, no network)."""
from pathlib import Path

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from scipy.spatial.transform import Rotation

from perovskite_alignn import data, graph
from perovskite_alignn.model import ALIGNN
from perovskite_alignn.train import main

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"
CUBIC = [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]


def test_line_graph_holds_the_octahedral_angles():
    s = Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"], CUBIC)
    g = graph.build(s, k=6)
    ti_bonds = (g["edge_index"][1] == 1).nonzero().flatten()
    cos = g["cos"][torch.isin(g["line_index"][0], ti_bonds)]
    assert sorted(set(np.round(cos.numpy(), 6).tolist())) == [-1.0, 0.0]  # trans (180) and cis (90) O-Ti-O


def test_perovskite_check():
    assert data.is_perovskite(Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"], CUBIC), ["O"])
    ilmenite = Structure.from_spacegroup("R-3", Lattice.hexagonal(5.088, 14.09), ["Fe", "Ti", "O"],
                                         [[0, 0, 0.3554], [0, 0, 0.1464], [0.317, 0.020, 0.2464]])
    assert not data.is_perovskite(ilmenite, ["O"])  # ABX3, but no corner-sharing octahedra with an A cavity


def test_prediction_is_invariant_to_rotation_and_atom_order():
    s = Structure.from_str(data.synthetic(1).structure[0], fmt="json")
    rot = Rotation.from_euler("xyz", [30, 50, 10], degrees=True).as_matrix()
    variants = [s, Structure(s.lattice.matrix @ rot.T, s.species, s.cart_coords @ rot.T, coords_are_cartesian=True),
                Structure.from_sites([s[i] for i in [3, 1, 4, 0, 2]])]
    torch.manual_seed(0)
    model = ALIGNN(hidden=16).eval()
    with torch.no_grad():
        p = model(graph.collate([graph.build(v) for v in variants]))[0]
    assert p.numpy() == pytest.approx(np.full(3, float(p[0])), abs=1e-4)


def test_fetch_train_predict_end_to_end(tmp_path, capsys):
    sets = ["--config", str(CONFIG), "--set", "data.source=synthetic", f"data.table={tmp_path}/t.parquet",
            f"data.graphs={tmp_path}/g.pt",
            f"out={tmp_path}/run", "train.epochs=3", "train.label_fraction=0.5", "model.hidden=16"]
    main(["fetch", *sets])
    main(["train", *sets])
    report = (tmp_path / "run" / "report.txt").read_text()
    assert "test MAE" in report and "nan" not in report.split("test MAE")[1].split("\n")[0]
    Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"], CUBIC).to(filename=str(tmp_path / "SrTiO3.cif"))
    capsys.readouterr()
    main(["predict", str(tmp_path / "SrTiO3.cif"), *sets])
    assert "SrTiO3.cif: SrTiO3  E_hull" in capsys.readouterr().out
