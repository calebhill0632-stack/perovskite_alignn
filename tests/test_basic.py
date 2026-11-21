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
SRTIO3 = Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"], data.CUBIC)


def test_line_graph_holds_the_octahedral_angles():
    g = graph.build(SRTIO3, k=6)
    ti_bonds = (g["edge_index"][1] == 1).nonzero().flatten()
    cos = g["cos"][torch.isin(g["line_index"][0], ti_bonds)]
    assert sorted(set(np.round(cos.numpy(), 6).tolist())) == [-1.0, 0.0]


def test_perovskite_check():
    assert data.is_perovskite(SRTIO3, ["O"])
    ilmenite = Structure.from_spacegroup("R-3", Lattice.hexagonal(5.088, 14.09), ["Fe", "Ti", "O"],
                                         [[0, 0, 0.3554], [0, 0, 0.1464], [0.317, 0.020, 0.2464]])
    assert not data.is_perovskite(ilmenite, ["O"])
    double = Structure(Lattice.cubic(3.95), ["Ba", "Fe", "O", "O", "O"], data.CUBIC).make_supercell([2, 2, 2])
    for i in [i for i, site in enumerate(double) if site.specie.symbol == "Fe"
              and np.floor(site.frac_coords * 2 + 1e-6).astype(int).sum() % 2]:
        double.replace(i, "Mo")
    assert double.composition.reduced_formula == "Ba2FeMoO6" and data.is_perovskite(double, ["O"])


def test_prediction_is_invariant_to_rotation_and_atom_order():
    s = Structure.from_str(data.synthetic(1).structure[0], fmt="json")
    rot = Rotation.from_euler("xyz", [30, 50, 10], degrees=True).as_matrix()
    ideal = SRTIO3.copy().make_supercell([2, 1, 1])
    variants = [s, Structure(s.lattice.matrix @ rot.T, s.species, s.cart_coords @ rot.T, coords_are_cartesian=True),
                Structure.from_sites([s[i] for i in [3, 1, 4, 0, 2]]), ideal,
                Structure.from_sites([ideal[i] for i in np.random.default_rng(1).permutation(len(ideal))])]
    torch.manual_seed(0)
    model = ALIGNN(hidden=16).eval()
    with torch.no_grad():
        p = model(graph.collate([graph.build(v) for v in variants]))[0].numpy()
    assert p[:3] == pytest.approx(np.full(3, p[0]), abs=1e-4) and p[4] == pytest.approx(p[3], abs=1e-5)


def test_fetch_train_predict_end_to_end(tmp_path, capsys):
    sets = ["--config", str(CONFIG), "--set", "data.source=synthetic", f"data.table={tmp_path}/t.parquet",
            f"data.graphs={tmp_path}/g.pt", f"out={tmp_path}/run", "train.epochs=3", "train.seeds=[0,1]",
            "train.label_fraction=0.5", "model.hidden=16"]
    main(["fetch", *sets])
    main(["train", *sets])
    report = (tmp_path / "run" / "report.txt").read_text()
    assert "MAE median guess" in report and "nan" not in report.split("\nMAE ")[1].split("\n")[0]
    main(["train", *sets, "graph.k=6"])
    assert torch.load(tmp_path / "g.pt", weights_only=True)["settings"]["k"] == 6
    SRTIO3.to(filename=str(tmp_path / "SrTiO3.cif"))
    (tmp_path / "broken.cif").write_text("not a cif")
    capsys.readouterr()
    main(["predict", str(tmp_path / "SrTiO3.cif"), str(tmp_path / "broken.cif"), *sets])
    out = capsys.readouterr().out
    assert "SrTiO3.cif: SrTiO3  E_hull" in out and " ± " in out and "broken.cif: skipped" in out
    with pytest.raises(SystemExit, match="unknown setting 'train.epoch'"):
        main(["train", *sets, "train.epoch=5"])
