import pandas as pd
import pytest
from citeqc.cli import main


def _write(tmp_path, synthetic):
    adt, rna, meta = synthetic
    adt.to_csv(tmp_path / "adt.csv"); rna.to_csv(tmp_path / "rna.csv")
    meta.to_csv(tmp_path / "meta.csv")
    return [str(tmp_path / f) for f in ("adt.csv", "rna.csv", "meta.csv")]


def test_couple_end_to_end(tmp_path, synthetic, capsys):
    a, r, m = _write(tmp_path, synthetic)
    out = tmp_path / "res.csv"
    assert main(["couple", "--adt", a, "--rna", r, "--meta", m,
                 "--out", str(out), "--draws", "3"]) == 0
    res = pd.read_csv(out).set_index("marker")
    assert res.loc["CD8", "median_rho"] > res.loc["CD152", "median_rho"]
    assert {"pct_rank", "mechanism"} <= set(res.columns)
    assert res.loc["CD152", "mechanism"] == "vesicular"


def test_diagnose_exit_code_signals_flags(tmp_path, synthetic):
    adt, _, meta = synthetic
    adt = adt.copy(); adt.loc[meta.donor == "d2", "CD19"] = None
    adt.to_csv(tmp_path / "adt.csv"); meta.to_csv(tmp_path / "meta.csv")
    rc = main(["diagnose", "--adt", str(tmp_path / "adt.csv"),
               "--meta", str(tmp_path / "meta.csv")])
    assert rc == 1                                   # a flag must be non-zero


def test_diagnose_without_inputs_errors():
    with pytest.raises(SystemExit):
        main(["diagnose"])


def test_map_query(capsys):
    assert main(["map", "--query", "ctla"]) == 0
    assert "CTLA4" in capsys.readouterr().out
