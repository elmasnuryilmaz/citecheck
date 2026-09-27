import numpy as np
import pandas as pd
import pytest
from citeqc import coupling


def test_recovers_coupled_and_decoupled_markers(synthetic):
    adt, rna, meta = synthetic
    res = coupling.measure(adt, rna, meta,
                           {"CD8": "CD8A", "CD152": "CTLA4", "CD19": "CD19"},
                           group_size=10, n_draw=4, min_groups=25, seed=1)
    r = res.set_index("marker").median_rho
    assert r["CD8"] > 0.5, "a marker generated from its own transcript must couple"
    assert abs(r["CD152"]) < 0.2, "an independent marker must not couple"
    assert r["CD8"] > r["CD152"] and r["CD8"] > r["CD19"]


def test_reports_every_stratum(synthetic):
    adt, rna, meta = synthetic
    res = coupling.measure(adt, rna, meta, {"CD8": "CD8A"},
                           group_size=10, n_draw=3, min_groups=25)
    assert res.loc[0, "n_strata"] == 4          # 2 donors x 2 cell types
    assert res.loc[0, "n_obs"] == 12            # 4 strata x 3 draws


def test_seed_is_deterministic(synthetic):
    adt, rna, meta = synthetic
    kw = dict(group_size=10, n_draw=3, min_groups=25)
    a = coupling.measure(adt, rna, meta, {"CD8": "CD8A"}, seed=7, **kw)
    b = coupling.measure(adt, rna, meta, {"CD8": "CD8A"}, seed=7, **kw)
    pd.testing.assert_frame_equal(a, b)


def test_missing_stratum_column_is_an_error(synthetic):
    adt, rna, meta = synthetic
    with pytest.raises(ValueError, match="stratum column"):
        coupling.measure(adt, rna, meta.drop(columns=["celltype"]), {"CD8": "CD8A"})


def test_missing_depth_column_is_an_error(synthetic):
    adt, rna, meta = synthetic
    with pytest.raises(ValueError, match="depth column"):
        coupling.measure(adt, rna, meta.drop(columns=["nCount_ADT"]), {"CD8": "CD8A"})


def test_disjoint_barcodes_is_an_error(synthetic):
    adt, rna, meta = synthetic
    with pytest.raises(ValueError, match="share no cell barcodes"):
        coupling.measure(adt.rename(index=lambda s: s + "_x"), rna, meta, {"CD8": "CD8A"})


def test_clr_centres_each_cell():
    x = np.array([[1.0, 10.0, 100.0], [5.0, 5.0, 5.0]])
    out = coupling.clr(x, axis=1)
    assert np.allclose(out.mean(axis=1), 0)


def test_rank_percentile_orders_by_coupling():
    df = pd.DataFrame({"median_rho": [0.4, 0.1, -0.05]})
    assert list(coupling.rank_percentile(df)) == [1.0, pytest.approx(2/3), pytest.approx(1/3)]
