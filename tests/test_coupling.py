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


# ----------------------------------------------------------------- return_raw
def test_return_raw_leaves_the_default_output_unchanged(synthetic):
    adt, rna, meta = synthetic
    kw = dict(group_size=10, n_draw=3, min_groups=25, seed=11)
    default = coupling.measure(adt, rna, meta, {"CD8": "CD8A", "CD152": "CTLA4"}, **kw)
    summary, raw = coupling.measure(adt, rna, meta, {"CD8": "CD8A", "CD152": "CTLA4"},
                                    return_raw=True, **kw)
    pd.testing.assert_frame_equal(default, summary)


def test_raw_observations_reproduce_the_summary(synthetic):
    adt, rna, meta = synthetic
    summary, raw = coupling.measure(adt, rna, meta, {"CD8": "CD8A", "CD152": "CTLA4"},
                                    group_size=10, n_draw=3, min_groups=25, seed=5,
                                    return_raw=True)
    med = raw.groupby("marker").rho.median()
    for m in summary.marker:
        assert med[m] == pytest.approx(summary.set_index("marker").loc[m, "median_rho"])
    # one column per stratum variable, plus the variance columns
    assert {"donor", "celltype", "draw", "rho", "sd_adt", "sd_rna", "n_groups"} <= set(raw.columns)
    assert (raw.sd_adt > 0).all() and (raw.sd_rna > 0).all()
    assert set(raw.donor) == {"d1", "d2"}


def test_raw_supports_a_donor_bootstrap(synthetic):
    """The reason return_raw exists: resample donors, not rows."""
    adt, rna, meta = synthetic
    _, raw = coupling.measure(adt, rna, meta, {"CD8": "CD8A", "CD152": "CTLA4"},
                              group_size=10, n_draw=3, min_groups=25, seed=2,
                              return_raw=True)
    rng = np.random.default_rng(0)
    donors = raw.donor.unique()
    meds = []
    for _ in range(50):
        pick = rng.choice(donors, len(donors), replace=True)
        sub = pd.concat([raw[raw.donor == d] for d in pick])
        meds.append(sub[sub.marker == "CD8"].rho.median())
    assert np.percentile(meds, 2.5) > 0.4          # coupled marker stays coupled


# ---------------------------------------------------------------- specificity
def _with_noise_genes(synthetic, n_noise=12, seed=3):
    adt, rna, meta = synthetic
    rng = np.random.default_rng(seed)
    rna = rna.copy()
    for k in range(n_noise):
        rna[f"N{k}"] = rng.normal(size=len(rna))
    return adt, rna, meta


def test_specificity_separates_a_coupled_marker_from_the_background(synthetic):
    adt, rna, meta = _with_noise_genes(synthetic)
    M, S = coupling.specificity(adt, rna, meta, {"CD8": "CD8A", "CD152": "CTLA4"},
                                genes=[c for c in rna.columns], group_size=10, n_draw=3,
                                min_groups=25, seed=1)
    s = S.set_index("marker")
    assert s.loc["CD8", "spec_percentile"] == 1.0 and s.loc["CD8", "rank_of_cognate"] == 1
    assert s.loc["CD8", "z_vs_background"] > 3
    assert s.loc["CD152", "spec_percentile"] < 0.95        # an independent marker is not special
    assert M.shape == (2, rna.shape[1])


def test_specificity_cognate_entries_equal_measure_medians(synthetic):
    adt, rna, meta = _with_noise_genes(synthetic)
    pairs = {"CD8": "CD8A", "CD152": "CTLA4"}
    kw = dict(group_size=10, n_draw=3, min_groups=25, seed=9)
    summary = coupling.measure(adt, rna, meta, pairs, **kw).set_index("marker")
    _, S = coupling.specificity(adt, rna, meta, pairs, genes=list(rna.columns), **kw)
    for m in ("CD8", "CD152"):
        assert S.set_index("marker").loc[m, "cognate_rho"] == pytest.approx(
            summary.loc[m, "median_rho"], abs=1e-6)


def test_specificity_refuses_an_oversized_background(synthetic):
    adt, rna, meta = _with_noise_genes(synthetic)
    with pytest.raises(ValueError, match="GB"):
        coupling.specificity(adt, rna, meta, {"CD8": "CD8A"}, genes=list(rna.columns),
                             max_gb=1e-9, group_size=10, n_draw=3, min_groups=25)


def test_specificity_missing_columns_are_errors(synthetic):
    adt, rna, meta = synthetic
    with pytest.raises(ValueError, match="stratum column"):
        coupling.specificity(adt, rna, meta.drop(columns=["celltype"]), {"CD8": "CD8A"})
    with pytest.raises(ValueError, match="no antibody/gene pair"):
        coupling.specificity(adt, rna, meta, {"NOPE": "NOGENE"})


def test_specificity_needs_a_real_background(synthetic):
    adt, rna, meta = synthetic          # only 3 genes: too few for a percentile
    with pytest.raises(ValueError, match="background genes"):
        coupling.specificity(adt, rna, meta, {"CD8": "CD8A"}, group_size=10, n_draw=2, min_groups=25)
