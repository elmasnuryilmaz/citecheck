import numpy as np
import pandas as pd
import pytest
from citeqc import diagnostics as dx


# ---------------------------------------------------------------- panel
def test_panel_consistency_passes_on_uniform_panel(synthetic):
    adt, _, meta = synthetic
    f = dx.panel_consistency(adt, meta)
    assert f.passed and f.detail["n_core"] == adt.shape[1]


def test_panel_consistency_flags_and_returns_core(synthetic):
    adt, _, meta = synthetic
    adt = adt.copy()
    adt.loc[meta.donor == "d2", "CD19"] = np.nan      # d2 lacks CD19
    f = dx.panel_consistency(adt, meta)
    assert not f.passed
    assert set(f.detail["core_panel"]) == {"CD8", "CD152"}


# -------------------------------------------------------------- ambient
def test_ambient_check_flags_when_controls_look_real():
    stat = pd.Series({"CD3": 3.4, "CD8": 3.1, "CD19": 3.3, "CD20": 3.5})
    f = dx.ambient_check(stat, ["CD19", "CD20"])
    assert not f.passed and f.detail["ratio"] > 0.5


def test_ambient_check_passes_when_controls_are_quiet():
    stat = pd.Series({"CD3": 4.0, "CD8": 4.0, "CD19": 0.3, "CD20": 0.2})
    assert dx.ambient_check(stat, ["CD19", "CD20"]).passed


def test_ambient_check_needs_present_controls():
    with pytest.raises(ValueError, match="control_markers"):
        dx.ambient_check(pd.Series({"CD3": 1.0}), ["CD19"])


# ----------------------------------------------------------- group size
def test_size_confounding_flags_pervasive_drift():
    rng = np.random.default_rng(0)
    sizes = rng.integers(5, 500, 200)
    # every feature drifts up with size: the artefact signature
    vals = pd.DataFrame({f"g{i}": np.log10(sizes) + rng.normal(scale=0.5, size=200)
                         for i in range(40)})
    f = dx.size_confounding(vals, sizes)
    assert not f.passed and f.detail["direction"] == "up" and f.detail["skew"] > 0.9


def test_size_confounding_passes_on_independent_features():
    rng = np.random.default_rng(1)
    sizes = rng.integers(5, 500, 300)
    vals = pd.DataFrame({f"g{i}": rng.normal(size=300) for i in range(40)})
    assert dx.size_confounding(vals, sizes).passed


def test_size_confounding_checks_length():
    with pytest.raises(ValueError, match="one entry per row"):
        dx.size_confounding(pd.DataFrame({"a": [1.0, 2.0, 3.0]}), [1, 2])


def test_equal_n_resample_is_exact_and_drops_small_groups():
    groups = {"a": list(range(20)), "b": list(range(3))}
    draws = list(dx.equal_n_resample(groups, n=5, n_draw=3, seed=0))
    assert len(draws) == 3
    for d in draws:
        assert set(d) == {"a"}                 # 'b' too small, excluded
        assert len(d["a"]) == 5 and len(set(d["a"])) == 5


# ------------------------------------------------------------- p-values
def test_pvalue_sanity_calls_signal():
    rng = np.random.default_rng(0)
    p = np.concatenate([rng.uniform(0, 0.001, 200), rng.uniform(size=1800)])
    f = dx.pvalue_sanity(p)
    assert f.passed and f.detail["verdict"] == "enriched"


def test_pvalue_sanity_calls_uniform_null():
    p = np.random.default_rng(1).uniform(size=5000)
    f = dx.pvalue_sanity(p)
    assert f.passed and f.detail["verdict"] == "uniform"


def test_pvalue_sanity_flags_depletion_and_spike():
    rng = np.random.default_rng(2)
    # 70% of tests are constant features piling up at one interior p value
    p = np.concatenate([rng.uniform(0.35, 0.40, 3500), rng.uniform(size=1500)])
    f = dx.pvalue_sanity(p)
    assert not f.passed
    assert f.detail["spike_height"] > 3


def test_pvalue_sanity_needs_enough_values():
    with pytest.raises(ValueError, match="at least 50"):
        dx.pvalue_sanity([0.1, 0.2])


# ----------------------------------------------------------- saturation
def test_saturation_check_flags_pinned_features():
    n = 500
    sat = pd.DataFrame(np.zeros((400, 6)))
    var = pd.DataFrame(np.random.default_rng(0).uniform(0.2, 0.8, (100, 6)))
    psi = pd.concat([sat, var], ignore_index=True)
    f = dx.saturation_check(psi)
    assert not f.passed
    assert f.detail["fraction_saturated"] == pytest.approx(0.8)
    assert f.detail["n_testable"] == 100


def test_saturation_check_passes_on_variable_data():
    psi = pd.DataFrame(np.random.default_rng(0).uniform(0.1, 0.9, (300, 6)))
    assert dx.saturation_check(psi).passed


def test_finding_str_is_readable():
    f = dx.Finding("x", False, "because")
    assert str(f) == "[FLAG] x: because"


def test_panel_consistency_separates_unstained_groups(synthetic):
    adt, _, meta = synthetic
    adt = adt.copy()
    adt.loc[meta.donor == "d2", :] = np.nan            # d2 not stained at all
    f = dx.panel_consistency(adt, meta)
    assert not f.passed
    assert f.detail["empty_groups"] == ["d2"]
    assert f.detail["n_core"] == adt.shape[1]          # core from stained donors only
    assert "no antibody data" in f.summary
