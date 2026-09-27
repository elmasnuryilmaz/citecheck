"""Protein ~ cognate-mRNA coupling.

The measurement answers: within a homogeneous group of cells, does a marker's
antibody signal track its own transcript?

Three design constraints make the number meaningful, and all three are enforced
here rather than left to the caller:

1. **Stratify by donor x cell type.** Differences in cell-type composition move
   protein and mRNA together, so an unstratified correlation mostly measures
   composition and is strongly inflated.
2. **Use equal-sized pseudo-groups.** A mean over cells is less noisy when more
   cells are averaged, so groups of unequal size introduce a size-driven
   gradient (see `diagnostics.equal_n_check`).
3. **Regress out library depth** in both modalities before correlating.

Absolute values depend on how finely cell types are split, so compare *ranks*
between datasets, not raw correlations (`rank_percentile`).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["measure", "clr", "log_normalise", "rank_percentile"]


def clr(x, axis: int = 1) -> np.ndarray:
    """Centered log-ratio, the standard ADT normalisation (axis=1: per cell)."""
    lg = np.log1p(np.asarray(x, dtype=float))
    return lg - np.nanmean(lg, axis=axis, keepdims=True)


def log_normalise(counts: pd.DataFrame, total=None, scale: float = 1e4) -> pd.DataFrame:
    """log1p(counts per `scale`), the standard RNA normalisation."""
    tot = counts.sum(axis=1) if total is None else pd.Series(total, index=counts.index)
    return np.log1p(counts.div(tot.replace(0, np.nan), axis=0) * scale)


def _residual(y: np.ndarray, X: np.ndarray, min_obs: int) -> np.ndarray:
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    out = np.full(len(y), np.nan)
    if ok.sum() < min_obs:
        return out
    beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
    out[ok] = y[ok] - X[ok] @ beta
    return out


def measure(adt: pd.DataFrame, rna: pd.DataFrame, meta: pd.DataFrame,
            pairs: dict, *, strata=("donor", "celltype"), group_size: int = 10,
            n_draw: int = 10, min_groups: int = 25, seed: int = 0,
            depth_cols=("nCount_ADT", "nCount_RNA")) -> pd.DataFrame:
    """Measure protein~mRNA coupling for each antibody/gene pair.

    Parameters
    ----------
    adt, rna : cells x features, already normalised (see `clr`, `log_normalise`).
    meta     : cells x metadata; must contain `strata` and `depth_cols`.
    pairs    : {antibody: gene}, e.g. from `mapping.map_antibodies`.

    Returns
    -------
    DataFrame with one row per marker: median_rho (the headline statistic),
    q25/q75 across strata and draws, n_obs, n_strata.

    Raises
    ------
    ValueError if required columns are missing or no cells are shared.
    """
    for c in strata:
        if c not in meta.columns:
            raise ValueError(f"meta is missing stratum column {c!r}")
    for c in depth_cols:
        if c not in meta.columns:
            raise ValueError(f"meta is missing depth column {c!r}; "
                             "pass raw total counts per cell")
    rng = np.random.default_rng(seed)
    idx = adt.index.intersection(rna.index).intersection(meta.index)
    if len(idx) == 0:
        raise ValueError("adt, rna and meta share no cell barcodes")
    adt, rna, meta = adt.loc[idx], rna.loc[idx], meta.loc[idx]
    pairs = {a: g for a, g in pairs.items() if a in adt.columns and g in rna.columns}
    if not pairs:
        raise ValueError("no antibody/gene pair is present in both matrices")

    abs_, genes = list(pairs), [pairs[a] for a in pairs]
    acc = {a: [] for a in abs_}
    min_cells = group_size * min_groups
    for key, sub in meta.groupby(list(strata), observed=True):
        if len(sub) < min_cells:
            continue
        cells = sub.index.values
        A = adt.loc[cells, abs_].to_numpy(float)
        R = rna.loc[cells, genes].to_numpy(float)
        dA = np.log10(sub[depth_cols[0]].to_numpy(float) + 1)
        dR = np.log10(sub[depth_cols[1]].to_numpy(float) + 1)
        ng = len(cells) // group_size
        if ng < min_groups:
            continue
        lab = np.repeat(np.arange(ng), group_size)
        for _ in range(n_draw):
            sel = rng.permutation(len(cells))[:ng * group_size]
            order = np.argsort(lab, kind="stable")
            take = sel[order].reshape(ng, group_size)
            gA, gR = A[take].mean(1), R[take].mean(1)
            XA = np.column_stack([np.ones(ng), dA[take].mean(1)])
            XR = np.column_stack([np.ones(ng), dR[take].mean(1)])
            for j, a in enumerate(abs_):
                ya = _residual(gA[:, j], XA, min_groups)
                yr = _residual(gR[:, j], XR, min_groups)
                ok = np.isfinite(ya) & np.isfinite(yr)
                if ok.sum() < min_groups or np.std(ya[ok]) == 0 or np.std(yr[ok]) == 0:
                    continue
                r = stats.spearmanr(ya[ok], yr[ok]).statistic
                if np.isfinite(r):
                    acc[a].append((key, r))

    rows = []
    for a, v in acc.items():
        if len(v) < 10:
            continue
        rs = np.array([r for _, r in v])
        rows.append(dict(marker=a, gene=pairs[a], median_rho=float(np.median(rs)),
                         q25=float(np.percentile(rs, 25)), q75=float(np.percentile(rs, 75)),
                         n_obs=len(rs), n_strata=len({k for k, _ in v})))
    if not rows:
        raise ValueError("no marker had enough strata; lower min_groups/group_size "
                         "or check that strata are populated")
    return pd.DataFrame(rows).sort_values("median_rho", ascending=False, ignore_index=True)


def rank_percentile(df: pd.DataFrame, col: str = "median_rho") -> pd.Series:
    """Within-dataset percentile rank — the unit to compare across datasets."""
    return df[col].rank(pct=True)
