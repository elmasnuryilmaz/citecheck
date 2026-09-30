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
   gradient (see `diagnostics.size_confounding` and `diagnostics.equal_n_resample`).
3. **Regress out library depth** in both modalities before correlating.

Absolute values depend on how finely cell types are split, so compare *ranks*
between datasets, not raw correlations (`rank_percentile`).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["measure", "specificity", "clr", "log_normalise", "rank_percentile"]


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
            depth_cols=("nCount_ADT", "nCount_RNA"), return_raw: bool = False):
    """Measure protein~mRNA coupling for each antibody/gene pair.

    Parameters
    ----------
    adt, rna : cells x features, already normalised (see `clr`, `log_normalise`).
    meta     : cells x metadata; must contain `strata` and `depth_cols`.
    pairs    : {antibody: gene}, e.g. from `mapping.map_antibodies`.
    return_raw : also return every stratum x draw observation. Needed for
               anything that must respect the donor structure, such as a donor
               bootstrap, a per-cell-type breakdown, or weighting by variance.

    Returns
    -------
    DataFrame with one row per marker: median_rho (the headline statistic),
    q25/q75 across strata and draws, n_obs, n_strata.

    With ``return_raw=True`` a tuple ``(summary, raw)``. `raw` has one row per
    observation: marker, gene, one column per stratum variable, draw, rho, the
    standard deviation of the residualised protein (sd_adt) and mRNA (sd_rna)
    across pseudo-groups, and the number of pseudo-groups. Observations are
    *not* independent (draws within a stratum share cells; strata within a donor
    share the donor), so resample donors, not rows.

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
    raw = []
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
        for draw in range(n_draw):
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
                    if return_raw:
                        raw.append((a, pairs[a], key, draw, float(r),
                                    float(np.std(ya[ok])), float(np.std(yr[ok])),
                                    int(ok.sum())))

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
    summary = pd.DataFrame(rows).sort_values("median_rho", ascending=False, ignore_index=True)
    if not return_raw:
        return summary
    raw_df = pd.DataFrame(raw, columns=["marker", "gene", "_key", "draw", "rho",
                                        "sd_adt", "sd_rna", "n_groups"])
    keys = raw_df.pop("_key")
    for i, s in enumerate(strata):
        raw_df[s] = [(k if isinstance(k, tuple) else (k,))[i] for k in keys]
    raw_df = raw_df[raw_df.marker.isin(summary.marker)].reset_index(drop=True)
    cols = ["marker", "gene", *strata, "draw", "rho", "sd_adt", "sd_rna", "n_groups"]
    return summary, raw_df[cols]


def specificity(adt: pd.DataFrame, rna: pd.DataFrame, meta: pd.DataFrame, pairs: dict, *,
                genes=None, strata=("donor", "celltype"), group_size: int = 10,
                n_draw: int = 10, min_groups: int = 25, seed: int = 0,
                depth_cols=("nCount_ADT", "nCount_RNA"), max_gb: float = 2.0):
    """Is a marker's coupling to its own transcript above the background of other genes?

    A small correlation is hard to read on its own. This correlates every
    antibody with every gene in `genes`, using the same stratification, pseudo-
    grouping and depth adjustment as `measure`, and asks where the cognate gene
    falls within that antibody's row.

    Parameters
    ----------
    genes  : genes forming the background. Default: the cognate genes of `pairs`,
             i.e. the other markers' genes. Pass a broader list for a wider
             background, but note memory grows with markers x genes x observations.
    max_gb : refuse to run if the stored observations would exceed this size.

    Returns
    -------
    (matrix, summary). `matrix` is antibodies x genes, the median correlation.
    `summary` has one row per antibody: cognate_rho, background_mean,
    background_sd, spec_percentile (share of other genes the cognate gene beats:
    1.0 means the protein tracks its own transcript better than any other),
    z_vs_background, rank_of_cognate and n_genes.

    The background is not a null: genes that truly co-vary with the antibody
    (CD3E with CD8A in T cells) sit in it. Read the percentile as descriptive.
    With the same `seed`, the cognate entries equal `measure`'s medians.
    """
    for c in strata:
        if c not in meta.columns:
            raise ValueError(f"meta is missing stratum column {c!r}")
    for c in depth_cols:
        if c not in meta.columns:
            raise ValueError(f"meta is missing depth column {c!r}")
    rng = np.random.default_rng(seed)
    idx = adt.index.intersection(rna.index).intersection(meta.index)
    if len(idx) == 0:
        raise ValueError("adt, rna and meta share no cell barcodes")
    adt, rna, meta = adt.loc[idx], rna.loc[idx], meta.loc[idx]
    pairs = {a: g for a, g in pairs.items() if a in adt.columns and g in rna.columns}
    if not pairs:
        raise ValueError("no antibody/gene pair is present in both matrices")
    abs_ = list(pairs)
    gene_list = list(dict.fromkeys(list(genes) if genes is not None else sorted(set(pairs.values()))))
    gene_list += [g for g in dict.fromkeys(pairs.values()) if g not in gene_list]
    gene_list = [g for g in gene_list if g in rna.columns]
    if len(gene_list) < 6:
        raise ValueError(f"only {len(gene_list)} background genes: a percentile needs at least 6. "
                         "Pass `genes=` (for example other markers' genes, or a sample of expressed genes)")
    min_cells = group_size * min_groups
    n_strata = sum(1 for _, s in meta.groupby(list(strata), observed=True)
                   if len(s) >= min_cells and len(s) // group_size >= min_groups)
    need_gb = n_strata * n_draw * len(abs_) * len(gene_list) * 4 / 1e9
    if need_gb > max_gb:
        raise ValueError(f"would store ~{need_gb:.1f} GB of observations (limit {max_gb}); "
                         "use fewer genes or draws, or raise max_gb")
    stack = []
    for key, sub in meta.groupby(list(strata), observed=True):
        if len(sub) < min_cells:
            continue
        cells = sub.index.values
        ng = len(cells) // group_size
        if ng < min_groups:
            continue
        A = adt.loc[cells, abs_].to_numpy(float)
        R = rna.loc[cells, gene_list].to_numpy(float)
        dA = np.log10(sub[depth_cols[0]].to_numpy(float) + 1)
        dR = np.log10(sub[depth_cols[1]].to_numpy(float) + 1)
        for _ in range(n_draw):
            sel = rng.permutation(len(cells))[:ng * group_size]
            take = sel.reshape(ng, group_size)
            gA, gR = A[take].mean(1), R[take].mean(1)
            XA = np.column_stack([np.ones(ng), dA[take].mean(1)])
            XR = np.column_stack([np.ones(ng), dR[take].mean(1)])
            eA = gA - XA @ np.linalg.lstsq(XA, gA, rcond=None)[0]
            eR = gR - XR @ np.linalg.lstsq(XR, gR, rcond=None)[0]
            okA, okR = eA.std(0) > 1e-12, eR.std(0) > 1e-12
            rA, rR = stats.rankdata(eA, axis=0), stats.rankdata(eR, axis=0)
            zA = (rA - rA.mean(0)) / np.maximum(rA.std(0), 1e-12)
            zR = (rR - rR.mean(0)) / np.maximum(rR.std(0), 1e-12)
            C = (zA.T @ zR) / ng
            C[~okA, :] = np.nan
            C[:, ~okR] = np.nan
            stack.append(C.astype(np.float32))
    if not stack:
        raise ValueError("no stratum had enough cells; lower min_groups/group_size")
    M = pd.DataFrame(np.nanmedian(np.stack(stack), axis=0), index=abs_, columns=gene_list)
    rows = []
    for a in abs_:
        g = pairs[a]
        row = M.loc[a].drop(labels=[g]).dropna()
        c = M.loc[a, g]
        if not np.isfinite(c) or len(row) < 5:
            continue
        rows.append(dict(marker=a, gene=g, cognate_rho=float(c), background_mean=float(row.mean()),
                         background_sd=float(row.std()), spec_percentile=float((row <= c).mean()),
                         z_vs_background=float((c - row.mean()) / row.std()),
                         rank_of_cognate=int((row > c).sum() + 1), n_genes=len(row) + 1))
    return M, pd.DataFrame(rows)


def rank_percentile(df: pd.DataFrame, col: str = "median_rho") -> pd.Series:
    """Within-dataset percentile rank — the unit to compare across datasets."""
    return df[col].rank(pct=True)
