"""Five checks for artefacts that produce confident-looking false positives.

Each check is built around the same question: *what would this analysis report
if there were no signal at all?* None of them needs a ground truth — they use
negative controls that the data already contains.

    panel_consistency   antibody panel differs between donors/batches
    control_check       markers that cannot be expressed behave like real signal
    size_confounding    a per-group statistic tracks group size
    pvalue_sanity       the p-value distribution is not what a null looks like
    saturation_check    bounded values (PSI, fractions) are pinned at 0/1

`control_check` diagnoses; it does not explain. When it flags, test group size
(`size_confounding`, `equal_n_resample`) before reaching for an ambient-signal
correction: in the data this package was developed on, resampling to equal N
removed the gradient and regressing out ambient signal did not.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["Finding", "panel_consistency", "control_check", "ambient_check", "size_confounding",
           "pvalue_sanity", "saturation_check", "equal_n_resample"]


@dataclass
class Finding:
    check: str
    passed: bool
    summary: str
    detail: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{'PASS' if self.passed else 'FLAG'}] {self.check}: {self.summary}"


# ---------------------------------------------------------------- 1. panel
def panel_consistency(adt: pd.DataFrame, meta: pd.DataFrame, by: str = "donor",
                      min_detect: float = 0.5) -> Finding:
    """Is the same antibody panel measured in every donor/batch?

    Heterogeneous panels are common in multi-site CITE-seq and silently turn
    "marker absent" into "marker not measured". Returns the core panel — the
    markers present everywhere — which is what cross-donor analysis may use.
    """
    if by not in meta.columns:
        raise ValueError(f"meta is missing {by!r}")
    idx = adt.index.intersection(meta.index)
    cov = adt.loc[idx].notna().groupby(meta.loc[idx, by]).mean()
    measured = cov > min_detect
    sizes = measured.sum(axis=1)
    # a group with no measured marker was not stained at all: a separate problem
    empty = list(sizes.index[sizes == 0])
    stained = measured.loc[sizes > 0]
    core = list(stained.columns[stained.all()]) if len(stained) else []
    versions = sorted(set(sizes[sizes > 0]))
    uniform = len(versions) <= 1 and not empty
    parts = []
    if empty:
        parts.append(f"{len(empty)} {by}(s) have no antibody data ({', '.join(map(str, empty[:5]))}"
                     f"{' ...' if len(empty) > 5 else ''}) — exclude them")
    if len(versions) > 1:
        parts.append(f"panel size varies across the remaining {by}s ({versions}); "
                     f"core panel = {len(core)} markers measured everywhere")
    return Finding(
        "panel_consistency", uniform,
        (f"all {len(sizes)} {by}s share one {versions[0]}-marker panel"
         if uniform else "; ".join(parts)),
        dict(panel_sizes=sizes.to_dict(), core_panel=core, n_core=len(core),
             n_total=adt.shape[1], empty_groups=empty, panel_versions=versions))


# --------------------------------------------------------------- 2. controls
def control_check(stat: pd.Series, control_markers, signal_markers=None,
                  ratio_threshold: float = 0.5) -> Finding:
    """Do markers that *cannot* be expressed behave like real signal?

    `stat` is your own per-marker test statistic (Z, -log10 p, effect size...).
    `control_markers` should be absent from the cells analysed — CD19/CD20 on
    T cells, CD3 on monocytes. Choose markers that are truly absent: CD16, CD56
    and CD161 occur on CD8 T-cell subsets and make poor controls there.

    Flags when the controls reach a comparable magnitude to the real markers.
    That means the statistic contains a marker-independent gradient. It does
    not say where the gradient comes from. Group size is the first suspect
    (`size_confounding`); ambient signal is the second, and a correction for it
    should be judged on control markers *held out* of the correction.
    """
    s = pd.Series(stat).abs()
    ctrl = s.reindex([m for m in control_markers if m in s.index]).dropna()
    if ctrl.empty:
        raise ValueError("none of control_markers is present in stat")
    sig = (s.reindex([m for m in signal_markers if m in s.index]).dropna()
           if signal_markers is not None else s.drop(ctrl.index, errors="ignore"))
    ratio = float(ctrl.mean() / sig.mean()) if len(sig) and sig.mean() else np.inf
    ok = ratio < ratio_threshold
    return Finding(
        "control_check", ok,
        (f"negative controls are {ratio:.2f}x the real-marker magnitude"
         + ("" if ok else " — the statistic contains a marker-independent gradient. "
                          "Test group size first (size_confounding / equal_n_resample); "
                          "if an ambient correction is tried, judge it on held-out controls")),
        dict(ratio=ratio, control_mean=float(ctrl.mean()),
             signal_mean=float(sig.mean()) if len(sig) else None,
             controls_used=list(ctrl.index)))


ambient_check = control_check          # previous name, kept so existing code keeps working


# ------------------------------------------------------------ 3. group size
def size_confounding(values: pd.DataFrame, sizes, method: str = "spearman",
                     skew_threshold: float = 0.75) -> Finding:
    """Does a per-group statistic track how many cells each group contains?

    `values` is groups x features (clonotype pseudobulk, cluster means...);
    `sizes` the number of cells per group. Because a mean over more cells is
    less noisy, low-expressed features drift upward with group size. The
    signature is a *pervasive one-directional* association across features —
    real biology does not move almost every feature the same way.
    """
    v = values.select_dtypes("number")
    n = np.log10(np.asarray(sizes, dtype=float) + 1)
    if len(n) != len(v):
        raise ValueError("sizes must have one entry per row of values")
    r = v.apply(lambda c: pd.Series(c).corr(pd.Series(n), method=method))
    r = r.dropna()
    if r.empty:
        raise ValueError("no feature had variance to correlate")
    skew = max((r > 0).mean(), (r < 0).mean())
    ok = skew < skew_threshold
    return Finding(
        "size_confounding", ok,
        (f"{skew:.0%} of features move the same way with group size"
         + ("" if ok else " — resample to a fixed number of cells per group "
                          "(see equal_n_resample) and re-run")),
        dict(skew=float(skew), n_features=int(len(r)),
             median_r=float(r.median()), direction="up" if (r > 0).mean() > 0.5 else "down"))


def equal_n_resample(cell_groups: dict, n: int, n_draw: int = 10, seed: int = 0):
    """Yield `n_draw` dicts {group: array of exactly `n` cell ids}.

    Groups with fewer than `n` cells are dropped — that exclusion is the price
    of removing the size artefact, and is reported by the caller.
    """
    rng = np.random.default_rng(seed)
    usable = {g: np.asarray(c) for g, c in cell_groups.items() if len(c) >= n}
    for _ in range(n_draw):
        yield {g: rng.choice(c, n, replace=False) for g, c in usable.items()}


# ----------------------------------------------------------------- 4. p / PSI
def pvalue_sanity(pvalues, alpha=(0.001, 0.01, 0.05), spike_bins: int = 20) -> Finding:
    """Is this p-value distribution consistent with a working test?

    Three regimes matter. *Enriched* near zero means signal. *Uniform* means no
    signal. *Depleted* near zero means the test is broken — commonly because
    many features are constant or saturated and yield p close to 1, which also
    shows up as a spike somewhere in the interior.
    """
    p = pd.Series(pvalues).dropna()
    p = p[(p >= 0) & (p <= 1)]
    if len(p) < 50:
        raise ValueError("need at least 50 p-values")
    enr = {a: float((p < a).sum() / (a * len(p))) for a in alpha}
    pi0 = float(min(1.0, (p > 0.5).sum() / (0.5 * len(p))))
    hist, edges = np.histogram(p, bins=spike_bins, range=(0, 1))
    k = int(np.argmax(hist))
    spike = float(hist[k] / len(p) * spike_bins)   # 1.0 == uniform
    interior_spike = spike > 3 and 0 < k < spike_bins - 1
    lowest = enr[min(alpha)]
    if lowest < 0.8:
        verdict, ok = "depleted", False
    elif lowest > 1.5:
        verdict, ok = "enriched", True
    else:
        verdict, ok = "uniform", True
    if interior_spike:
        ok = False
    return Finding(
        "pvalue_sanity", ok,
        (f"{verdict} near zero (observed/expected at p<{min(alpha)} = {lowest:.2f}), pi0={pi0:.2f}"
         + (f"; spike at p~{edges[k]:.2f} ({spike:.1f}x uniform) — many features are "
            "constant or saturated, drop them before testing" if interior_spike else "")
         + ("" if ok or interior_spike else " — a working test is never depleted near zero")),
        dict(enrichment=enr, pi0=pi0, verdict=verdict,
             spike_at=float(edges[k]), spike_height=spike, n=len(p)))


def saturation_check(values: pd.DataFrame, lo: float = 0.02, hi: float = 0.98,
                     min_range: float = 0.05, max_saturated: float = 0.25) -> Finding:
    """What fraction of bounded features (PSI, fractions) is pinned at 0 or 1?

    Saturated features carry no information but do enter multiple-testing
    correction and flatten the p-value distribution. Returns the index of the
    features worth testing.
    """
    v = values.select_dtypes("number")
    v = v[v.notna().all(axis=1)]
    if v.empty:
        raise ValueError("no feature is complete across samples")
    sat = (v <= lo).all(axis=1) | (v >= hi).all(axis=1)
    variable = (~sat) & ((v.max(axis=1) - v.min(axis=1)) > min_range)
    frac = float(sat.mean())
    ok = frac <= max_saturated
    return Finding(
        "saturation_check", ok,
        (f"{frac:.0%} of features are saturated across all samples; "
         f"{int(variable.sum()):,} of {len(v):,} are testable"
         + ("" if ok else " — restrict to the testable set before differential testing")),
        dict(fraction_saturated=frac, n_testable=int(variable.sum()),
             n_complete=int(len(v)), testable=list(v.index[variable])))
