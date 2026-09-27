# citeqc

Measure how well each antibody in a CITE-seq experiment is predicted by its own
transcript — and check your analysis for four artefacts that produce confident,
biologically plausible false positives.

The short version of why this exists: **a third to a half of surface markers
cannot be read off the transcriptome**, and which ones is predictable from
molecular mechanism rather than random. CD45RA is an alternative splice product
of *PTPRC*; an antibody distinguishes CD45RA from CD45RO, total *PTPRC* mRNA
cannot. CTLA-4 and CD40L sit in intracellular vesicles until the cell is
triggered, so their surface level is set by trafficking, not transcription.

Across four public CITE-seq datasets (~410,000 cells, 118 markers, 2017–2024),
CD45RA was the **lowest-coupled marker in every dataset that measured it**
(rank 1 of 13, 1 of 113, 1 of 29).

## Install

```bash
pip install citeqc
```

Optional `.h5mu` support:

```bash
pip install 'citeqc[anndata]'
```

## Measure coupling

```python
import pandas as pd
from citeqc import coupling, mapping

adt  = ...   # cells x antibodies, CLR-normalised
rna  = ...   # cells x genes, log-normalised
meta = ...   # cells x [donor, celltype, nCount_ADT, nCount_RNA]

pairs, unmapped = mapping.map_antibodies(adt.columns, rna.columns)
result = coupling.measure(adt, rna, meta, pairs)
result["pct_rank"] = coupling.rank_percentile(result)
```

`result` has one row per antibody: `median_rho` is the headline number, with
`q25`/`q75` across strata and draws.

Same thing from the shell:

```bash
citeqc couple --adt adt.csv --rna rna.csv --meta meta.csv --out coupling.csv
```

Add `--normalise` if your matrices are raw counts, and `--transpose` if they are
features × cells.

### How to read the result

Absolute correlations depend on how finely you split cell types, so they are
**not comparable between datasets** — compare `pct_rank` instead. Within a
dataset, calibrate against a marker that should couple well in the cells you are
looking at (CD8/*CD8A* in CD8⁺ T cells, CD19/*CD19* in B cells). That value is
your ceiling; markers far below it are not readable from RNA.

### Three constraints the function enforces for you

1. **Stratification by donor × cell type.** Cell-type composition moves protein
   and mRNA together, so an unstratified correlation largely measures
   composition. This is the single biggest source of inflated coupling.
2. **Equal-sized pseudo-groups.** A mean over more cells is less noisy, so
   unequal group sizes create a size-driven gradient across every feature.
3. **Library depth regressed out** in both modalities.

## Check your analysis

Five checks, each built on a negative control the data already contains. They
answer one question: *what would this analysis report if there were no signal?*

```python
from citeqc import diagnostics as dx

dx.panel_consistency(adt, meta, by="donor")   # is one panel measured everywhere?
dx.ambient_check(my_zscores, controls=["CD19", "CD20"])
dx.size_confounding(pseudobulk, sizes=n_cells_per_group)
dx.pvalue_sanity(my_pvalues)
dx.saturation_check(psi_matrix)               # bounded values: PSI, fractions
```

Each returns a `Finding` with `.passed`, a `.summary` you can print, and
`.detail` with the numbers.

```bash
citeqc diagnose --adt adt.csv --meta meta.csv --pvalues p.csv
```

Exit code is non-zero if any check flags, so it drops into CI.

| Check | What goes wrong | How you can tell |
|---|---|---|
| `panel_consistency` | Antibody panels differ between donors; "absent" and "not measured" get confused | Panel size varies by donor; some donors unstained |
| `ambient_check` | Ambient signal makes every marker look real | Lineage-impossible markers (CD19 on T cells) score like genuine ones |
| `size_confounding` | Group means drift with group size, inflating low-expressed features | Almost every feature moves the same direction with size |
| `pvalue_sanity` | Constant or saturated features flatten the null | p-values *depleted* near zero, or a spike in the interior |
| `saturation_check` | Bounded features pinned at 0/1 carry no information but enter FDR | Large fraction identical across all samples |

`size_confounding` deserves emphasis. In our own analysis, before correction it
produced 8,689 of 11,382 genes at FDR<0.05 — 8,621 "up" against 68 "down", the
latter all ribosomal. After resampling to a fixed number of cells per group,
**zero** genes survived. `diagnostics.equal_n_resample` does the resampling.

## The antibody → gene table

145 entries covering common TotalSeq naming, with mechanism annotations where a
documented post-transcriptional mechanism exists.

```bash
citeqc map --query cd45
citeqc map --mechanism vesicular
```

`map_antibodies` tries the raw name, a normalised form (`CD3--UCHT1-TSA` → `CD3`),
then a form with a trailing replicate index removed (`CD38-1` → `CD38`).
Antibodies it cannot place are **returned as unmapped, never guessed**. Extend it
with `map_antibodies(..., extra={"MyAb": "GENE"})`.

Three caveats: the mapping is one-to-one, so it cannot express multi-gene or
post-translationally modified epitopes; `HLA-DR` is mapped to *HLA-DRA* alone;
and CD45RA and CD45RO both map to *PTPRC*, which is the point rather than a
limitation.

## What it does not do

It does not normalise for you beyond the helpers in `coupling`, denoise ADT
(see totalVI or dsb), or correct ambient RNA (see SoupX, CellBender — note that
neither addresses the antibody signal). It measures and it checks.

## Development

```bash
pip install -e '.[test]'
pytest
```

39 tests, including synthetic data where the coupled and decoupled markers are
known by construction, and a validation script that reproduces published results
on real data to Spearman ρ = 1.000.

## Citing

If you use this, please cite the accompanying paper (in preparation) and the
datasets you analyse.

## License

MIT
