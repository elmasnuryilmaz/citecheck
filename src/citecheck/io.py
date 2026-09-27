"""Readers. CSV/TSV/parquet always; AnnData/MuData when those packages exist."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

__all__ = ["read_matrix", "from_mudata"]


def read_matrix(path, transpose: bool = False) -> pd.DataFrame:
    """Read a cells x features table. `transpose` if the file is features x cells."""
    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    else:
        sep = "\t" if p.suffix in (".tsv", ".txt") or "".join(p.suffixes).startswith(".tsv") else ","
        df = pd.read_csv(p, sep=sep, index_col=0)
    if transpose:
        df = df.T
    df.index = df.index.astype(str)
    return df


def from_mudata(path, adt_key: str = "prot", rna_key: str = "rna",
                strata=("donor", "celltype")):
    """Pull (adt, rna, meta) out of a .h5mu file. Requires `mudata`."""
    try:
        import mudata
    except ImportError as e:                                  # pragma: no cover
        raise ImportError("reading .h5mu needs the optional dependency: "
                          "pip install 'citecheck[anndata]'") from e
    md = mudata.read(str(path))
    for k in (adt_key, rna_key):
        if k not in md.mod:
            raise KeyError(f"modality {k!r} not in file; found {list(md.mod)}")
    adt = md.mod[adt_key].to_df()
    rna = md.mod[rna_key].to_df()
    meta = md.obs.copy()
    for c in strata:
        if c not in meta.columns:
            raise KeyError(f"obs is missing stratum column {c!r}")
    if "nCount_ADT" not in meta:
        meta["nCount_ADT"] = md.mod[adt_key].X.sum(axis=1)
    if "nCount_RNA" not in meta:
        meta["nCount_RNA"] = md.mod[rna_key].X.sum(axis=1)
    return adt, rna, meta
