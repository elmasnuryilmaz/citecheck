"""Antibody -> cognate gene lookup."""
from __future__ import annotations
import re
from functools import lru_cache
from pathlib import Path
import pandas as pd

_DATA = Path(__file__).resolve().parent / "data" / "adt_to_gene.tsv"

MECHANISMS = ("splice_isoform", "vesicular", "IRE_3UTR")

_SUFFIXES = re.compile(
    r"(-TotalSeq[ABC]?|_TotalSeq[ABC]?|-\(?\w+-\d+\)?-TSA|--[A-Za-z0-9._-]+-TSA"
    r"|-prot|_prot|-ADT|_ADT)$", re.I)
_TRAILING_INDEX = re.compile(r"-\d+$")


@lru_cache(maxsize=1)
def _load_cached() -> pd.DataFrame:
    df = pd.read_csv(_DATA, sep="\t")
    df["note"] = df["note"].fillna("")
    return df


def load_map() -> pd.DataFrame:
    """The bundled antibody->gene table (a copy; edit freely)."""
    return _load_cached().copy()


def normalise(name: str) -> str:
    """Strip common CITE-seq panel decorations from an antibody name."""
    s = str(name).strip()
    s = s.split("--")[0]                      # 'CD3--UCHT1-TSA' -> 'CD3'
    s = _SUFFIXES.sub("", s)
    s = s.split("_")[0] if s.count("_") and not s.startswith("HLA") else s
    return s.strip()


def map_antibodies(antibodies, genes=None, extra: dict | None = None):
    """Map antibody names to cognate gene symbols.

    Tries the raw name, then a normalised form, then the normalised form with a
    trailing replicate index removed ('CD38-1' -> 'CD38').

    Returns (pairs, unmapped) where pairs is {antibody: gene}. If `genes` is
    given, pairs are restricted to genes actually present.
    """
    tbl = load_map()
    lut = dict(zip(tbl.adt, tbl.gene))
    if extra:
        lut.update(extra)
    gene_set = set(genes) if genes is not None else None
    pairs, unmapped = {}, []
    for ab in antibodies:
        for key in (str(ab), normalise(ab), _TRAILING_INDEX.sub("", normalise(ab))):
            if key in lut:
                g = lut[key]
                if gene_set is None or g in gene_set:
                    pairs[ab] = g
                break
        else:
            unmapped.append(ab)
            continue
        if ab not in pairs:
            unmapped.append(ab)
    return pairs, unmapped


def mechanism_of(antibody: str) -> str:
    """Documented post-transcriptional mechanism for an antibody, or ''."""
    tbl = _load_cached().set_index("adt")
    for key in (str(antibody), normalise(antibody),
                _TRAILING_INDEX.sub("", normalise(antibody))):
        if key in tbl.index:
            return tbl.loc[key, "note"]
    return ""
