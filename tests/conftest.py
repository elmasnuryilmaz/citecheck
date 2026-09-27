import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic():
    """Two donors x two cell types. CD8 is coupled by construction; CD152 is not.

    Cognate genes are CD8A and CTLA4; a third antibody (CD19) is pure noise and
    serves as the lineage-impossible negative control.
    """
    rng = np.random.default_rng(0)
    rows = []
    for donor in ("d1", "d2"):
        for ct in ("T", "Mono"):
            n = 600
            latent = rng.normal(size=n)                  # true CD8A abundance
            rows.append(pd.DataFrame({
                "cell": [f"{donor}_{ct}_{i}" for i in range(n)],
                "donor": donor, "celltype": ct,
                "g_CD8A": latent,
                "g_CTLA4": rng.normal(size=n),
                "g_CD19": rng.normal(size=n),
                "p_CD8": latent + rng.normal(scale=0.35, size=n),   # coupled
                "p_CD152": rng.normal(size=n),                      # decoupled
                "p_CD19": rng.normal(size=n),                       # control
                "nCount_ADT": rng.integers(500, 4000, n),
                "nCount_RNA": rng.integers(2000, 20000, n),
            }))
    df = pd.concat(rows).set_index("cell")
    adt = df[["p_CD8", "p_CD152", "p_CD19"]].rename(
        columns=lambda c: c[2:])
    rna = df[["g_CD8A", "g_CTLA4", "g_CD19"]].rename(columns=lambda c: c[2:])
    meta = df[["donor", "celltype", "nCount_ADT", "nCount_RNA"]]
    return adt, rna, meta
