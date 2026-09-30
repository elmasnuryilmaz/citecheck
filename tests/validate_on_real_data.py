"""Doğrulama: paket, makaledeki SCAR sonucunu yeniden üretiyor mu?

Prototip (src/60_coupling_core.py) ile üretilen results/V4_scar.csv'ye karşı
paketlenmiş citeqc.coupling.measure() karşılaştırılır.
"""
import os, sys, numpy as np, pandas as pd, pathlib
from scipy import stats
from citeqc import coupling, mapping, diagnostics

ROOT=pathlib.Path(os.environ.get("PROJECT_ROOT", "."))
if not (ROOT/"results/V4_scar.csv").exists():
    sys.exit("set PROJECT_ROOT to the analysis repository (needs results/V4_scar.csv and data/processed/)")
P=ROOT/"data/processed"
meta=pd.read_parquet(P/"gse275871_cellmeta.parquet")
CORE=pd.read_csv(P/"core_adt_panel.csv")["marker"].tolist()
R1=pd.read_csv(P/"rna_panel_matrix.csv",index_col=0).T
R2=pd.read_csv(P/"rna_panel2_matrix.csv",index_col=0).T
R2=R2[[c for c in R2.columns if c not in R1.columns]]
R=R1.join(R2); R.index=R.index.astype(str)
s=meta[meta.Project.fillna("").str.startswith("SCAR")&(meta.RNADoublet.astype(str)=="singlet")]
idx=s.index.intersection(R.index); s=s.loc[idx]; R=R.loc[idx]
A=s[CORE].copy()
M=pd.DataFrame({"donor":s.Project.values,"celltype":s.Celltype.values,
                "nCount_ADT":A.mean(axis=1).values*1000,
                "nCount_RNA":s.nCount_RNA.values},index=s.index)

pairs,unmapped=mapping.map_antibodies(A.columns,R.columns)
print(f"paket {len(pairs)}/{len(CORE)} antikoru eşledi; eşleşmeyen: {unmapped}")

res=coupling.measure(A,R,M,pairs,group_size=10,n_draw=10,min_groups=25,seed=4)
ref=pd.read_csv(ROOT/"results/V4_scar.csv")
ref["marker"]=ref.marker.replace({"CD3-1":"CD3","CD4-2":"CD4","CD38-1":"CD38","CD56-1":"CD56"})
res["m"]=[mapping.normalise(x).replace("-1","").replace("-2","") if x.startswith(("CD3-","CD4-","CD38-","CD56-")) else x for x in res.marker]
mg=res.merge(ref,left_on="gene",right_on="gene",suffixes=("_pkg","_ref"))
r,p=stats.spearmanr(mg.median_rho_pkg,mg.median_rho_ref)
mad=float((mg.median_rho_pkg-mg.median_rho_ref).abs().median())
print(f"\nortak marker: {len(mg)}")
print(f"prototip ile sıralama uyumu: Spearman rho={r:.4f}  p={p:.2e}")
print(f"medyan mutlak fark: {mad:.4f}")
print("\nen düşük 6 (paket):")
print(res.nsmallest(6,"median_rho")[["marker","gene","median_rho"]].to_string(index=False))
assert r>0.95, f"paket prototipi yeniden üretemedi (rho={r:.3f})"
assert mad<0.02, f"değerler çok sapıyor (MAD={mad:.4f})"
print("\n✓ DOĞRULANDI: paket makaledeki sonucu yeniden üretiyor")

print("\n--- tanılamalar aynı veride ---")
print(diagnostics.panel_consistency(A,M,by="donor"))
sz=s.groupby([s.Project,s.Celltype]).size()
