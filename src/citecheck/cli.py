"""Command-line interface: citecheck couple | diagnose | map"""
from __future__ import annotations
import argparse
import sys
import pandas as pd
from . import coupling, diagnostics, mapping
from .io import read_matrix


def _load(a):
    adt = read_matrix(a.adt, a.transpose)
    rna = read_matrix(a.rna, a.transpose)
    meta = read_matrix(a.meta)
    return adt, rna, meta


def cmd_couple(a) -> int:
    adt, rna, meta = _load(a)
    if a.normalise:
        adt = pd.DataFrame(coupling.clr(adt.values), index=adt.index, columns=adt.columns)
        rna = coupling.log_normalise(rna)
    pairs, unmapped = mapping.map_antibodies(adt.columns, rna.columns)
    print(f"mapped {len(pairs)}/{adt.shape[1]} antibodies to cognate genes", file=sys.stderr)
    if unmapped:
        print(f"  unmapped: {', '.join(map(str, unmapped[:12]))}"
              f"{' ...' if len(unmapped) > 12 else ''}", file=sys.stderr)
    res = coupling.measure(adt, rna, meta, pairs, strata=tuple(a.strata),
                           group_size=a.group_size, n_draw=a.draws,
                           min_groups=a.min_groups, seed=a.seed)
    res["pct_rank"] = coupling.rank_percentile(res)
    res["mechanism"] = [mapping.mechanism_of(m) for m in res.marker]
    res.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(res)} markers)", file=sys.stderr)
    with pd.option_context("display.width", 200):
        print(res.head(a.show).to_string(index=False))
        if len(res) > a.show:
            print("...")
            print(res.tail(a.show).to_string(index=False))
    return 0


def cmd_diagnose(a) -> int:
    findings = []
    if a.adt and a.meta:
        adt, meta = read_matrix(a.adt, a.transpose), read_matrix(a.meta)
        findings.append(diagnostics.panel_consistency(adt, meta, by=a.by))
    if a.stat:
        s = read_matrix(a.stat).iloc[:, 0]
        findings.append(diagnostics.ambient_check(s, a.controls))
    if a.groups:
        g = read_matrix(a.groups)
        if a.size_col not in g.columns:
            raise SystemExit(f"--groups must contain a {a.size_col!r} column")
        findings.append(diagnostics.size_confounding(
            g.drop(columns=[a.size_col]), g[a.size_col]))
    if a.pvalues:
        p = read_matrix(a.pvalues).iloc[:, 0]
        findings.append(diagnostics.pvalue_sanity(p))
    if a.bounded:
        findings.append(diagnostics.saturation_check(read_matrix(a.bounded)))
    if not findings:
        raise SystemExit("nothing to check — pass at least one of "
                         "--adt/--meta, --stat, --groups, --pvalues, --bounded")
    for f in findings:
        print(f)
    n_flag = sum(not f.passed for f in findings)
    print(f"\n{len(findings) - n_flag}/{len(findings)} checks passed", file=sys.stderr)
    return 1 if n_flag else 0


def cmd_map(a) -> int:
    tbl = mapping.load_map()
    if a.mechanism:
        tbl = tbl[tbl.note == a.mechanism]
    if a.query:
        q = a.query.lower()
        tbl = tbl[tbl.adt.str.lower().str.contains(q) | tbl.gene.str.lower().str.contains(q)]
    print(tbl.to_string(index=False))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="citecheck",
        description="Protein~mRNA coupling and artefact diagnostics for CITE-seq data.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("couple", help="measure protein~mRNA coupling per antibody")
    c.add_argument("--adt", required=True, help="cells x antibodies")
    c.add_argument("--rna", required=True, help="cells x genes")
    c.add_argument("--meta", required=True,
                   help="cells x metadata (strata + nCount_ADT + nCount_RNA)")
    c.add_argument("--out", default="coupling.csv")
    c.add_argument("--strata", nargs="+", default=["donor", "celltype"])
    c.add_argument("--group-size", type=int, default=10)
    c.add_argument("--draws", type=int, default=10)
    c.add_argument("--min-groups", type=int, default=25)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--show", type=int, default=10, help="rows to print from each end")
    c.add_argument("--transpose", action="store_true", help="inputs are features x cells")
    c.add_argument("--normalise", action="store_true",
                   help="inputs are raw counts: apply CLR (ADT) and log-CP10K (RNA)")
    c.set_defaults(func=cmd_couple)

    d = sub.add_parser("diagnose", help="run artefact checks")
    d.add_argument("--adt"); d.add_argument("--meta")
    d.add_argument("--by", default="donor", help="panel-consistency grouping column")
    d.add_argument("--transpose", action="store_true")
    d.add_argument("--stat", help="one-column table: per-marker test statistic")
    d.add_argument("--controls", nargs="+", default=["CD19", "CD20", "CD14", "CD16"],
                   help="lineage-impossible markers for the cells analysed")
    d.add_argument("--groups", help="groups x features, plus a size column")
    d.add_argument("--size-col", default="n_cells")
    d.add_argument("--pvalues", help="one-column table of p-values")
    d.add_argument("--bounded", help="bounded values (PSI/fractions), features x samples")
    d.set_defaults(func=cmd_diagnose)

    m = sub.add_parser("map", help="inspect the antibody->gene table")
    m.add_argument("--query", help="substring of antibody or gene")
    m.add_argument("--mechanism", choices=list(mapping.MECHANISMS))
    m.set_defaults(func=cmd_map)

    a = p.parse_args(argv)
    return a.func(a)


if __name__ == "__main__":                                    # pragma: no cover
    raise SystemExit(main())
