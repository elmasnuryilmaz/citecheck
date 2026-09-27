import pytest
from citecheck import mapping


def test_bundled_table_is_sane():
    t = mapping.load_map()
    assert len(t) > 100
    assert t.adt.is_unique
    assert t.gene.notna().all()
    assert set(t.note.unique()) <= {"", "pan", *mapping.MECHANISMS}


@pytest.mark.parametrize("raw,expected", [
    ("CD3--UCHT1-TSA", "CD3E"),
    ("CD45RA", "PTPRC"),
    ("CD38-1", "CD38"),          # replicate index stripped
    ("CD8a", "CD8A"),
    ("CD279_PD-1--RMP1-30-TSA", "PDCD1"),
])
def test_maps_panel_naming_variants(raw, expected):
    pairs, _ = mapping.map_antibodies([raw])
    assert pairs[raw] == expected


def test_unmapped_are_reported_not_guessed():
    pairs, unmapped = mapping.map_antibodies(["CD3", "NOT_AN_ANTIBODY"])
    assert "CD3" in pairs and "NOT_AN_ANTIBODY" not in pairs
    assert unmapped == ["NOT_AN_ANTIBODY"]


def test_gene_filter_drops_absent_genes():
    pairs, unmapped = mapping.map_antibodies(["CD3", "CD19"], genes=["CD3E"])
    assert pairs == {"CD3": "CD3E"}
    assert "CD19" in unmapped


def test_mechanism_lookup():
    assert mapping.mechanism_of("CD45RA") == "splice_isoform"
    assert mapping.mechanism_of("CD152") == "vesicular"
    assert mapping.mechanism_of("CD3") == ""


def test_load_map_returns_a_copy():
    a = mapping.load_map(); a.loc[0, "gene"] = "MUTATED"
    assert mapping.load_map().loc[0, "gene"] != "MUTATED"
