"""citecheck — protein/mRNA coupling and artefact diagnostics for CITE-seq data.

Two entry points:

    from citecheck import coupling, diagnostics

`coupling.measure()` quantifies, for each antibody, how well its signal is
predicted by its cognate transcript — the question "can I read this marker
off the transcriptome?".

`diagnostics.*` implement four checks for artefacts that generate large,
plausible-looking false positives in multimodal single-cell analysis.
Each check is built around a negative control rather than a threshold.
"""
from .mapping import load_map, map_antibodies, MECHANISMS
from .coupling import measure, clr
from . import diagnostics

__version__ = "0.1.0"
__all__ = ["measure", "clr", "load_map", "map_antibodies", "MECHANISMS", "diagnostics"]
