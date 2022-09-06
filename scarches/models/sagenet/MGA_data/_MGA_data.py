
from copy import copy
from squidpy.datasets._utils import AMetadata

_scRNAseq = AMetadata(
    name="scRNAseq",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31767704",
)

_seqFISH1_1 = AMetadata(
    name="seqFISH1_1",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716029",
)

_seqFISH2_1 = AMetadata(
    name="seqFISH2_1",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716041",
)


_seqFISH3_1 = AMetadata(
    name="seqFISH3_1",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31716089",
)


_seqFISH1_2 = AMetadata(
    name="seqFISH1_2",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31920353",
)

_seqFISH2_2 = AMetadata(
    name="seqFISH2_2",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31920644",
)


_seqFISH3_2 = AMetadata(
    name="seqFISH3_2",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31920890",
)


for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "scRNAseq",
    "seqFISH1_1",
    "seqFISH2_1",
    "seqFISH3_1",
    "seqFISH1_2",
    "seqFISH2_2",
    "seqFISH3_2",
    
]
