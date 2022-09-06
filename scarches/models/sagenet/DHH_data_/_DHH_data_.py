from copy import copy
from squidpy.datasets._utils import AMetadata

ST = AMetadata(
    name="ST",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31796207",
)

scRNAseq = AMetadata(
    name="scRNAseq",
    doc_header="",
    # shape=(270876, 43),
    url="https://figshare.com/ndownloader/files/31796219",
)




for name, var in copy(locals()).items():
    if isinstance(var, AMetadata):
        var._create_function(name, globals())


__all__ = [  # noqa: F822
    "ST",
    "scRNAseq",
    
]
