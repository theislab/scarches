
from copy import copy
# from squidpy.datasets._utils import AMetadata

class AMetadata(Metadata):
    """Metadata class for :class:`anndata.AnnData`."""
    """Copyright: squidpy"""

    _DOC_FMT = """
    {doc_header}
    The shape of this :class:`anndata.AnnData` object ``{shape}``.
    Parameters
    ----------
    path
        Path where to save the dataset.
    kwargs
        Keyword arguments for :func:`scanpy.read`.
    Returns
    -------
    The dataset."""

    def _create_signature(self) -> Signature:
        return signature(lambda _: _).replace(
            parameters=[
                Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
                Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
            ],
            return_annotation=anndata.AnnData,
        )

    def _download(self, fpath: PathLike, backup_url: str, **kwargs: Any) -> AnnData:
        kwargs.setdefault("sparse", True)
        kwargs.setdefault("cache", True)

        return read(filename=fpath, backup_url=backup_url, **kwargs)

    @property
    def _extension(self) -> str:
        return ".h5ad"


class ImgMetadata(Metadata):
    """Metadata class for :class:`squidpy.im.ImageContainer`."""

    _DOC_FMT = """
    {doc_header}
    The shape of this image is ``{shape}``.
    Parameters
    ----------
    path
        Path where to save the .tiff image.
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.add_img`.
    Returns
    -------
    :class:`squidpy.im.ImageContainer`
        The image data."""
    # not the perfect annotation, but better than nothing
    _EXT = ".tiff"

    def _create_signature(self) -> Signature:
        return signature(lambda _: _).replace(
            parameters=[
                Parameter("path", kind=Parameter.POSITIONAL_OR_KEYWORD, annotation=PathLike, default=None),
                Parameter("kwargs", kind=Parameter.VAR_KEYWORD, annotation=Any),
            ],
        )

    def _download(self, fpath: PathLike, backup_url: str, **kwargs: Any) -> Any:
        from squidpy.im import ImageContainer  # type: ignore[attr-defined]

        check_presence_download(Path(fpath), backup_url)

        img = ImageContainer()
        img.add_img(fpath, layer="image", library_id=self.library_id, **kwargs)

        return img

    @property
    def _extension(self) -> str:
        return ".tiff"

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
