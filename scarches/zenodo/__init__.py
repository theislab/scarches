from typing import Union

from ..models import TRVAE, SCVI, SCANVI, TOTALVI
from .file import *
from .deposition import *
from .zip import *


def upload_model(model: Union[TRVAE, SCVI, SCANVI, TOTALVI, str],
                 deposition_id: str,
                 access_token: str,
                 model_name: str = None):
    """Uploads trained ``model`` to Zenodo.

        Parameters
        ----------
        model: :class:`~scarches.models.TRVAE`, :class:`~scarches.models.SCVI`, :class:`~scarches.models.SCANVI`, :class:`~scarches.models.TOTALVI`, str
            An instance of one of classes defined in ``scarches.models`` module or a path to a saved model.
        deposition_id: str
            ID of a deposition in your Zenodo account.
        access_token: str
            Your Zenodo access token.
        model_name: str
            An optional name of the model to upload

        Returns
        -------
        download_link: str
            Generated direct download link for the uploaded model in the deposition. Please **Note** that the link is usable **after** your published your deposition.
    """
    if model_name is None:
        model_name = type(model).__name__

    if isinstance(model, str):
        model_path = model
    else:
        model_path = f"tmp_{model_name}"
        model.save(model_path)

    output_base_name = f"./tmp/scarches-{model_name}"
    output_path = output_base_name + ".zip"
    zip_model_directory(output_path=output_base_name, directory=model_path)
    download_link = upload_file(file_path=output_path,
                                deposition_id=deposition_id,
                                access_token=access_token)
    print("Model has been successfully uploaded")
    return download_link


def download_model(download_link: str,
                   save_path: str = './',
                   make_dir: bool = False):
    """Downloads the zip file of the model in the ``link`` and saves it in ``save_path`` and extracts.

        Parameters
        ----------
        link: str
            Direct downloadable link.
        save_path: str
            Directory path for downloaded file
        make_dir: bool
            Whether to make the ``save_path`` if it does not exist in the system.

        Returns
        -------
        extract_dir: str
            Full path to the folder of the model.
    """
    if not save_path.endswith("/"):
        save_path += "/"
    if download_link != '':
        file_path, response = download_file(download_link, f'{save_path}downloaded_model.zip', make_dir)
    else:
        raise Exception("Download link does not exist for the specified task")

    if os.path.exists(file_path) and file_path.endswith(".zip"):
        extract_dir = os.path.dirname(file_path)
        unzip_model_directory(file_path, extract_dir=extract_dir)
    else:
        raise Exception("The model should be in zip archive")

    return extract_dir
