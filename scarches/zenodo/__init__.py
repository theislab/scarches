from typing import Union

from scarches.models import scArches, scArchesZINB, scArchesNB, CVAE
from .file import *
from .deposition import *
from .zip import *


def upload_model(model: Union[scArches, CVAE, scArchesNB, scArchesZINB],
                 deposition_id: str,
                 access_token: str):
    """uploads trained ``model`` to Zenodo.

        Parameters
        ----------
        model: :class:`~scarches.models.scNet`, :class:`~scarches.models.CVAE`, :class:`~scarches.models.CVAE_NB`, :class:`~scarches.models.CVAE_ZINB`
            An instance of one of classes defined in ``scNet.models`` module.
        deposition_id: str
            ID of a deposition in your Zenodo account.
        access_token: str
            Your Zenodo Access token.

        Returns
        -------
        download_link: str
            Generated direct download link for the uploaded model in the deposition. Please **Note** that the link is usable **after** your published your deposition.
    """
    model_path = model.model_path
    output_base_name = f"/tmp/scNet-{model.task_name}"
    output_path = output_base_name + ".zip"
    zip_model_directory(output_path=output_base_name, directory=model_path)
    download_link = upload_file(file_path=output_path,
                                deposition_id=deposition_id,
                                access_token=access_token)
    print("Model has been successfully uploaded")
    return download_link
