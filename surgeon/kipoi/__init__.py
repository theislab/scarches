import os

from surgeon.kipoi._data import create_dataloader_yaml
from surgeon.kipoi._model import create_network_yaml
from surgeon.kipoi._upload import upload_file_to_zenodo
from surgeon.models import CVAE
from shutil import copyfile


def create_kipoi_model(model_name: str,
                       model: CVAE,
                       model_path: str,
                       data_path: str,
                       access_token=None):
    path_to_save = os.path.join("~/.kipoi/models/", model_name)

    if os.path.exists(path_to_save):
        raise Exception("Please enter a unique model name")
    if access_token is None:
        raise Exception("You have to enter access token")

    os.makedirs(path_to_save, exist_ok=True)
    model_url, model_md5 = upload_file_to_zenodo(model_path, access_token=access_token, publish=True)
    data_url, data_md5 = upload_file_to_zenodo(data_path, access_token=access_token, publish=True)

    create_network_yaml(model, model_url, model_md5, path_to_save)
    create_dataloader_yaml(model, data_url, data_md5, path_to_save)

    copyfile("./surgeon/kipoi/dataloader.py", f"~/.kipoi/models/{model_name}/")
