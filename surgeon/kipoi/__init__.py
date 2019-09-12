import os
from shutil import copyfile

from surgeon.kipoi._data import create_dataloader_yaml
from surgeon.kipoi._model import create_network_yaml
from surgeon.kipoi._upload import upload_file_to_zenodo
from surgeon.models import CVAE


def create_kipoi_model(model_name: str,
                       model: CVAE,
                       upload=False,
                       **kwargs):
    path_to_save = os.path.join("~/.kipoi/models/", model_name)
    model_path = os.path.join(model.model_path, "cvae.h5")

    if os.path.exists(path_to_save):
        raise Exception("Please feed a unique model name")

    if upload:
        access_token = kwargs.get("access_token", None)
        data_path = kwargs.get("data_path", None)

        if access_token is None:
            raise Exception("You have to feed access token")

        if data_path is None:
            raise Exception("You have to feed data path")

        model_url, model_md5 = upload_file_to_zenodo(model_path, access_token=access_token, publish=True)
        data_url, data_md5 = upload_file_to_zenodo(data_path, access_token=access_token, publish=True)

    else:
        model_url = kwargs.get("model_url", None)
        model_md5 = kwargs.get("model_md5", None)

        data_url = kwargs.get("data_url", None)
        data_md5 = kwargs.get("data_md5", None)

        if not model_url or not model_md5:
            raise Exception("You have to feed model_url and model_md5 arguments")

        if not data_url or not data_md5:
            raise Exception("You have to feed model_url and model_md5 arguments")

    os.makedirs(path_to_save, exist_ok=True)
    create_network_yaml(model, model_url, model_md5, path_to_save)
    create_dataloader_yaml(model, data_url, data_md5, path_to_save)

    copyfile("./surgeon/kipoi/dataloader.py", f"~/.kipoi/models/{model_name}/dataloader.py")

    print("model is ready to submit!")
    print("sumitting model...")
