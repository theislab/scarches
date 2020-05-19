from scnet.models import scNet
from .file import *
from .deposition import *
from .zip import *


def upload_model(model: scNet,
                 deposition_id: str,
                 access_token: str):
    model_path = model.model_path
    output_base_name = f"/tmp/scNet-{model.task_name}"
    output_path = output_base_name + ".zip"
    zip_model_directory(output_path=output_base_name, directory=model_path)
    download_link = upload_file(file_path=output_path,
                                deposition_id=deposition_id,
                                access_token=access_token)
    print("Model has been successfully uploaded")
    return download_link
