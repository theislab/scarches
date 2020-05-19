import os

import requests


def download_file(link: str,
                  save_path: str = None,
                  make_dir: bool = False
                  ):
    from urllib.request import urlretrieve

    if make_dir:
        path = os.path.dirname(save_path)
        os.makedirs(path, exist_ok=True)
    else:
        if not os.path.isdir(save_path):
            raise ValueError("`save_path` is not a valid path. You may want to try setting `make_dir` to True.")

    if not os.path.exists(save_path):
        print(f"Downloading...", end="\t")
        file_path, http_response = urlretrieve(link, save_path)
        print(f"Finished! File has been successfully saved to {file_path}.")
    else:
        file_path, http_response = save_path, None
        print("File already exists!")

    return file_path, http_response


def upload_file(file_path: str,
                deposition_id: str,
                access_token: str):
    file_name = file_path.split("/")[-1]
    data = {
        'filename': file_name,
    }

    files = {'file': open(file_path, 'rb')}

    r = requests.post(f'https://zenodo.org/api/deposit/depositions/{deposition_id}/files',
                      params={'access_token': access_token}, data=data, files=files)
    r_dict = r.json()

    if r.status_code != 201:
        raise Exception(r_dict['message'])

    filename = r_dict['filename']
    download_link = f"https://zenodo.org/record/{deposition_id}/files/{filename}?download=1"
    return download_link

