import os

import requests


def download_file(link: str,
                  save_path: str = None,
                  make_dir: bool = False
                  ):
    """Downloads the file in the ``link`` and saves it in ``save_path``.

        Parameters
        ----------
        link: str
            Direct downloadable link.
        save_path: str
            Path with the name and extension of downloaded file.
        make_dir: bool
            Whether to make the ``save_path`` if it does not exist in the system.

        Returns
        -------
        file_path: str
            Full path with name and extension of downloaded file.
        http_response: :class:`~http.client.HTTPMessage`
            ``HttpMessage`` object containing status code and information about the http request.
    """
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
    """Downloads the file in the ``link`` and saves it in ``save_path``.

        Parameters
        ----------
        file_path: str
            Full path with the name and extension of the file you want to upload.
        deposition_id: str
            ID of a deposition in your Zenodo account.
        access_token: str
            Your Zenodo Access token.

        Returns
        -------
        file_path: str
            Full path with name and extension of downloaded file.
        http_response: :class:`~http.client.HTTPMessage`
            ``HttpMessage`` object containing status code and information about the http request.
    """
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
