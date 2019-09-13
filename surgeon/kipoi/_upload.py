import json

import requests


def upload_file_to_zenodo(file_path, access_token=None,
                          publish=True):
    if access_token is None:
        raise Exception("You have to feed access token")
    r = requests.get('https://zenodo.org/api/deposit/depositions',
                     params={'access_token': access_token})
    
    deposition_id = r.json()[0]['id']
    file_name = file_path.split("/")[-1]
    data = {
        'filename': file_name,
        'title': 'scNet upload',
        'creators': [{'name': 'Naghipourfar, Mohsen',
                      'affiliation': 'Zenodo'}]

    }

    files = {'file': open(file_path, 'rb')}

    r = requests.post(f'https://zenodo.org/api/deposit/depositions/{deposition_id}/files',
                      params={'access_token': access_token}, data=data, files=files)
    print(r)
    print(r.json())
    r_status = r.json()

    md5 = r_status['checksum']
    link = r_status['links']['download']
    if publish:
        requests.post(f'https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish',
                      params={'access_token': access_token})

    return link, md5
