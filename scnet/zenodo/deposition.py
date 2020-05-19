import json
import requests


def create_deposition(access_token: str,
                      upload_type: str,
                      title: str,
                      description: str,
                      **kwargs):
    url = "https://zenodo.org/api/deposit/depositions"
    headers = {"Content-Type": "application/json"}

    data = {"metadata": {"upload_type": upload_type,
                         'title': title,
                         'description': description,
                         **kwargs}}

    r = requests.post(url,
                      params={'access_token': access_token},
                      data=json.dumps(data),
                      headers=headers)
    if r.status_code == 201:
        print("New Deposition has been successfully created!")
        return str(r.json()['id'])
    else:
        raise Exception(r.json()['message'])


def update_deposition(deposition_id: str,
                      access_token: str,
                      metadata: dict):
    url = f"https://zenodo.org/api/deposit/depositions/{deposition_id}?access_token={access_token}"
    headers = {"Content-Type": "application/json"}

    r = requests.put(url, data=json.dumps(metadata), headers=headers)

    if r.status_code == 200:
        print("Deposition has been successfully updated!")
    else:
        raise Exception(r.json()['message'])


def delete_deposition(deposition_id: str,
                      access_token: str):
    r = requests.delete(f'https://zenodo.org/api/deposit/depositions/{deposition_id}',
                        params={'access_token': access_token})

    if r.status_code == 201:
        print(f"Deposition with id = {deposition_id} has been successfullu deleted!")
    else:
        raise Exception(r.json()['message'])


def publish_deposition(deposition_id: str,
                       access_token: str):
    r = requests.post(f'https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish',
                      params={'access_token': access_token})
    if r.status_code == 202:
        print(f"Deposition with id = {deposition_id} has been successfully published!")
    else:
        raise Exception(r.json()['message'])


def get_all_deposition_ids(access_token: str):
    r = requests.get('https://zenodo.org/api/deposit/depositions',
                     params={'access_token': access_token})

    if r.status_code != 200:
        raise Exception(r.json()['message'])

    deposition_ids = []
    for deposition_dict in r.json():
        deposition_ids.append(str(deposition_dict['id']))

    return deposition_ids

