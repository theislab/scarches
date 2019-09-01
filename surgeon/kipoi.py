import os
import yaml

from surgeon.models import CVAE


def create_network_yaml(network: CVAE,
                        model_url: str = None,
                        model_md5: str = None,
                        path_to_save: str = "./surgeon.yaml"):
    data = dict(
        defined_as='kipoi.model.KerasModel',
        args=dict(
            weights=dict(
                url=model_url,
                md5=model_md5,
            ),
        ),

        default_dataloader=None,

        info=dict(
            authors=[
                dict(
                    name="Mohsen Naghipourfar",
                    github="Naghipourfar",
                    email="mohsen.naghipourfar@gmail.com",
                ),
                dict(
                    name="Mohammad Lotfollahi",
                    github="M0hammadL",
                    email="mohammad.lotfollahi@helmholtz-muenchen.de",
                ),
            ],
            doc="CVAE model",
            cite_as=None,
            trained_on="Dataset X. held-out Studies A, B, C",
            licence="MIT",
        ),

        dependencies=dict(  # TODO: Add more depenedencies
            conda=[
                "python=3.6",
                "bioconda::scanpy"
            ],
            pip=[
                "keras>=2.2.0",
            ],
        ),

        schema=dict(
            inputs=[
                dict(
                    name="gene counts",
                    shape=(None, network.x_dim),
                ),
                dict(
                    name="study",
                    shape=(None, network.n_conditions),
                    doc="one hot encoded vector of batches (studies)",
                ),
                dict(
                    name="size factors",
                    shape=(None, 1),
                )
            ],
            targets=[
                dict(
                    name="predicted counts",
                    shape=(None, network.x_dim),
                ),
            ]
        )
    )

    with open(path_to_save, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    print(f"Model YAML has been saved to {os.path.abspath(path_to_save)}!")
