import os

import yaml
from collections import OrderedDict
from scnet.models import CVAE


def create_dataloader_yaml(network: CVAE,
                           data_url: str = None,
                           data_md5: str = None,
                           path_to_save: str = "./dataloader.yaml"):
    data = OrderedDict(
        defined_as='dataloader.KipoiDatasetWrapper',
        args=OrderedDict(
            weights=OrderedDict(
                url=data_url,
                md5=data_md5,
            ),
        ),

        info=OrderedDict(
            authors=[
                OrderedDict(
                    name="Mohsen Naghipourfar",
                    github="Naghipourfar",
                    email="mohsen.naghipourfar@gmail.com",
                ),
                OrderedDict(
                    name="Mohammad Lotfollahi",
                    github="M0hammadL",
                    email="mohammad.lotfollahi@helmholtz-muenchen.de",
                ),
            ],
            doc="CVAE model",
            licence="MIT",
        ),

        dependencies=OrderedDict(  # TODO: Add more depenedencies
            conda=[
                "python=3.6",
                "bioconda::scanpy"
            ],
            pip=[
                "keras>=2.2.0",
            ],
        ),

        output_schema=OrderedDict(
            inputs=[
                OrderedDict(
                    name="genes",
                    shape=(network.x_dim,),
                ),
                OrderedDict(
                    name="study",
                    shape=(network.n_conditions,),
                    doc="one hot encoded vector of batches (studies)",
                ),
                OrderedDict(
                    name="size_factors",
                    shape=(1,),
                )
            ],
            targets=[
                OrderedDict(
                    name="predicted_genes",
                    shape=(network.x_dim,),
                ),
            ]
        )
    )

    with open(os.path.join(path_to_save, "dataloader.yaml"), 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    print(f"DataLoader YAML has been saved to {os.path.abspath(path_to_save)}")
