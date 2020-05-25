from scnet.models import CVAE


class scNetNB(CVAE):
    """
        scNet network with NB for loss function class. This class contains the implementation of scNet network.

        Parameters
        ----------
        x_dimension: int
            number of gene expression space dimensions.
        n_conditions: int
            number of conditions used for one-hot encoding.
        z_dimension: int
            number of latent space dimensions.
        task_name: str
            name of the task.

        kwargs:
            `learning_rate`: float
                scNet's optimizer's step size (learning rate).
            `alpha`: float
                KL divergence coefficient in the loss function.
            `eta`: float
                Reconstruction coefficient in the loss function.
            `dropout_rate`: float
                dropout rate for Dropout layers in scNet's architecture.
            `model_path`: str
                path to save model config and its weights.
            `clip_value`: float
                Optimizer's clip value used for clipping the computed gradients.
            `output_activation`: str
                Output activation of scNet which Depends on the range of data.
            `use_batchnorm`: bool
                Whether use batch normalization in scNet or not.
            `architecture`: list
                Architecture of scNet. Must be a list of integers.
            `gene_names`: list
                names of genes fed as scNet's input. Must be a list of strings.
    """

    def __init__(self, x_dimension, n_conditions, task_name="unknown", z_dimension=100, **kwargs):
        kwargs.update({'loss_nb': 'nb', 'beta': 0,
                       "model_name": "cvae_nb", "class_name": "CVAE_NB"})
        super().__init__(x_dimension, n_conditions, task_name, z_dimension, **kwargs)


    @classmethod
    def from_config(cls, config_path, new_params=None, compile=True, construct=True):
        """create ``CVAE_NB`` object from exsiting ``CVAE_NB``'s config file.

        Parameters
        ----------
        config_path: str
            Path to class' config json file.
        new_params: dict, optional
            Python dict of parameters which you wanted to assign new values to them.
        compile: bool
            ``True`` by default. if ``True``, will compile class' model after creating an instance.
        construct: bool
            ``True`` by default. if ``True``, will construct class' model after creating an instance.
        """
        import json
        with open(config_path, 'rb') as f:
            class_config = json.load(f)

        class_config['construct_model'] = construct
        class_config['compile_model'] = compile

        if new_params:
            class_config.update(new_params)

        return cls(**class_config)
