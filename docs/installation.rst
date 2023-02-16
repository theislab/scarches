Installation
============


scArches requires Python 3.7 or 3.8. We recommend to use Miniconda.

PyPI
--------


The easiest way to get scArches is through pip using the following command::

    sudo pip install -U scarches


Conda Environment
---------------------


You can also use our environment file. This will create the conda environment 'scarches' with
the required dependencies::

    git clone https://github.com/theislab/scarches
    cd scarches
    conda env create -f envs/scarches_linux.yml
    conda activate scarches


Development
---------------

You can also get the latest development version of scArches from `Github <https://github.com/theislab/scarches/>`_ using the following steps:
First, clone scArches using ``git``::

    git clone https://github.com/theislab/scarches


Then, ``cd`` to the scArches folder and run the install command::

    cd scarches
    python3 setup.py install

On Windows machines you may need to download a C++ compiler if you wish to build from source yourself.

Dependencies
------------

The list of dependencies for scArches can be found in the `requirements.txt <https://github.com/theislab/scarches/blob/master/docs/requirements.txt>`_ file in the repository.

If you run into issues, do not hesitate to approach us or raise a `GitHub issue <https://github.com/theislab/scarches/issues/new/choose>`_.
