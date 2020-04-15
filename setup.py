from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.readlines()[1]

setup(name='scnet',
      version='1.0',
      description='Transfer learning with Architecture Surgery on Single-cell data',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/theislab/scnet',
      author='Mohammad Lotfollahi, Mohsen Naghipourfar',
      author_email='mohammad.lotfollahi@helmholtz-zentrum.de, naghipourfar@ce.sharif.edu',
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ],
      )