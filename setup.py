from pathlib import Path
import os

from setuptools import setup, find_packages

long_description = Path('README.rst').read_text('utf-8')

try:
    from scarches import __author__, __email__, __version__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''
    __version__ = '0.6.1'

# otherwise readthedocs fails
# because somewhere in the dependency tree there is the sklearn deprecated package
os.environ["SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL"] = "True"

setup(name='scArches',
      version=__version__,
      description='Transfer learning with Architecture Surgery on Single-cell data',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/theislab/scarches',
      author=__author__,
      author_email=__email__,
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
        "scanpy[leiden]>=1.6.0", # includes anndata
        "scHPL>=1.0.0",
    	"numpy>=1.19.2",
    	"scipy>=1.5.2",
    	"scikit-learn>=0.23.2",
    	"matplotlib>=3.3.1",
    	"pandas>=1.1.2",
        "torch>=1.8.0",
    	"scvi-tools>=0.12.1",
    	"tqdm>=4.56.0",
    	"requests",
        "gdown",
        "muon",
      ],
      classifiers=[
          "Programming Language :: Python :: 3.9",
          "Programming Language :: Python :: 3.10",
          "Programming Language :: Python :: 3.11",
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          "License :: OSI Approved :: MIT License",
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
      ],
      doc=[
          'sphinx',
          'sphinx_rtd_theme',
          'sphinx_autodoc_typehints',
          'typing_extensions; python_version < "3.8"',
      ],
      )
