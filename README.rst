.. image:: https://cdn.cp.adobe.io/content/2/dcx/315ea3d9-927f-477b-ae7f-039540ec026d/rendition/preview.jpg/version/1/format/jpg/dimension/width/size/1200

.. image:: https://travis-ci.com/Olllom/lettuce.svg?branch=master
        :target: https://travis-ci.com/Olllom/lettuce

.. .. image:: https://img.shields.io/pypi/v/lettuce.svg
        :target: https://pypi.python.org/pypi/lettuce

.. image:: https://readthedocs.org/projects/lettuceboltzmann/badge/?version=latest
        :target: https://lettuceboltzmann.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3757641.svg
        :target: https://doi.org/10.5281/zenodo.3757641

.. image:: https://img.shields.io/lgtm/grade/python/g/Olllom/lettuce.svg?logo=lgtm&logoWidth=18
        :target: https://lgtm.com/projects/g/Olllom/lettuce/context:python



Lettuce is a lattice Boltzmann code in Python that provides high level features:

* GPU-accelerated computation based on PyTorch
* Rapid Prototyping in 2D and 3D
* Usage of neuronal networks based on Pytorch with lattice Boltzmann

GPU-accelerated Lattice Boltzmann in Python



Attritubtes
-----------
* Documentation_
* Examples_


.. _Documentation: https://lettuceboltzmann.readthedocs.io
.. _Examples: https://github.com/Olllom/lettuce/tree/master/examples

Installation
------------

* Install the anaconda package manager from www.anaconda.org
* Create a new conda environment and install all dependencies::

    conda create -n lettuce -c pytorch -c conda-forge\
         "pytorch>=1.1" matplotlib pytest click cudatoolkit "pyevtk>=1.1"


* Activate the conda environment::

    conda activate lettuce

* Clone this repository from github
* Change into the cloned directory
* Run the install script::

    python setup.py install

* Run the test cases::

    python setup.py test

* Check out the convergence order, running on CPU::

    lettuce --no-cuda convergence

* For running a CUDA-driven LBM simulation on one GPU omit the `--no-cuda`. If CUDA is not found,
  make sure that cuda drivers are installed and compatible with the installed cudatoolkit
  (see conda install command above).

* Check out the performance, running on GPU::

    lettuce benchmark

Credits
-------
We use the following third-party packages:

* pytorch_
* numpy_
* pytest_
* click_
* matplotlib_
* versioneer_
* pyevtk_


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

.. _pytorch: https://github.com/pytorch/pytorch
.. _numpy: https://github.com/numpy/numpy
.. _pytest: https://github.com/pytest-dev/pytest
.. _click: https://github.com/pallets/click
.. _matplotlib: https://github.com/matplotlib/matplotlib
.. _versioneer: https://github.com/python-versioneer/python-versioneer
.. _pyevtk: https://github.com/pyscience-projects/pyevtk

License
-----------
* Free software: MIT license, as found in the LICENSE_ file.

.. _LICENSE: https://github.com/Olllom/lettuce/blob/master/LICENSE
