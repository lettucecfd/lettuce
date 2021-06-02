.. image:: https://raw.githubusercontent.com/lettucecfd/lettuce/master/.source/img/logo_lettuce_typo.png

.. image:: https://travis-ci.com/lettucecfd/lettuce.svg?branch=master
        :target: https://travis-ci.com/lettucecfd/lettuce

.. image:: https://img.shields.io/pypi/v/lettuce.svg
        :target: https://pypi.python.org/pypi/lettuce

.. image:: https://readthedocs.org/projects/lettuceboltzmann/badge/?version=latest
        :target: https://lettuceboltzmann.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3757641.svg
        :target: https://doi.org/10.5281/zenodo.3757641

.. image:: https://img.shields.io/lgtm/grade/python/g/lettucecfd/lettuce.svg?logo=lgtm&logoWidth=18
        :target: https://lgtm.com/projects/g/lettucecfd/lettuce/context:python


GPU-accelerated Lattice Boltzmann Simulations in Python
-------------------------------------------------------

Lettuce is a Computational Fluid Dynamics framework based on the lattice Boltzmann method (LBM).

It provides

* GPU-accelerated computation based on PyTorch
* Rapid Prototyping in 2D and 3D
* Usage of neural networks and automatic differentiation within LBM


Getting Started
---------------

The following Python code will run a two-dimensional Taylor-Green vortex on a GPU:

.. code:: python

    import torch
    from lettuce import BGKCollision, StandardStreaming, Lattice, D2Q9, TaylorGreenVortex2D, Simulation

    device = "cuda:0"   # for running on cpu: device = "cpu"
    dtype = torch.float32

    lattice = Lattice(D2Q9, device, dtype)
    flow = TaylorGreenVortex2D(resolution=256, reynolds_number=10, mach_number=0.05, lattice=lattice)
    collision = BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = StandardStreaming(lattice)
    simulation = Simulation(flow=flow, lattice=lattice,  collision=collision, streaming=streaming)
    mlups = simulation.step(num_steps=1000)

    print("Performance in MLUPS:", mlups)


More advanced examples_ are available as jupyter notebooks:

* `A First Example`_
* `Decaying Turbulence`_

.. _examples: https://github.com/lettucecfd/lettuce/tree/master/examples
.. _A First Example: https://github.com/lettucecfd/lettuce/tree/master/examples/A_first_example.ipynb
.. _Decaying Turbulence: https://github.com/lettucecfd/lettuce/tree/master/examples/DecayingTurbulence.ipynb

A complete documentation is available here_.

.. _here: https://lettuceboltzmann.readthedocs.io


Installation
------------

* Install the anaconda package manager from www.anaconda.org
* Create a new conda environment and install all dependencies::

    conda create -n lettuce -c pytorch -c conda-forge\
         "pytorch>=1.2" matplotlib pytest click cudatoolkit "pyevtk>=1.2"


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

.. _LICENSE: https://github.com/lettucecfd/lettuce/blob/master/LICENSE

