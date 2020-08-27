=======
lettuce
=======
.. image:: https://travis-ci.com/Olllom/lettuce.svg?branch=master
        :target: https://travis-ci.com/Olllom/lettuce

.. .. image:: https://img.shields.io/pypi/v/lettuce.svg
        :target: https://pypi.python.org/pypi/lettuce

.. image:: https://readthedocs.org/projects/lettuceboltzmann/badge/?version=latest
    :target: https://lettuceboltzmann.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3757641.svg
   :target: https://doi.org/10.5281/zenodo.3757641

GPU-accelerated Lattice Boltzmann in Python

* Free software: MIT license
* Documentation: https://lettuceboltzmann.readthedocs.io

Features
--------
* Single-GPU performance (2D): 650 MLUPS on V100


Getting Started
---------------

* Install the anaconda package manager from www.anaconda.org
* Create a new conda repository and install all dependencies::

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

* Check out the performance, running on CPU::

    lettuce benchmark


A first example
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


Next steps
----------
* Jonas Latt's approach of storing f_i-w_i instead of f_i, for better numerical accuracy in single/(half) precision;
  this can be added as a different Lattice class.
* Standard Streaming and BGK collision as C++ functions, as an example and for testing performance gains
  https://pytorch.org/tutorials/advanced/cpp_extension.html
* Multi-block lattices.
* Semi-Lagrangian streaming step.


Future Ideas
------------
* Utilize multiple CPUs. Starting point: https://github.com/pytorch/pytorch/issues/9873
* Utilize MPI to scale across multiple nodes. Starting point: https://pytorch.org/tutorials/intermediate/dist_tuto.html


Credits
-------
We use the following third-party packages:

* pytorch
* numpy
* pytest
* click
* matplotlib
* versioneer
* pyevtk


This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
