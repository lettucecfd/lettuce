.. image:: https://raw.githubusercontent.com/lettucecfd/lettuce/master/.source/img/logo_lettuce_typo.png

.. image:: https://github.com/lettucecfd/lettuce/actions/workflows/CI.yml/badge.svg
        :target: https://github.com/lettucecfd/lettuce/actions/workflows/CI.yml
        :alt: CI Status

.. image:: https://github.com/mcbs/lettuce/actions/workflows/codeql.yml/badge.svg
        :target: https://github.com/lettucecfd/lettuce/actions/workflows/codeql.yml
        :alt: Codeql Status

.. image:: https://readthedocs.org/projects/lettucecfd/badge/?version=latest
        :target: https://lettucecfd.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
        
.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3757641.svg
        :target: https://doi.org/10.5281/zenodo.3757641



GPU-accelerated Lattice Boltzmann Simulations in Python
-------------------------------------------------------

Lettuce is a Computational Fluid Dynamics framework based on the lattice Boltzmann method (LBM).

- **GPU-Accelerated Computation**: Utilizes PyTorch for high performance and efficient GPU utilization.
- **Rapid Prototyping**: Supports both 2D and 3D simulations for quick and reliable analysis.
- **Advanced Techniques**: Integrates neural networks and automatic differentiation to enhance LBM.
- **Optimized Performance**: Includes custom PyTorch extensions for native CUDA kernels.

Resources
---------

- `Documentation`_
- Presentation at CFDML2021 - `Paper`_ | `Preprint`_ | `Slides`_ | `Video`_ | `Code`_

.. _Paper: https://www.springerprofessional.de/en/lettuce-pytorch-based-lattice-boltzmann-framework/19862378
.. _Documentation: https://lettuceboltzmann.readthedocs.io
.. _Preprint: https://arxiv.org/pdf/2106.12929.pdf
.. _Slides: https://drive.google.com/file/d/1jyJFKgmRBTXhPvTfrwFs292S4MC3Fqh8/view
.. _Video: https://www.youtube.com/watch?v=7nVCuuZDCYA
.. _Code: https://github.com/lettucecfd/lettuce-paper


Getting Started
---------------

The following Python code will run a two-dimensional Taylor-Green vortex on a GPU:

.. code:: python

    import torch
    import lettuce as lt

    lattice = lt.Lattice(lt.D2Q9, device='cuda', dtype=torch.float64, use_native=False)  # for running on cpu: device='cpu'
    flow = lt.TaylorGreenVortex2D(resolution=128, reynolds_number=100, mach_number=0.05, lattice=lattice)
    collision = lt.BGKCollision(lattice, tau=flow.units.relaxation_parameter_lu)
    streaming = lt.StandardStreaming(lattice)
    simulation = lt.Simulation(flow=flow, lattice=lattice, collision=collision, streaming=streaming)
    mlups = simulation.step(num_steps=1000)
    print("Performance in MLUPS:", mlups)

More advanced examples_ are available as jupyter notebooks.

Please ensure you have Jupyter installed to run these notebooks.

.. _examples: https://github.com/lettucecfd/lettuce/tree/master/examples

Installation
------------

* Install the anaconda package manager from www.anaconda.org
* Create a new conda environment and activate it::

    conda create -n lettuce
    conda activate lettuce

* Follow the recommendations at https://pytorch.org/get-started/locally/ to install pytorch based on your GPU's CUDA version. To get your CUDA version, do::

    nvcc --version

* You may need to install the nvidia toolkit. You may follow the instructions at: https://developer.nvidia.com/cuda-downloads. You may need to check the compatibility of your NVIDIA driver with the desired CUDA version: https://docs.nvidia.com/deploy/cuda-compatibility/.
For the latest versions (if supported by your GPU) use::

    conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia

* Install the remaining dependencies::

    conda activate lettuce4ad
    conda install -c pytorch -c conda-forge matplotlib pytest click pyevtk  mmh3 h5py scipy pandas numpy

* Clone this repository from github
* Change into the cloned directory
* If you want to only USE lettuce, run the install script::

    python setup.py install

* If you are a developer, do::

    python setup.py develop

* Run the test cases::

    python setup.py test

* Check out the convergence order, running on CPU::

    lettuce --no-cuda convergence

* For running a CUDA-driven LBM simulation on one GPU omit the `--no-cuda`. If CUDA is not found,
  make sure that cuda drivers are installed and compatible with the installed cudatoolkit
  (see conda install command above).

* Check out the performance, running on GPU::

    lettuce benchmark


Citation
--------
If you use Lettuce in your research, please cite the following paper::

    @inproceedings{bedrunka2021lettuce,
      title={Lettuce: PyTorch-Based Lattice Boltzmann Framework},
      author={Bedrunka, Mario Christopher and Wilde, Dominik and Kliemank, Martin and Reith, Dirk and Foysi, Holger and Kr{\"a}mer, Andreas},
      booktitle={High Performance Computing: ISC High Performance Digital 2021 International Workshops, Frankfurt am Main, Germany, June 24--July 2, 2021, Revised Selected Papers},
      pages={40},
      organization={Springer Nature}
    }


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
* h5py_
* mmh3_


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
.. _h5py: https://github.com/h5py/h5py
.. _mmh3: https://github.com/hajimes/mmh3

License
-----------
* Free software: MIT license, as found in the LICENSE_ file.

.. _LICENSE: https://github.com/lettucecfd/lettuce/blob/master/LICENSE

