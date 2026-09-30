![lettuce](https://raw.githubusercontent.com/lettucecfd/lettuce/master/.source/img/logo_lettuce_typo.png)

[![CI Status](https://github.com/lettucecfd/lettuce/actions/workflows/CI.yml/badge.svg)](https://github.com/lettucecfd/lettuce/actions/workflows/CI.yml)
[![Documentation Status](https://readthedocs.org/projects/lettucecfd/badge/?version=latest)](https://lettucecfd.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3757641.svg)](https://doi.org/10.5281/zenodo.3757641)

# GPU-accelerated Lattice Boltzmann Simulations in Python

Lettuce is a Computational Fluid Dynamics framework based on the lattice Boltzmann method (LBM).

- **GPU-Accelerated Computation**: Utilizes PyTorch for high performance and efficient GPU utilization.
- **Rapid Prototyping**: Supports both 2D and 3D simulations for quick and reliable analysis.
- **Advanced Techniques**: Integrates neural networks and automatic differentiation to enhance LBM.
- **Optimized Performance**: Includes custom PyTorch extensions for native CUDA kernels.

## Resources

- [Documentation](https://lettuceboltzmann.readthedocs.io)
- Presentation at CFDML2021 -
  [Paper](https://www.springerprofessional.de/en/lettuce-pytorch-based-lattice-boltzmann-framework/19862378) |
  [Preprint](https://arxiv.org/pdf/2106.12929.pdf) |
  [Slides](https://drive.google.com/file/d/1jyJFKgmRBTXhPvTfrwFs292S4MC3Fqh8/view) |
  [Video](https://www.youtube.com/watch?v=7nVCuuZDCYA) |
  [Code](https://github.com/lettucecfd/lettuce-paper)

## Getting Started

To find some very simple examples of how to use lettuce, please have a look at the
[examples](https://github.com/lettucecfd/lettuce/tree/master/examples). These will guide you through lettuce's main
features. Please ensure you have Jupyter installed to run the Jupyter notebooks.

## Installation

- Install the uv package manager from <https://docs.astral.sh/uv/>

- Clone this repository from github and change to it:

  ```console
  git clone https://github.com/lettucecfd/lettuce
  cd lettuce
  ```

- Create a new virtual environment and activate it:

  ```console
  uv venv
  source .venv/bin/activate
  ```

- The `pyproject.toml` file currently requires at least **CUDA 12.4** (we successfully tested CUDA 12.4, 12.6, 12.8 and
  13.0). If your GPU does not support this version, you may need to downgrade it. Please note that we cannot guarantee
  the maintenance for older CUDA versions.

- Run the install command, depending on your needs (run one of the three options below):

  1. use lettuce (no development) with GPU support:

     ```console
     uv pip install .
     ```

  2. use lettuce (no development) with CPU only or specific older CUDA versions (if you do not have access to a GPU or
     an older GPU) use (cpu, cu124, cu126):

     ```console
     uv pip install ".[cpu]"
     ```

  3. use and **develop** lettuce (code changes take effect in program execution): use the changeable-installation-flag
     (`-e`):

     ```console
     uv pip install -e .
     ```

- Check out the convergence order, running on CPU:

  ```console
  lettuce --no-cuda convergence
  ```

- For running a CUDA-driven LBM simulation on one GPU omit the `--no-cuda`. If CUDA is not found, make sure that
  CUDA-capable GPU drivers are installed and compatible with the installed cudatoolkit (check cuda version number).

- Check out the performance, running on GPU:

  ```console
  lettuce benchmark
  ```

- Run the test cases:

  ```console
  pytest tests
  ```

## Depending on lettuce from another project

The distribution is published as **`lettucecfd`**; the import name stays
`lettuce`. Note that the extras above only constrain the PyTorch *version* —
uv's index configuration is not carried in the published package metadata, so
`uv add lettucecfd[cu128]` on its own installs the default PyTorch build from
PyPI. To get a specific CUDA build, copy the index configuration into your own
`pyproject.toml`:

```toml
[project]
dependencies = ["lettucecfd[cu128]"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

Substitute `cu124`, `cu126`, `cu130` or `cpu` as needed — the extra and the
index URL have to name the same variant.

## Citation

If you use Lettuce in your research, please cite the following paper:

```bibtex
@inproceedings{bedrunka2021lettuce,
  title={Lettuce: PyTorch-Based Lattice Boltzmann Framework},
  author={Bedrunka, Mario Christopher and Wilde, Dominik and Kliemank, Martin and Reith, Dirk and Foysi, Holger and Kr{\"a}mer, Andreas},
  booktitle={High Performance Computing: ISC High Performance Digital 2021 International Workshops, Frankfurt am Main, Germany, June 24--July 2, 2021, Revised Selected Papers},
  pages={40},
  organization={Springer Nature}
}
```

## Credits

We use the following third-party packages:

- [pytorch](https://github.com/pytorch/pytorch)
- [numpy](https://github.com/numpy/numpy)
- [pytest](https://github.com/pytest-dev/pytest)
- [click](https://github.com/pallets/click)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [setuptools-scm](https://github.com/pypa/setuptools-scm)
- [pyevtk](https://github.com/pyscience-projects/pyevtk)
- [h5py](https://github.com/h5py/h5py)
- [mmh3](https://github.com/hajimes/mmh3)

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.

## License

- Free software: MIT license, as found in the [LICENSE](https://github.com/lettucecfd/lettuce/blob/master/LICENSE) file.
