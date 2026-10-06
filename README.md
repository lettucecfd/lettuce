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

Lettuce is published on PyPI as **`lettucecfd`**; the import name and the command line tool stay `lettuce`.
Python 3.12 or newer is required.

```console
pip install lettucecfd
```

or, in a project managed with [uv](https://docs.astral.sh/uv/):

```console
uv add lettucecfd
```

This installs the default PyTorch build from PyPI. To select a specific build (CPU only, or a particular CUDA
version), see the section *Depending on lettuce from another project* below.

Check the installation by computing the convergence order on the CPU:

```console
lettuce --no-cuda convergence
```

For a CUDA-driven simulation on one GPU, omit `--no-cuda`, e.g. to measure the performance:

```console
lettuce benchmark
```

If CUDA is not found, make sure that CUDA-capable GPU drivers are installed and compatible with the CUDA version
of the installed PyTorch build.

## Installing from source

To work on lettuce itself, install the [uv](https://docs.astral.sh/uv/) package manager, clone the repository and
let uv set up the environment, choosing one of the hardware extras `cpu`, `cu124`, `cu126`, `cu128` or `cu130`:

```console
git clone https://github.com/lettucecfd/lettuce
cd lettuce
uv sync --extra cpu    # or cu124, cu126, cu128, cu130
```

This installs lettuce in editable mode together with the development tools, so code changes take effect
immediately. Run the test suite with:

```console
uv run --extra cpu pytest tests
```

We successfully tested CUDA 12.4, 12.6, 12.8 and 13.0. See [CONTRIBUTING.md](https://github.com/lettucecfd/lettuce/blob/master/CONTRIBUTING.md) for details on the
development setup.

## Depending on lettuce from another project

The distribution is published as **`lettucecfd`**; the import name stays
`lettuce`. Note that the extras (`cpu`, `cu124`, `cu126`, `cu128`, `cu130`) only constrain the PyTorch *version* —
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
