# Physics-inspired Equivariant Descriptors of Non-bonded Interactions - Auxiliary Code Repository

[![arXiv](https://img.shields.io/badge/arXiv-2308.13208-B31B1B.svg)](https://arxiv.org/abs/2308.13208)
[![Materials
Cloud](https://img.shields.io/badge/Materials%20Cloud-10.24435/materialscloud:23--99-AACAFB.svg)](https://doi.org/10.24435/materialscloud:23-99)

The repository includes codes used for the work "Physics-inspired Equivariant
Descriptors of Non-bonded Interactions" available at https://arxiv.org/abs/2308.13208.

Besides the provided code this repository brings together multiple software project
written in lab-cosmo:

- https://github.com/lab-cosmo/metatensor as a data storage format for atomistic machine
  learning;
- https://github.com/luthaf/rascaline to compute LODE-based representations;
- https://github.com/lab-cosmo/equisolve to computing machine learning models based on
  metatensor objects.

## Installation

You'll need a C++ compiler, CMake, and a [Rust](htpps://rustup.rs/) compiler installed
on your machine. Then, in a fresh Python environment (virtualenv or conda), run the
following command to install the code and all dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements_rascaline.txt
```

## Usage

The [examples](examples) folder contains Python source code to train and evaluate the
linear ad well as neural network (nn) potential based on LODE descripotors. For the
examples in this repository we take a small subset of the dimer dataset from the work.
The complete data set is available for download at [Materials
Cloud](https://doi.org/10.24435/materialscloud:23-99).

Additionally, the [examples](examples) folder contains Python classes to generate
splined radial integrals for
[rascaline](https://luthaf.fr/rascaline/latest/references/api/python/utils/splines.html)
