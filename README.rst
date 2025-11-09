LBM for gas-solid porous thermal chemical process
=================================================
Author: Zeshi YUE
Version : 1.0
Date : Nov. 9 2025
Installation
------------

* Install the uv package manager from https://docs.astral.sh/uv/

* Clone this repository from github and change to it::

    git clone https://github.com/lettucecfd/lettuce
    cd lettuce

* Create a new virtual environment and activate it::

    uv venv
    source .venv/bin/activate

* The `pyproject.toml` file currently requires at least **CUDA 12.4** (we successfully tested CUDA 12.4, 12.6, 12.8 and 13.0). If your GPU does not support this version, you may need to downgrade it. Please note that we cannot guarantee the maintenance for older CUDA versions.

* If you want to only **use** lettuce, run the installation (from the lettuce base directory!).::

    uv pip install .

* If you do not have access to a GPU or an older GPU, you can install the CPU-only version or older CUDA supported versions of lettuce by using the following commands. (cpu, cu124, cu126)::

    uv pip install ".[cpu]"

* If you are a **developer**, add the changeable-installation-flag (`-e`)::

    uv pip install -e .

* Check out the convergence order, running on CPU::

    lbm_pyro --no-cuda convergence

* For running a CUDA-driven LBM simulation on one GPU omit the `--no-cuda`. If CUDA is not found, make sure that cuda drivers are installed and compatible with the installed cudatoolkit.

* Check out the performance, running on GPU::

    lbm_pyro benchmark

* Run the test cases::

    pytest tests
## Introduction
## Algorithm

## Structure

## Validation
### 2D
#### Poiseulle
#### Lid-driven
#### Cylinder
#### Catalysis
#### Pyrolysis
### 3D
### 2D axial



copyright