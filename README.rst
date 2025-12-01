LBM for gas-solid porous thermal chemical process
=================================================
Author: Zeshi YUE
Version : 1.0
Date : Nov. 9 2025
Installation
------------

* Install the uv package manager from https://docs.astral.sh/uv/

* Clone this repository from github and change to it::

    git clone https://github.com/
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

Introduction
------------
Algorithm
---------
Structure
---------
lbm 2D
++++++
lbm 3D
++++++
cantera tool
+++++++++++++
In order to support chemkin file input, thanks to the open-sourced kinetic software Cantera, we could convert the chemkin file into yaml which is more readable. 
This lbm software could read this yaml to set the thermodynamics data, transport data and reactions.

The cantera could calculate the thermodynamical and transport properties of the mixtures in an implicit way. The coefficients described in the yaml file is not visible in ct.Solution class.
We should analyse the yaml file by ourselves. So there is no need to include cantera package in the project which is mainly used for zero D and one D simulaition of chemical reactions.

**yaml strcture**

refer to https://cantera.org/stable/userguide/input-tutorial.html

1. description

2. generator

3. input-files

4. cantera-version

5. date

6. units

 - length
 - time
 - quantity
 - activation-energy

7.phases : list of phases 

 - name 
 - themo : 

  + for gas : ideal-gas/Redlich-Kwong/Peng-Robinson/plasma
  + for surface and interface : ideal-surface/coverage-dependent-surface/edge
  + for single-species phase : fixed-stoichiometry/electron-cloud/pure-fluid/liquid-water-IAPWS95
  + for ideal solid and liquid : ideal-molar-solution/ideal-condensed/ideal-solution-VPSS/lattice/compound-lattice
  + for non-ideal solid and liquid solution : binary-solution-tabulated/Debye-Huckel
 - elements 
 - species 
 - kinetics : gas
 - transport : mixture-averaged
 - state

8.species : list of species

 - name 
 - composition 
 - equation-of-state

  + model: constant-volume/density-temperature-polynomial/molar-volume-temperature-polynomial/Peng-Robinson/Redlich-Kwong/HKFT/liquid-water-IAPWS95

 - thermo

  + model : NASA7/NASA9/Shomate/constant-cp/piecewise-Gibbs
  + temperature-ranges : two ranges
  + data : depend on the model
  + note 注释

 - transport

  + model 
  + geometry 
  + well-depth 
  + diameter 
  + polarizability 
  + rotational-relaxtion

9. reactions

list[name composition thermo[model temperature-ranges分段 data分段 note注释] transport[model geometry well-depth diameter polarizability rotational-relaxtion]]
# reactions list[equation+] default->rate-constant (duplicate); type:three-body + efficiencies; type:falloff+ low-P-rate-constant+high-P-rate-constant+ Troe+ efficiencies



GEO 2D
++++++++
GEO 3D
++++++++
Validation
----------
2D
++++++++++++
1. Poiseulle
2. Lid-driven
3. Cylinder
4. Catalysis
5. Pyrolysis

3D
++++++++++++



copyright