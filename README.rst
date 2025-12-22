LBM for gas-solid porous thermal chemical process
=================================================
Author: Zeshi YUE

Version : 1.0

Date : Dec. 22 2025

.. include :: docs/source/user-guide/introduction.rst
.. include :: docs/source/installation.rst








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