===================================================
LBM for gas-solid porous thermal chemical process
===================================================

A lattice Boltzmann method (LBM) codebase for simulating gas–solid
porous media with coupled thermal and chemical processes. This repository
contains the core solver, utilities for geometry and parameter setup,
and documentation with a user guide and installation instructions.

Highlights
----------
- LBM-based fluid solver tailored for porous media
- Coupled heat transport and simple chemical reaction models
- Tools for preprocessing porous geometries and postprocessing results
- Documentation and user guide under docs/

Requirements
------------
The project is written in Python (3.8+ recommended). Typical dependencies
include numerical and scientific Python packages such as:

- numpy
- scipy
- taichi (for parallel computation)
- matplotlib (for plotting / visualization)
- sphinx (for building docs)

See docs/source/installation.rst for full installation instructions and
platform-specific notes.

Quick start
-----------
1. Create and activate a Python virtual environment:

   .. code-block:: bash

      uv venv
      source .venv/bin/activate   # On Windows: .venv\Scripts\activate

2. Install the package in editable mode (if project provides setup/pyproject):

   .. code-block:: bash

      uv pip install -e .

3. Run a simulation under the directory with a yaml configuration file:

   .. code-block:: bash

      lbm

4. After the simulation, visualize the result by pyvista: 

.. code-block:: bash

  pvLbm

Or by paraview which should be installed in your computer.

.. code-block:: bash

  paraLbm



Documentation
-------------
Full documentation and the user guide are in the docs/ tree. See:

- docs/source/user-guide/introduction.rst
- docs/source/installation.rst

To build the documentation locally:

.. code-block:: bash

   cd docs
   make html

The built HTML will be in docs/_build/html.

Development and testing
-----------------------
- Run tests with pytest (if a tests/ directory exists):

  .. code-block:: bash

     pytest -q

- Use flake8 / black / ruff for linting and formatting if configured.

Contributing
------------
Contributions, bug reports and feature requests are welcome. Please follow
these steps:

1. Fork the repository.
2. Create a feature branch for your change.
3. Add tests for your change (where applicable).
4. Open a pull request describing the change.

Please consult CONTRIBUTING.md if present for repository-specific
guidelines.

Authors and credits
-------------------
.. include:: AUTHORS.rst

License
-------
This project is distributed under the terms of the MIT License (or other
license provided in LICENSE). See the LICENSE file for details.

Citing this work
----------------
If you use this code in published research, please cite the repository
or the associated paper (if any). Add citation details here when
available.

Included documentation snippets
-------------------------------
For convenience, parts of the documentation and authorship are included
directly below (these include directives will pull in the files when the
project is viewed on GitHub or when Sphinx builds the docs):

.. include:: docs/source/user-guide/introduction.rst
.. include:: docs/source/installation.rst


.. include :: AUTHORS.rst


.. include :: docs/source/user-guide/introduction.rst
.. include :: docs/source/installation.rst
