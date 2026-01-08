"""LBM2D module
================

Provides the 2D Lattice Boltzmann Method (LBM) solver implementation
used for pyrolysis simulations. The main public class is
:class:`LBM2DSolver`.

This module composes several mixin classes that implement different
aspects of the solver: core data structures, evolution operators,
boundary conditions, IO, and initialization.

Notes
-----
The implementation uses Taichi for performance and relies on
data-oriented programming idioms.
"""

import taichi as ti
from ._core import LBM2D_BASE

from ._info import LBM2D_INFO
from ._evolution import LBM2D_EVOLUTION
from ._boundary import LBM2D_BOUNDARY
from ._io import LBM2D_INPUT,LBM2D_OUTPUT
from ._initialize import LBM2D_INITIALIZATION
from ..util.flag import *


@ti.data_oriented
class LBM2DSolver(LBM2D_EVOLUTION,LBM2D_BOUNDARY,LBM2D_INPUT,LBM2D_OUTPUT,LBM2D_INITIALIZATION,LBM2D_INFO,LBM2D_BASE):
    """2D LBM solver.

    Parameters
    ----------
    X : float
        Length of the simulation field in x direction.
    Y : float
        Length of the simulation field in y direction.
    Z : float
        Length of the simulation field in z direction.
    dx : float, optional
        Grid spacing (default is 1).
    dt : float, optional
        Time step (default is 1).
    name : str, optional
        Name identifier for the solver instance (default: ``"default_LBM"``).
    isThermal : bool, optional
        Enable thermal coupling.
    isChemical : bool, optional
        Enable chemical reactions.
    isPoro : bool, optional
        Enable porous media model.
    isRadiation : bool, optional
        Enable radiation heat transfer.

    Attributes
    ----------


    Notes
    -----
    The class aggregates evolution, boundary, IO, initialization and info
    mixins and uses Taichi ``@ti.data_oriented`` for performance.
    """
    def __init__(self, X, Y, Z, dx=1, dt=1, name="default_LBM", isThermal=False, isChemical=False, isPoro=False, isRadiation=False):
        """Initialize the solver.

        See the class docstring for detailed parameter descriptions.
        """
        super().__init__(X, Y, Z, dx, dt, name, isThermal, isChemical, isPoro, isRadiation)
