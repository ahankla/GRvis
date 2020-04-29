"""
Calculate quantities such as magnetic pressure and components of the stress-energy tensor
by reading Athena++ output data files.
"""

# Python modules
import numpy as np
import os
from raw_data_utils import *
from metric_utils import *

check_nan_flag = False

def calculate_quality_factor_phi(filename, x1_min=None, x1_max=None, x2_min=None, x2_max=None,
                   x3_min=None, x3_max=None, write_to_file=False):
    """
    filename holds the raw data for one time step
    if write_to_file, will save data to .hdf5 file.
    returns quality factor as defined in XX
    """
    return


def calculate_mass_flux(raw_data, metric=None):
    """
    Calculate the mass flux over radial shells.
    Here mass flux is defined as the integral over theta and phi (with sqrt(-g) Jacobian)
    of rho*u1, where u1 = u^1 is the upper radial component of the 4-velocity
    (NOT the output projected velocity!).

    Note that mass flux is the negative of \dot M as commonly defined
    (e.g. White, Quataert, Gammie 2020 Eq. 6)
    Hence mass_flux > 0 means outflow mass_flux < 0 means inflow (accretion).

    INPUTS:
    - raw_data: tuple of density, radial 4-velocity, and the coordinates
        e.g. (density, u1, coords),
        where u1 is the radial component of the four velocity,
        e.g. the output of get_four_velocity_from_output
        coords is (x1v, x2v, x3v).
    """

    (density, u1, coords) = raw_data
    (x1v, x2v, x3v) = coords

    if metric is None:
        metric = kerrschild(x1v, x2v, x3v)

    rhour = density*u1
    integrand = rhour*metric.jacobian

    # the np.trapz yields a difference in the second or third decimal place
    # compared to just simply summing.
    phi_integral = np.trapz(integrand, axis=0, x=x3v) # remember indexing is [phi, theta, r]
    mass_flux = np.trapz(phi_integral, axis=0, x=x2v) # remember indexing is [theta, r]

    return mass_flux

