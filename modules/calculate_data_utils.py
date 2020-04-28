"""
Calculate quantities such as magnetic pressure and components of the stress-energy tensor
by reading Athena++ output data files.
"""

# Python modules
import numpy as np
from raw_data_utils import *
from metric_utils import *

check_nan_flag = False

def quality_factor_phi(filename, x1_min=None, x1_max=None, x2_min=None, x2_max=None,
                   x3_min=None, x3_max=None, write_to_file=False):
    """
    filename holds the raw data for one time step
    if write_to_file, will save data to .hdf5 file.
    returns quality factor as defined in XX
    """
    return


def mass_flux(raw_data=None, coords=None, x1_min=None, x1_max=None, **kwargs):
    """
    Calculate the mass flux over radial shells between radii x1_min and x1_max.
    Here mass flux is defined as the integral over theta and phi (with sqrt(-g) Jacobian)
    of rho*u1, where u1 = u^1 is the upper radial component of the 4-velocity
    (NOT the output projected velocity!).

    Note that mass flux is the negative of \dot M as commonly defined
    (e.g. White, Quataert, Gammie 2020 Eq. 6)
    Hence mass_flux > 0 means outflow mass_flux < 0 means inflow (accretion).

    Default is to use raw_data
    (tuple of density, radial 4-velocity, and the coordinates
    e.g. (density, u1, coords),
    where u1 is the radial component of the four velocity,
    e.g. the output of get_four_velocity_from_output
    coords is (x1v, x2v, x3v). Is None if raw_data is None and will be calculated.)
    but if raw_data is None then specifying filename will load the data
    from that file.

    write_to_file will output the mass_flux at every radius to a .txt file
    load_from_file will attempt to load from a pre-existing .txt file first. If the
       file does not exist, it will calculate the values (and write to file)
    """
    write_to_file = kwargs.get("write_to_file", True)

    if raw_data is None:
        to_load = kwargs.get("load_from_file", True)
        reduced_filename = kwargs.get("reduced_filename", None)
        if reduced_filename is None:
            # must have both a time and reduced_data_path specified
            reduced_data_path = kwargs.get("reduced_data_path")
            time = kwargs.get("time")
            # build pathname
            reduced_filename = reduced_data_path + XX

        if to_load and os.path.exists(reduced_filename):
            mass_flux = np.loadtxt(reduced_filename, skiprows=1)
            return mass_flux
        else:
            write_to_file = True

        raw_filename = kwargs.get("raw_filename")
        # XX build path from time, raw_file_path

        # Calculate mass flux
        read_data = read_athdf(raw_filename,
                               quantities=["rho", "vel1", "vel2", "vel3"],
                               x1_min=x1_min, x1_max=x1_max)
        four_velocity = get_four_velocity_from_output((read_data["vel1"], read_data["vel2"], read_data["vel3"]))
        coords = (read_data["x1v"], read_data["x2v"], read_data["x3v"]) # XX will need to chop this to match x1_min, x1_max
        raw_data = (read_data["rho"], four_velocity[1], coords)


    # Here is the actual heart of calculating the mass flux
    (density, u1, coords) = raw_data
    (x1v, x2v, x3v) = coords
    metric = kerrschild(x1v, x2v)
    rhour = density*u1
    integrand = rhour * metric # XX metric needs to be expanded into phi dimension

    phi_integral = np.trapz(integrand, axis=0, x=x3v) # remember indexing is [phi, theta, r]
    mass_flux = np.trapz(phi_integral, axis=0, x=x2v) # remember indexing is [theta, r]

    if write_to_file:
        filename = "mass_flux_t{}".format(times[i])
        np.savetxt(mass_flux, reduced_data_path + filename, header=str(x1v))

    return mass_flux

