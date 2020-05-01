import numpy as np
import sys
sys.path.append('../modules/')
from raw_data_utils import *
from metric_utils import *
from calculate_data_utils import *
import matplotlib.pyplot as plt
import argparse

def main(config, specs, raw_data_path, reduced_data_path, times, **kwargs):
    """
    Currently, a very bare-bones method to calculate and plot the
    angular momentum flux at given times over radius.

    INPUTS:
    - config: the configuration of the Athena++ executable (LH naming convention)
    - specs: the specifications of the desired simulation (LH naming convention)
    - raw_data_path: the directory where the raw data is housed
        ...it does not include the file names.
        .athdf filename will be constructed from config and specs.
    - reduced_data_path: the directory where the reduced data should be saved,
        which is needed both to write to the file (if calculating from scratch)
        and to load from (if not calculating from scratch)
    - times: the timesteps to be loaded, i.e. an array of integers.

    TO DO:
    - add option to bypass config/specs
    - add option to time average
    - add option to load from file instead of calculating
    - make legend over time nicer

    """

    load_from_file = kwargs.get("load_from_file", True)
    write_to_file = kwargs.get("write_to_file", True)

    metric = None
    coords = None

    if not os.path.exists(reduced_data_path):
        os.makedirs(reduced_data_path)
    # bleh
    # if times is None:
        # get_all_times(raw_data_path)


    for time in times:
        # ----- First have to get data ------
        filename = raw_data_path + config + "_" + specs + ".prim.{:05}.athdf".format(time)
        # just load mdot quantities
        raw_data = read_athdf(filename, quantities=["rho", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3"])
        out_vel = (raw_data["vel1"], raw_data["vel2"], raw_data["vel3"])
        out_B = (raw_data["Bcc1"], raw_data["Bcc2"], raw_data["Bcc3"])
        sim_time = raw_data["Time"]
        x1v = raw_data["x1v"]
        x2v = raw_data["x2v"]
        x3v = raw_data["x3v"]

        if metric is None:
            metric = kerrschild(x1v, x2v, x3v)
        if coords is None:
            coords = (x1v, x2v, x3v)

        four_vel = metric.get_four_velocity_from_output(out_vel)
        proj_b = metric.get_proj_bfield_from_outputB_fourV(out_B, four_vel)
        # calculate mass flux
        mass_flux_over_r = calculate_mass_flux_over_radial_shells((raw_data["rho"], four_vel[1], coords))
        # calculate T^1_3
        Trphi = calculate_Trphi((four_vel, proj_b, coords))
        # Remember Jdot = - Trphi flux
        ang_mom_flux_over_r = -1.0*calculate_flux_over_radial_shells((Trphi, coords))

        reduced_data_fp = reduced_data_path + "norm_ang-mom_flux_over_r_at_t{:05}.txt".format(time)
        norm_ang_mom_flux = ang_mom_flux_over_r/mass_flux_over_r

        if write_to_file:
            x1vstr = ', '.join(map(str, x1v))
            np.savetxt(reduced_data_fp, norm_ang_mom_flux, header=x1vstr)


        label = "$t={:.2f}~GM/c^3$".format(sim_time)

        # ----- Now the actual plotting ------
        plt.plot(x1v, norm_ang_mom_flux, marker="o", markersize=3, label=label)

    plt.xlabel(r"$r/r_g$")
    plt.ylabel(r"$\dot J/\dot M$")
    titstr = "Angular Momentum flux (normalized to mass flux) through each radial shell"
    plt.legend()
    plt.title(titstr)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    # Easily adaptable to running from command line (batch)
    # or stand-alone

    # ---- Stand-alone -----
    raw_data_path = "/mnt/c/Users/liaha/scratch/"
    config = "1.1.1-torus2_b-gz2"
    specs = "a0beta500torBeta_br32x32x64rl2x2"
    times = np.arange(26, 27)

    reduced_data_path = "/mnt/c/Users/liaha/research/projects/ellie/reduced_data/constBeta/"
    main(config, specs, raw_data_path, reduced_data_path, times)

    # ---- with command line arguments ----
    # parser = argparse.ArgumentParser()
    # parser.add_argument('config',
                        # help='configuration of the athena++ executable')
    # parser.add_argument('specs',
                        # help='specifications for this particular run')
    # parser.add_argument('raw_data_path',
                        # help='location of the raw .athdf files')
    # parser.add_argument('reduced_data_path',
                        # help='location of the reduced data, where mdot data will be saved')
    # parser.add_argument('times',
                        # help='times over which mdot should be calculated')
    # args = parser.parse_args()
    ## Need to add kwargs
    # main(**vars(args))


