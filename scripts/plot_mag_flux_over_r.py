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
    accretion rate at given times over radius.

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
        raw_data = read_athdf(filename, quantities=["Bcc1"])
        sim_time = raw_data["Time"]
        x1v = raw_data["x1v"]
        x2v = raw_data["x2v"]
        x3v = raw_data["x3v"]

        # don't need to load metric every time since the coords stay the same
        if metric is None:
            metric = kerrschild(x1v, x2v, x3v)
        if coords is None:
            coords = (x1v, x2v, x3v)

        mag_flux_over_r = calculate_magnetic_flux_over_radial_shells((raw_data["Bcc1"], coords))

        reduced_data_fp = reduced_data_path + "mag_flux_over_r_at_t{:05}.txt".format(time)
        total_mag_flux = np.trapz(mag_flux_over_r, x=raw_data["x1v"])
        print(total_mag_flux)

        if write_to_file:
            x1vstr = ', '.join(map(str, raw_data["x1v"]))
            np.savetxt(reduced_data_fp, mag_flux_over_r, header=x1vstr)


        label = "$t={:.2f}~GM/c^3$".format(sim_time)

        # ----- Now the actual plotting ------
        # Remember mdot = - mass flux
        plt.plot(raw_data["x1v"], -1.0*mag_flux_over_r, marker="o", markersize=3, label=label)

    plt.xlabel(r"$r/r_g$")
    plt.ylabel(r"$\Phi$")
    titstr = "Magnetic flux through each radial shell"
    plt.legend()
    plt.title(titstr)
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


