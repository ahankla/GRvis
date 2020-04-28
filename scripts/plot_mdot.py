import numpy as np


def main(raw_data_path, reduced_data_path, times=None, **kwargs):
    # raw_data_path is the directory where the raw data is housed.
    # It does not include the file names.

    write_to_file = True
    load_from_file = True
    x1_min = None
    x1_max = None

    filenames = kwargs.get("filenames", None)
    if times is None:
        get_all_times(raw_data_path)

    if filenames is None:
        filenames = []
        # construct using paths and times
        for time in times:
            filename = raw_data_path + ".prim.{}.athdf".format(time)
            filenames.append(filename)

    for (i, time) in np.enumerate(times):
        filename = filenames[i]
        raw_data = read_athdf(filename, quantities=["rho", "vel1", "vel2", "vel3"])

        out_vel = (raw_data["vel1"], raw_data["vel2"], raw_data["x3v"])
        metric = kerrschild(raw_data["x1v"], raw_data["x2v"])
        four_velocity = metric.get_four_velocity_from_output(out_vel)
        mass_flux = mass_flux((raw_data["rho"], four_velocity[1]), x1_min, x1_max, write_to_file=write_to_file, load_from_file=load_from_file)

