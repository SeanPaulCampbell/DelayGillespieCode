#!/usr/bin/env python
import Classes_Gillespie as Classy
import Functions_Gillespie as Gill
import numpy as np
import os
from pathlib import Path
import time as t
import datetime as dt
import multiprocessing as mp
safeProcessors = max(1, int(mp.cpu_count() * .5) - 1)


# Function Definitions
def run_pipeline(gillespie_parameters, processing_parameters, work_path, storage_path):
    [reaction_list, initial_state, system_size] = Initialize_Reactions(gillespie_parameters[:-1])
    [stop_time, burn_in_time, sample_rate] = processing_parameters

    signal = Gill.gillespie(reaction_list, stop_time, initial_state, system_size)
    archive_signal(signal, work_path + '{}signal.npy'.format(gillespie_parameters), storage_path)


def archive_signal(signal, file_name, storage):
    np.save(file_name, signal)
    os.system("gzip {} ;".format("'" + file_name + "'") +
              " mv {} {} &".format("'" + file_name + ".gz'", storage.replace(" ",r"\ ")))


def get_files(path, string="signal"):
    only_files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    only_files = [f for f in only_files if string in f]
    return only_files


def Initialize_Reactions(parameters):
    [mu, cv] = parameters
    alpha = 5
    beta = 20
    gamma = np.log(2)
    c = 19
    b = 10
    exponent = 10
    initial_vector = np.array([0], dtype=int)

    dilution = Classy.Reaction(np.array([-1], dtype=int), [0], 'mobius_propensity', [0, gamma, 1, 0], 1, [0])
    production_const = Classy.Reaction(np.array([1], dtype=int), [0], 'mobius_propensity', [alpha, 0, 1, 0], 'gamma_distribution', [mu, cv * mu])
    production_delayed = Classy.Reaction(np.array([1], dtype=int), [0], 'increasing_hill_propensity', [beta, c, exponent, 1], 'gamma_distribution', [mu, cv * mu])
    reaction_list = [dilution, production_const, production_delayed]
    return [reaction_list, initial_vector, 1]


if __name__ == '__main__':
    with mp.Pool(safeProcessors) as pool2:
        mu_range = [1,2]
        cv_range = list(np.round(np.linspace(0,2,51),2))
        runs = 10
        run_range = list(range(runs))
        stopping_time = 100050
        burn_time = 50      ### This parameter is used only in post processing
        sampling_rate = 60  ### This parameter is used only in post processing
        path_to_raw_data = "2024-07-22/" ### edit to today's date!
        path_to_storage = "/home/spcampbe/servers/storage/data storage/" + path_to_raw_data
        Path(path_to_raw_data).mkdir(parents=True, exist_ok=True)
        Path(path_to_storage).mkdir(parents=True, exist_ok=True)

        parameter_sets = Gill.list_for_parallelization([mu_range, cv_range, run_range])
        existing_files = [f for f in get_files(path_to_storage)
                          if os.path.getsize(os.path.join(path_to_storage, f)) > 1]
        for file in existing_files:
            for index in range(len(parameter_sets)):
                if str(parameter_sets[index]) in file:
                    del parameter_sets[index]
                    break

        start_time = t.time()
        try:
            pool2.starmap(run_pipeline, [(parameter_set, [stopping_time, burn_time, sampling_rate],
                                          path_to_raw_data, path_to_storage) for parameter_set in parameter_sets])
        finally:
            print(t.time()-start_time)
            pool2.close()
            pool2.join()

'''
mu_range = [0,1,2]
cv_range = [0,1]
system_size_range = [1,4,9]
stopping_time = 10050
burn_time = 50        ### This parameter is used only in post processing
sampling_rate = 60    ### This parameter is used only in post processing
path_to_raw_data = "2024-06-28/"
path_to_storage = "/home/spcampbell/Documents/Research/data storage/" + path_to_raw_data
Path(path_to_raw_data).mkdir(parents=True, exist_ok=True)
Path(path_to_storage).mkdir(parents=True, exist_ok=True)
parameter_sets = Gill.list_for_parallelization([mu_range, cv_range, system_size_range])
existing_files = [f for f in get_files(path_to_storage)
                  if os.path.getsize(os.path.join(path_to_storage, f)) > 1]
for file in existing_files:
    for index in range(len(parameter_sets)):
        if str(parameter_sets[index]) in file:
            del parameter_sets[index]
            break

for parameter_set in parameter_sets:
    run_pipeline(parameter_set, [stopping_time, burn_time, sampling_rate], path_to_raw_data, path_to_storage)

'''
