#!/usr/bin/env python
import Classes_Gillespie as Classy
import Functions_Gillespie as Gill
import numpy as np
import os
from pathlib import Path
import pickle as pkl
import time as t
import datetime as dt
import multiprocessing as mp
safeProcessors = max(1, int(mp.cpu_count() * .5) - 1)


# Function Definitions
def run_pipeline(gillespie_parameters, processing_parameters, work_path, storage_path):
    [reaction_list, initial_state, system_size] = Initialize_Reactions(gillespie_parameters)
    [number_of_transitions, low_threshold, high_threshold] = processing_parameters
    signal = Gill.gillespie_transitions(reaction_list, initial_state, low_threshold, high_threshold, number_of_transitions, [], system_size)
    archive_signal(signal, work_path + '{}transitions.pkl'.format(gillespie_parameters), storage_path)


def archive_signal(signal, file_name, storage):
    os.system("touch " + file_name.replace(" ",r"\ "))
    with open(file_name, 'wb') as handle:
        pkl.dump(signal, handle, protocol=pkl.HIGHEST_PROTOCOL)
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


'''
[reacts, vec_init, size] = Initialize_Reactions([4,0.0])
signal = Gill.gillespie(reacts, 100000, vec_init)
os.system("touch test.pkl")
with open("test.pkl", "wb") as file:
    pkl.dump(signal, file)
os.system("gzip test.pkl")
'''
'''
#run_pipeline([1,0],   [4, 8.4, 30])
#run_pipeline([2,0],   [4, 8.4, 30])
#run_pipeline([2,.05], [4, 8.4, 30])
#run_pipeline([2,.1],  [4, 8.4, 30])
[reaction_list, initial_state, system_size] = Initialize_Reactions([4, .3])
last_state = initial_state
last_queue_state = []
check = 0
while check < 2:
    signal = Gill.gillespie(reaction_list, 100000, last_state, last_queue_state)
    if check == 0:
        for sig in signal:
            if sig["state"][0] >= 30:
                check += 1
                break
    else:
        for sig in signal:
            if sig["state"][0] < 5:
                check += 1
                break
    last_signal = signal[-1]
    last_queue = np.array(last_signal["queue"]) - last_signal["time"]
    last_queue_state = [(time, np.array([1], dtype=int)) for time in last_queue]
    last_state = last_signal["state"]


with open('god_i_hope_this_works.pickle', 'wb') as handle:
    pkl.dump(signal, handle, protocol=pkl.HIGHEST_PROTOCOL)


'''

if __name__ == '__main__':
    with mp.Pool(safeProcessors) as pool2:
        mu_range = [5]
        cv_range = list(np.round(np.linspace(0,0.6,61),2))
        number_of_transitions = 76
        batches = 150
        low_threshold = 8
        high_threshold = 30
        paths_to_raw_data = ["2024-10-23/batch{}/".format(batch) for batch in range(batches)] ### edit to today's date and current batch
        paths_to_storage = ["/home/spcampbe/servers/storage/data_storage/" + path for path in paths_to_raw_data]
        for batch in range(batches):
            Path(paths_to_raw_data[batch]).mkdir(parents=True, exist_ok=True)
            Path(paths_to_storage[batch]).mkdir(parents=True, exist_ok=True)

        parameter_sets = Gill.list_for_parallelization([mu_range, cv_range])
        pool_data = []
        for batch in range(batches):
            pool_data = pool_data + [{"parameters" : parameter_set, "batch" : batch} for parameter_set in parameter_sets]

        for batch in range(batches):
            existing_files = [f for f in get_files(paths_to_storage[batch], "transitions")
                              if os.path.getsize(os.path.join(paths_to_storage[batch], f)) > 1]
            for file in existing_files:
                index_extract = [(pool["batch"] == batch) for pool in pool_data]
                for index in [i for i,x in enumerate(index_extract) if x]:
                    if str(pool_data[index]["parameters"]) in file:
                        del pool_data[index]
                        break

        try:
            start_time = t.time()
            pool2.starmap(run_pipeline, [(pool["parameters"], [number_of_transitions, low_threshold, high_threshold],
                                          paths_to_raw_data[pool["batch"]], paths_to_storage[pool["batch"]]) for pool in pool_data])
            print(t.time()-start_time)
        finally:
            pool2.close()
            pool2.join()


