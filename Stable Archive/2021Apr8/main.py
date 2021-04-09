#!/usr/bin/env python
import Classes_Gillespie as Classy
import Functions_Gillespie as Gill
import numpy as np
import os
from pathlib import Path
import multiprocessing as mp
safeProcessors = max(1, int(mp.cpu_count() * .5))


# Function Definitions
def run_pipeline(gillespie_parameters, stop_time, path):
    [reaction_list, initial_state] = Initialize_Reactions(gillespie_parameters)

    signal = Gill.gillespie(reaction_list, stop_time, initial_state)
    save_signal(signal, path + '{}signal.npy'.format(gillespie_parameters))


def save_signal(signal, file_name):
    np.save(file_name, signal)
    os.system("gzip {}".format("'" + file_name + "'"))


def Initialize_Reactions(delay_parameters):
    [mu, cv] = delay_parameters
    alpha = 250
    beta = .1
    gamma_r = 150
    r0 = 1
    c0 = 10
    initial_vector = np.array([0, 500, 1000], dtype=int)

    dilution0 = Classy.Reaction(np.array([-1, 0, 0], dtype=int), 0,
                                'mobius_propensity', [0, beta, 1, 0], 1, [0])
    dilution1 = Classy.Reaction(np.array([0, -1, 0], dtype=int), 1,
                                'mobius_propensity', [0, beta, 1, 0], 1, [0])
    dilution2 = Classy.Reaction(np.array([0, 0, -1], dtype=int), 2,
                                'mobius_propensity', [0, beta, 1, 0], 1, [0])
    degradation0 = Classy.Reaction(np.array([-1, 0, 0], dtype=int), 0,
                                   'mobius_sum_propensity', [0, gamma_r, r0, 1], 1, [0])
    degradation1 = Classy.Reaction(np.array([0, -1, 0], dtype=int), 1,
                                   'mobius_sum_propensity', [0, gamma_r, r0, 1], 1, [0])
    degradation2 = Classy.Reaction(np.array([0, 0, -1], dtype=int), 2,
                                   'mobius_sum_propensity', [0, gamma_r, r0, 1], 1, [0])
    production1 = Classy.Reaction(np.array([0, 1, 0], dtype=int), 0,
                                  'decreasing_hill_propensity', [alpha, c0, 2], 0, [mu, mu * cv])
    production0 = Classy.Reaction(np.array([1, 0, 0], dtype=int), 2,
                                  'decreasing_hill_propensity', [alpha, c0, 2], 0, [mu, mu * cv])
    production2 = Classy.Reaction(np.array([0, 0, 1], dtype=int), 1,
                                  'decreasing_hill_propensity', [alpha, c0, 2], 0, [mu, mu * cv])

    reaction_list = [production1, production0, production2,
                     degradation0, degradation1, degradation2,
                     dilution0, dilution1, dilution2]
    return [reaction_list, initial_vector]


if __name__ == '__main__':
    with mp.Pool(safeProcessors) as pool2:
        mu_range = list(np.linspace(0, 15, 16))
        cv_range = list(np.linspace(0, 1.5, 16))

        stopping_time = 8300
        path_to_raw_data = "2021Apr/PromiscuousRepressilator/"
        Path(path_to_raw_data).mkdir(parents=True, exist_ok=True)

        parameter_sets = Gill.list_for_parallelization([mu_range, cv_range])
        for file in os.listdir(path_to_raw_data):
            for index in range(len(parameter_sets)):
                if str(parameter_sets[index]) in file:
                    del parameter_sets[index]
                    break
        try:
            pool2.starmap(run_pipeline, [(parameter_set, stopping_time, path_to_raw_data)
                                         for parameter_set in parameter_sets])
        finally:
            pool2.close()
            pool2.join()
    os.system("mv {}* /home/spcampbe/servers/storage/data\ storage/{}".format(path_to_raw_data, path_to_raw_data))


