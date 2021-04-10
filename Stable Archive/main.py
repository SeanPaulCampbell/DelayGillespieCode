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
    archive_signal(signal, path + '{}signal.npy'.format(gillespie_parameters),
                   "/home/spcampbe/servers/storage/data\ storage/" + path)


def archive_signal(signal, file_name, storage):
    np.save(file_name, signal)
    os.system("gzip {} ;".format("'" + file_name + "'") +
              " mv {} {} &".format("'" + file_name + ".gz'", storage))


def Initialize_Reactions(delay_parameters):
    [mu, cv] = delay_parameters
    factor = 2
    alpha_r = 5000
    alpha_a = 2 * alpha_r
    beta = .1
    gamma_r = 200
    gamma_a = 440
    r0 = 1
    c0 = c1 = 50
    initial_vector = np.array([0, 0], dtype=int)

    dilution0 = Classy.Reaction(np.array([-1, 0], dtype=int), 0,
                                'mobius_propensity', [0, beta, 1, 0], 1, [0])
    dilution1 = Classy.Reaction(np.array([0, -1], dtype=int), 1,
                                'mobius_propensity', [0, beta, 1, 0], 1, [0])
    degradation0 = Classy.Reaction(np.array([-1, 0], dtype=int), 0,
                                   'mobius_propensity', [0, gamma_r, r0, 1], 1, [0])
    degradation1 = Classy.Reaction(np.array([0, -1], dtype=int), 1,
                                   'mobius_propensity', [0, gamma_a, r0, 1], 1, [0])
    production0 = Classy.Reaction(np.array([1, 0], dtype=int), 0,
                                  'dual_feedback_decreasing_hill_propensity', [alpha_r, c0, c1, factor, 2],
                                  'gamma_distribution', [mu, mu * cv])
    production1 = Classy.Reaction(np.array([0, 1], dtype=int), 1,
                                  'dual_feedback_increasing_hill_propensity', [alpha_a, c0, c1, factor, 2],
                                  'trivial_distribution', [0])

    reaction_list = [production0, production1,
                     degradation0, degradation1,
                     dilution0, dilution1]
    return [reaction_list, initial_vector]


if __name__ == '__main__':
    with mp.Pool(safeProcessors) as pool2:
        mu_range = list(np.linspace(0, 15, 16))
        cv_range = list(np.linspace(0, 1.5, 16))

        stopping_time = 8300
        path_to_raw_data = "2021Apr/dualFeedback/"
        Path(path_to_raw_data).mkdir(parents=True, exist_ok=True)

        parameter_sets = Gill.list_for_parallelization([mu_range, cv_range])
        for file in os.listdir("/home/spcampbe/servers/storage/data storage/" + path_to_raw_data):
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

