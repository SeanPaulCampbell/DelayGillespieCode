#!/usr/bin/env python
import multiprocessing as mp
safeProcessors =  max(1, int(mp.cpu_count() * .5))
import Classes_Gillespie as Classy
import PostProcessing_Functions as Post
import Functions_Gillespie as Gill
import numpy as np
from pathlib import Path


# Function Definitions
def run_pipeline(gillespie_parameters, run_vector, path):
    reaction_list = Initialize_Classes(gillespie_parameters[:-1])
    initial_state = gillespie_parameters[-1]
    [stop_time, burn_in_time, sample_rate] = run_vector 
    
    signal = Gill.gillespie(reaction_list, stop_time, initial_state)
    signalcsv = path + '{}signal.csv'.format(gillespie_parameters)
    np.savetxt(signalcsv, gillespie_parameters[:7], delimiter=",")  # maybe make this a header
    np.savetxt(signalcsv, signal, delimiter=",")
    burned_in_signal = Post.burn_in_time_series(signal, burn_in_time)  # this needs to be passed by reference.
    uniform_sampling = Post.uniformly_sample(burned_in_signal, sample_rate)  # this needs to be passed by reference.

    peakscsv = path + "{}peaks.csv".format(gillespie_parameters)
    peaks = Post.chop_peaks(uniform_sampling, peakscsv)

    return peaks


def Initialize_Classes(gillespie_parameters):
    [alpha, beta, yr, r0, c0, mu, cv] = gillespie_parameters
    
    dilution0 = Classy.Reaction(np.array([-1, 0, 0], dtype=int), 0, 'mobius_propensity', [0, beta, 1, 0], 1, [0])
    dilution1 = Classy.Reaction(np.array([0, -1, 0], dtype=int), 1, 'mobius_propensity', [0, beta, 1, 0], 1, [0])
    dilution2 = Classy.Reaction(np.array([0, 0, -1], dtype=int), 2, 'mobius_propensity', [0, beta, 1, 0], 1, [0])

    degradation0 = Classy.Reaction(np.array([-1, 0, 0], dtype=int), 0, 'mobius_propensity', [0, yr, r0, 1], 1, [0]) # need to rewrite this so that all molecules are in the base
    degradation1 = Classy.Reaction(np.array([0, -1, 0], dtype=int), 1, 'mobius_propensity', [0, yr, r0, 1], 1, [0]) #  need to make it such that  denominator is R 0 +r 1 (t)+r 2 (t)+r 3 (t)
    degradation2 = Classy.Reaction(np.array([0, 0, -1], dtype=int), 2, 'mobius_propensity', [0, yr, r0, 1], 1, [0])

    production1 = Classy.Reaction(np.array([0,1,0], dtype=int), 0, 'decreasing_hill_propensity', [alpha, c0, 2], 0, [mu, mu * cv])
    production0 = Classy.Reaction(np.array([1,0,0], dtype=int), 2, 'decreasing_hill_propensity', [alpha, c0, 2], 0, [mu, mu * cv])
    production2 = Classy.Reaction(np.array([0,0,1], dtype=int), 1, 'decreasing_hill_propensity', [alpha, c0, 2], 0, [mu, mu * cv])

    reaction_list = np.array([production1, production0, production2, degradation0, degradation1, degradation2, dilution0,dilution1,dilution2])
    return reaction_list


#Note: keep function defitions above mp.Pool
with mp.Pool(safeProcessors) as pool2:
    alpha = [300]
    beta = [.1]
    gamma_r = [80]
    R0 = [1]
    C0 = [10]
    mu = [5]
    cv = [0]

    stop_time = 500
    burn_time = 50
    sample_rate = 10
    path = "Jan/"
    Path(path).mkdir(parents=True, exist_ok=True)

    initial_state = np.array([0,50,200], dtype=int)
    parameter_sets = Gill.list_for_parallelization([alpha, beta, gamma_r, R0, C0, mu, cv, [initial_state]])
    
    try:
        pool2.starmap(run_pipeline, [(parameter_set, [stop_time, burn_time, sample_rate], path) 
                                     for parameter_set in parameter_sets])
    finally:
        pool2.close()
        pool2.join()
