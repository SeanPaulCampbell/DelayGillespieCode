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
    reaction_list = Initialize_Classes(gillespie_parameters[0:7])
    initial_state = gillespie_parameters[7]


    [stop_time, burn_in_time, sample_rate] = run_vector  # remove runcount

    signal = Gill.gillespie(reaction_list, stop_time, initial_state)
    signalcsv = path + '{}signal.csv'.format(gillespie_parameters)
    np.savetxt(signalcsv, gillespie_parameters[:7], delimiter=",")  # maybe make this a header
    np.savetxt(signalcsv, signal, delimiter=",")
    burned_in_signal = Post.burn_in_time_series(signal, burn_in_time)  # this needs to be passed by reference.
    uniform_sampling = Post.uniformly_sample(burned_in_signal, sample_rate)  # this needs to be passed by reference.

    peakscsv = path + "{}peaks.csv".format(gillespie_parameters)
    peaks = Post.chop_peaks(uniform_sampling, peakscsv)


    return peaks


def Initialize_ClassesRepresilator(gillespie_parameters):
    [alpha, beta, yr, r0, c0, mu, cv] = gillespie_parameters
    destruction_delay = [0]
    desttoy_protein0 = [-1,0,0]
    mobius_propensity = 0
    reactionHill = 1
    #    dilution0 = Classy.Reaction(np.array(desttoy_protein0, dtype=int), protein in equation, mobius_propensity, [0, beta, 1, 0], distribution_id, destruction_delay)
    dilution0 = Classy.Reaction(np.array(desttoy_protein0, dtype=int), 0, mobius_propensity, [0, beta, 1, 0], 1, destruction_delay)
    dilution1 = Classy.Reaction(np.array([0,-1,0], dtype=int), 1, mobius_propensity, [0, beta, 1, 0], 1, destruction_delay)
    dilution2 = Classy.Reaction(np.array([0,0,-1], dtype=int), 2, mobius_propensity, [0, beta, 1, 0], 1, destruction_delay)

    enzymatic_degradation0 = Classy.Reaction(np.array([-1,0,0], dtype=int), 0, mobius_propensity, [0, yr, r0, 1], 1, destruction_delay) # need to rewrite this so that all molecules are in the base
    enzymatic_degradation1 = Classy.Reaction(np.array([0,-1,0], dtype=int), 1, mobius_propensity, [0, yr, r0, 1], 1, destruction_delay) #  need to make it such that  denominator is R 0 +r 1 (t)+r 2 (t)+r 3 (t)
    enzymatic_degradation2 = Classy.Reaction(np.array([0,0,-1], dtype=int), 2, mobius_propensity, [0, yr, r0, 1], 1, destruction_delay)

    production1 = Classy.Reaction(np.array([0,1,0], dtype=int), 0, reactionHill, [alpha, c0, 2], 0, [mu, mu * cv])
    production0 = Classy.Reaction(np.array([1,0,0], dtype=int), 2, reactionHill, [alpha, c0, 2], 0, [mu, mu * cv])

    production2 = Classy.Reaction(np.array([0,0,1], dtype=int), 1, reactionHill, [alpha, c0, 2], 0, [mu, mu * cv])

    reaction_list = np.array([production1, production0, production2, enzymatic_degradation0, enzymatic_degradation1, enzymatic_degradation2, dilution0,dilution1,dilution2])
    return reaction_list



def Initialize_Classes(gillespie_parameters):
    [alpha, beta, yr, r0, c0, mu, cv] = gillespie_parameters

    dilution = Classy.Reaction(np.array([-1], dtype=int), 0, 0, [0, beta, 1, 0], 1, [0])

    enzymatic_degradation = Classy.Reaction(np.array([-1], dtype=int), 0, 0, [0, yr, r0, 1], 1, [0])

    production = Classy.Reaction(np.array([1], dtype=int), 0, 1, [alpha, c0, 2], 0, [mu, mu * cv])

    reaction_list = np.array([production, enzymatic_degradation, dilution])
    return reaction_list
# Get parallel list running
# RUN FOR LOOP MATH MACHINE

# MAKE FILE THAT  makes code into compiled form compile.py


#Note: keep function defitions above mp.Pool the 2 shall never mix. Function definitions are like an oil spill on top of water
with mp.Pool(safeProcessors) as pool2:
    alpha = [300]
    beta = [.1]
    gamma_r = [80]
    R0 = [1]
    C0 = [10]
    mu = [5]
    cv = [.1]

    stop_time = 500
    burn_time = 50
    sample_rate = 10
    path = "Jan/"
    Path(path).mkdir(parents=True, exist_ok=True)

    #parameter_sets = Gill.list_for_parallelization([alpha, beta, gamma_r, R0, C0, mu, cv, [[0]]])
    initial_state_represilator = np.array([0,50,200], dtype=int)
    #[np.array([0,50,200])]
    #np.array([0,50,200], dtype=int)
    parameter_sets_represilator = Gill.list_for_parallelization([alpha,[.05,.2],gamma_r, R0,C0, [5,10],[.05,.5],[initial_state]])
    parameter_sets = Gill.list_for_parallelization([alpha, [.05, .2], gamma_r, R0, C0, [5, 10], [.05, .5], [[0]]])

    try:
        pool2.starmap(run_pipeline,
                          [(parameter_set, [stop_time, burn_time, sample_rate], path) for parameter_set in parameter_sets_represilator])
    finally:
        pool2.close()
        pool2.join()
