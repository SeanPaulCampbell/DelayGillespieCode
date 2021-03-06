#!/usr/bin/env python

import Classes_Gillespie as Classy
import PostProcessing_Functions as Post
import Functions_Gillespie as Gill
import numpy as np


def run_pipeline(gillespie_parameters, run_vector, path):
    reaction_list = Initialize_Classes(gillespie_parameters[0:7])
    initial_state = gillespie_parameters[7]

    initial_state = np.array([initial_state], dtype=int)
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


def Initialize_Classes(gillespie_parameters):
    [alpha, beta, yr, r0, c0, mu, cv] = gillespie_parameters

    dilution = Classy.Reaction(np.array([-1], dtype=int), 0, 0, [0, beta, 1, 0], 1, [0])

    enzymatic_degradation = Classy.Reaction(np.array([-1], dtype=int), 0, 0, [0, yr, r0, 1], 1, [0])

    production = Classy.Reaction(np.array([1], dtype=int), 0, 1, [alpha, c0, 2], 0, [mu, mu * cv])

    reaction_list = np.array([production, enzymatic_degradation, dilution])
    return reaction_list


alpha = 300
beta = .1
gamma_r = 80
R0 = 1
C0 = 10
mu = 5
cv = .1

stop_time = 500
burn_time = 50
sample_rate = 10

run_pipeline([alpha, beta, gamma_r, R0, C0, mu, cv, [0]], [stop_time, burn_time, sample_rate], "")
