#!/usr/bin/env python
import Classes_Gillespie as Classy
import PostProcessing_Functions as Post
import Functions_Gillespie as Gill
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join, getsize
from pathlib import Path
import psutil
import multiprocessing as mp
safeProcessors = max(1, int(mp.cpu_count() * .8))


# Function Definitions
def run_pipeline(gillespie_parameters, run_vector, path):
    try:
        reaction_list = Initialize_Classes(gillespie_parameters[:-1])
        initial_state = gillespie_parameters[-1]
        [stop_time, burn_in_time, sample_rate] = run_vector

        signal = Gill.gillespie(reaction_list, stop_time, initial_state)
        signalcsv = path + '{}signal.csv'.format(gillespie_parameters)
        np.savetxt(signalcsv, gillespie_parameters[:7], delimiter=",")  # maybe make this a header
        np.savetxt(signalcsv, signal, delimiter=",")
        signal = Post.burn_in_time_series(signal[:, :2], burn_in_time)  # this needs to be passed by reference.
        signal = Post.uniformly_sample(signal, sample_rate)

        peakscsv = path + "{}peaks.csv".format(gillespie_parameters)
        Post.chop_peaks(signal, peakscsv, 4000)
    except:
        print("Total, available, used, free virtual memory, active, inactive ", file=path + 'error.log')
        print("error" + str(gillespie_parameters) + " psutil memory " + str(psutil.virtual_memory()),
              file=path + 'error.log')


def Initialize_Classes(gillespie_parameters):
    [alpha, beta, yr, r0, c0, mu, cv] = gillespie_parameters

    dilution0 = Classy.Reaction(np.array([-1, 0, 0], dtype=int), 0, 'mobius_propensity', [0, beta, 1, 0], 1, [0])
    dilution1 = Classy.Reaction(np.array([0, -1, 0], dtype=int), 1, 'mobius_propensity', [0, beta, 1, 0], 1, [0])
    dilution2 = Classy.Reaction(np.array([0, 0, -1], dtype=int), 2, 'mobius_propensity', [0, beta, 1, 0], 1, [0])

    degradation0 = Classy.Reaction(np.array([-1, 0, 0], dtype=int), 0, 'mobius_propensity', [0, yr, r0, 1], 1, [0])
    degradation1 = Classy.Reaction(np.array([0, -1, 0], dtype=int), 1, 'mobius_propensity', [0, yr, r0, 1], 1, [0])
    degradation2 = Classy.Reaction(np.array([0, 0, -1], dtype=int), 2, 'mobius_propensity', [0, yr, r0, 1], 1, [0])

    production1 = Classy.Reaction(np.array([0, 1, 0], dtype=int), 0, 'decreasing_hill_propensity', [alpha, c0, 2], 0,
                                  [mu, mu * cv])
    production0 = Classy.Reaction(np.array([1, 0, 0], dtype=int), 2, 'decreasing_hill_propensity', [alpha, c0, 2], 0,
                                  [mu, mu * cv])
    production2 = Classy.Reaction(np.array([0, 0, 1], dtype=int), 1, 'decreasing_hill_propensity', [alpha, c0, 2], 0,
                                  [mu, mu * cv])

    reaction_list = np.array(
        [production1, production0, production2, degradation0, degradation1, degradation2, dilution0, dilution1,
         dilution2])
    return reaction_list


# Note: keep function definitions above mp.Pool
with mp.Pool(safeProcessors) as pool2:
    alpha = [250]
    beta = [.1]
    gamma_r = [150]
    R0 = [1]
    C0 = [10]
    mu = list(np.linspace(5, 10, 16))
    cv = list(np.linspace(0, .5, 16))
    
    stop_time = 8300
    burn_time = 300
    sample_rate = 10
    path_to_raw_data = "7Feb2021/"
    Path(path_to_raw_data).mkdir(parents=True, exist_ok=True)
    
    initial_vector = np.array([0, 500, 1000], dtype=int)
    parameter_sets = Gill.list_for_parallelization([alpha, beta, gamma_r, R0, C0, mu, cv, [initial_vector]])
    onlyPeakfiles = [f for f in listdir(path_to_raw_data) if isfile(join(path_to_raw_data, f))]
    onlyPeakfiles = [f for f in onlyPeakfiles if "peaks.csv" in f]
    onlyPeakfiles = [f for f in onlyPeakfiles if getsize(join(path_to_raw_data, f)) > 1]
    for peakfile in onlyPeakfiles:
        for index in range(len(parameter_sets)):
            if peakfile[:-9] == str(parameter_sets[index]):
                del parameter_sets[index]
                break
    
    try:
        pool2.starmap(run_pipeline, [(parameter_set, [stop_time, burn_time, sample_rate], path_to_raw_data)
                                     for parameter_set in parameter_sets])
    finally:
        pool2.close()
        pool2.join()

period_mean = np.zeros([len(mu), len(cv)])
period_cv = np.zeros([len(mu), len(cv)])
amplitude_mean = np.zeros([len(mu), len(cv)])
amplitude_cv = np.zeros([len(mu), len(cv)])

onlyPeakfiles = [f for f in listdir(path_to_raw_data) if isfile(join(path_to_raw_data, f))]
onlyPeakfiles = [f for f in onlyPeakfiles if "peaks.csv" in f]
for peakfile in onlyPeakfiles:
    peaks = np.array(pd.read_csv(path_to_raw_data + peakfile, header=None))
    params = np.array(peakfile[1:-10].split(', ')[-5:-3], dtype=float)
    indices = (np.where(mu == params[0])[0][0], np.where(cv == params[1])[0][0])

    period_mean[indices] = np.mean(peaks[:, 0])
    period_cv[indices] = np.std(peaks[:, 0]) / period_mean[indices]
    amplitude_mean[indices] = np.mean(peaks[:, 2])
    amplitude_cv[indices] = np.std(peaks[:, 2]) / amplitude_mean[indices]

np.savetxt(path_to_raw_data + "period_mean.csv", period_mean, delimiter=",")
np.savetxt(path_to_raw_data + "period_cv.csv", period_cv, delimiter=",")
np.savetxt(path_to_raw_data + "amplitude_mean.csv", amplitude_mean, delimiter=",")
np.savetxt(path_to_raw_data + "amplitude_cv.csv", amplitude_cv, delimiter=",")
