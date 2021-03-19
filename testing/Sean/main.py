#!/usr/bin/env python
import Classes_Gillespie as Classy
import PostProcessing_Functions as Post
import Functions_Gillespie as Gill
import numpy as np
import os
from pathlib import Path
import multiprocessing as mp
safeProcessors = max(1, int(mp.cpu_count() * .8))


# Function Definitions
def run_pipeline(gillespie_parameters, processing_parameters, path):
    [reaction_list, initial_state] = Initialize_Reactions(gillespie_parameters)
    [stop_time, burn_in_time, sample_rate] = processing_parameters

    signal = Gill.gillespie(reaction_list, stop_time, initial_state)
    archive_signal(signal, path + '{}signal.npy'.format(gillespie_parameters),
                   '/home/spcampbe/servers/storage/data\ storage')
    post_process(signal, path + "{}peaks.npy".format(gillespie_parameters),
                 burn_in_time, sample_rate, 3600)


def archive_signal(signal, file_name, storage):
    np.save(file_name, signal)
    os.system("gzip {} ;".format("'" + file_name + "'") +
              " mv {} {} &".format("'" + file_name + ".gz'", storage))


def post_process(signal, file_name, burn_in_time, sample_rate, chop_size=4000):
    signal = Post.burn_in_time_series(signal[:, :2], burn_in_time)
    signal = Post.uniformly_sample(signal, sample_rate)
    Post.chop_peaks(signal, file_name, chop_size)


def get_peak_files(path):
    onlyPeakfiles = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    onlyPeakfiles = [f for f in onlyPeakfiles if "peaks" in f]
    return onlyPeakfiles

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
        mu_range = list(np.linspace(5, 10, 16))
        cv_range = list(np.linspace(0, .5, 16))

        stopping_time = 8300
        burn_time = 300
        sampling_rate = 60
        path_to_raw_data = "2021Feb19/"
        Path(path_to_raw_data).mkdir(parents=True, exist_ok=True)

        parameter_sets = Gill.list_for_parallelization([mu_range, cv_range])
        onlyPeakfiles = [f for f in get_peak_files(path_to_raw_data)
                         if os.path.getsize(os.path.join(path_to_raw_data, f)) > 1]
        for peakfile in onlyPeakfiles:
            for index in range(len(parameter_sets)):
                if str(parameter_sets[index]) in peakfile:
                    del parameter_sets[index]
                    break

        try:
            pool2.starmap(run_pipeline, [(parameter_set, [stopping_time, burn_time, sampling_rate], path_to_raw_data)
                                         for parameter_set in parameter_sets])
        finally:
            pool2.close()
            pool2.join()

    period_mean = np.zeros([len(mu_range), len(cv_range)])
    period_cv = np.zeros([len(mu_range), len(cv_range)])
    amplitude_mean = np.zeros([len(mu_range), len(cv_range)])
    amplitude_cv = np.zeros([len(mu_range), len(cv_range)])

    for peakfile in get_peak_files(path_to_raw_data):
        peaks = np.load("'" + path_to_raw_data + peakfile + "'")
        params = np.array(peakfile[1:-10].split(', '), dtype=float)
        indices = (np.where(mu_range == params[0])[0][0], np.where(cv_range == params[1])[0][0])

        period_mean[indices] = np.mean(peaks[:, 0])
        period_cv[indices] = np.std(peaks[:, 0]) / period_mean[indices]
        amplitude_mean[indices] = np.mean(peaks[:, 2])
        amplitude_cv[indices] = np.std(peaks[:, 2]) / amplitude_mean[indices]

    np.savetxt(path_to_raw_data + "period_mean.csv", period_mean, delimiter=",")
    np.savetxt(path_to_raw_data + "period_cv.csv", period_cv, delimiter=",")
    np.savetxt(path_to_raw_data + "amplitude_mean.csv", amplitude_mean, delimiter=",")
    np.savetxt(path_to_raw_data + "amplitude_cv.csv", amplitude_cv, delimiter=",")
