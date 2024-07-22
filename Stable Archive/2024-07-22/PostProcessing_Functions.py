#!/usr/bin/env python
import numpy as np
import scipy.signal as sig
import os


def post_process(signal, burn_in_time, sample_rate, chop_size, path, file_name):
    signal = burn_in_time_series(signal, burn_in_time)
    signal = uniformly_sample(signal, sample_rate)
    return chop_peaks(signal, path + file_name + "peaks", chop_size)
    #peak_widths = sig.peak_widths(signal[:,1], signal_peaks)[0]
    #queue_peaks = detect_peaks(signal)
    #complete_signal_peak_data = signal[signal_peaks,:]
    #complete_queue_peak_data = signal[queue_peaks,:]
    #archive_data(complete_queue_peak_data, path + file_name + "queue_peaks_data")


def archive_data(data, file_name):
    np.save(file_name, data)
    os.system("gzip {}".format("'" + file_name + ".npy'"))


def unzip_data(file_name):
    os.system("gunzip {}".format("'" + file_name + "'"))
    data = np.load(file_name[:-3])
    os.system("gzip {}".format("'" + file_name[:-3] + "'"))
    return data


def count_births(signal):
    counter = 0
    for index in range(max(np.shape(signal))-1):
        if signal[index, 1] < signal[index+1,1]:
            counter += 1
    return counter


def burn_in_time_series(signal, burn_in_time):
    temp_signal = signal
    temp_signal[:, 0] = signal[:, 0] - burn_in_time

    new_start_time = np.where(temp_signal[:, 0] > 0)[0][0]
    new_start_state = np.zeros(temp_signal.shape[1])
    new_start_state[1:] = temp_signal[new_start_time - 1, 1:]
    temp_signal[new_start_time - 1, :] = new_start_state
    burned_in_signal = temp_signal[(new_start_time - 1):, :]
    return burned_in_signal


def uniformly_sample(signal, rate=0, number_of_samples=0):
    n = min(np.shape(signal))
    end = signal[np.shape(signal)[0] - 1, 0]
    if rate > 0:
        number_of_samples = int(end * rate)
    elif rate < 0 or number_of_samples < 1:
        raise ValueError("No samples specified or no sampling rate specified")
    uniform_sampling = np.float32(np.zeros([number_of_samples, n]))
    uniform_timestamps = np.linspace(0, end, number_of_samples)
    uniform_sampling[:, 0] = uniform_timestamps
    counter = 0
    for index in range(number_of_samples):
        while counter < max(np.shape(signal)):
            if signal[counter, 0] > uniform_timestamps[index]:
                uniform_sampling[index, 1:n] = signal[counter - 1, 1:n]
                break
            counter += 1
    return uniform_sampling


def is_max_in_window(signal, length_of_signal, window_size):
    def window_checker(index):
        if index <= window_size or index >= length_of_signal - window_size:
            return np.float32(1 + np.random.uniform())
        elif signal[index, 1] < signal[index - window_size, 1] \
                or signal[index, 1] < signal[index + window_size, 1]:
            return np.float32(1 + np.random.uniform())
        else:
            return np.float32(0)

    return window_checker


def compute_optimal_time_window(signal):
    """ first half of the peak detection algorithm """
    n = max(np.shape(signal))
    rows = int(np.ceil(n / 2) - 1)
    lms = np.zeros((rows, n), dtype="float16")
    for x in range(0, rows):
        lms[x, :] = np.array(list(map(is_max_in_window(signal, n, x + 1), range(n))))
    row_sum = np.sum(lms, axis=1)
    gamma = np.where(row_sum == np.amin(row_sum))
    rescaled_lms = np.vsplit(lms, gamma[0] + 1)[0]
    return rescaled_lms


def detect_peaks(signal):
    column_sd = np.std(compute_optimal_time_window(signal), axis=0)
    peaks_index = np.where(column_sd == 0)
    peaks = signal[peaks_index, :]
    peaks = peaks[0, :, :]
    peaks[:,1:] = np.round(peaks[:,1:])
    return peaks


def chop_peaks(signal, filename, chop_size=2000):
    all_data = []
    for index in range(int(max(np.shape(signal))/chop_size)):
        start = index * (chop_size + 1)
        peak_data = detect_peaks(signal[start:(start + chop_size), :])
        np.save(filename + str(index), peak_data)
        all_data.append(peak_data)
    return all_data


def post_process_from_file(zipped_signal, burn_in, sample_rate, window_size=4000):
    os.system("gunzip {} ;".format("'" + zipped_signal + "'"))
    signal = np.load(zipped_signal[:-3])
    os.system("gzip {} &".format("'" + zipped_signal[:-3] + "'"))
    signal = burn_in_time_series(signal[:, :2], burn_in)  # this needs to be passed by reference.
    signal = uniformly_sample(signal, sample_rate)

    peakscsv = zipped_signal[:-13] + "peaks"
    chop_peaks(signal, peakscsv, window_size)


def plot_timeseries(file):
    from matplotlib import pyplot as plt
    os.system("gunzip '" + file + "'")
    signal = np.load("'" + file[:-3] + "'")
    os.system("gzip '" + file[:-3] + "'")
    plt.plot(signal[:, 0], signal[:, 1])
    plt.savefig(file[:-7] + 'plot.png')

##def label_switching(signal, threshold, delay):
##    appendex = list(signal[0,:])
##    appendex.append(0)
##    switching_times = [appendex]
##    side = (signal[0,1] > threshold)
##    for index0 in range(1,len(list(signal[:,0]))):
##        if not (side == (signal[index0,1] > threshold)):
##            index1 = 1
##            while max(0,signal[index0,0]-delay) <= signal[index0-index1,0]:
##                if side == (signal[index0-index1,1] > threshold):
##                    break
##                index1 += 1
##            if max(0,signal[index0,0]-delay) > signal[index0-index1,0]:
##                appendex = list(signal[index0,:])
##                appendex.append(index0)
##                switching_times.append(appendex)
##                side = not side
##    return np.array(switching_times)

def label_switching(signal, low_thresh, high_thresh, initial_side):
    appendex = list(signal[0,:])
    side = initial_side
    appendex.append(side)
    switching_times = [appendex]
    for index0 in range(1,len(list(signal[:,0]))):
        if side and (signal[index0,1] <= low_thresh):
            side = not side
            append_switch(switching_times, signal[index0,:], side)
        elif (not side) and (signal[index0,1] >= high_thresh):
            side = not side
            append_switch(switching_times, signal[index0,:], side)
    return np.array(switching_times)


def append_switch(switching_times, signal_at_timestamp, side):
    appendex = list(signal_at_timestamp)
    appendex.append(side)
    switching_times.append(appendex)
    return 0
