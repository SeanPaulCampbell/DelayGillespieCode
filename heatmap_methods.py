import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


def collect_heatmap_data(filename, axis_labels):
    heat_data = np.genfromtxt(filename, delimiter=',')
    collected_data = pd.DataFrame(heat_data, index=np.round(axis_labels[1], 2), columns=np.round(axis_labels[0], 2))
    data_mean = np.mean(list(heat_data))
    return [collected_data, data_mean]


def make_heatmap(data, center_of_data, axis_titles):
    heat_map = sb.heatmap(data, center=center_of_data)
    heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
    heat_map.set_xticklabels(heat_map.get_xticklabels(), rotation=30)
    heat_map.invert_yaxis()
    plt.xlabel(axis_titles[0])
    plt.ylabel(axis_titles[1])


def print_heatmap_to_file(title):
    plt.title(title)
    plt.savefig(title + '.png')


def generate_standard_heatmaps(model_name, parameter_ranges):
    filenames = ['period_mean.csv', 'period_cv.csv', 'amplitude_mean.csv', 'amplitude_cv.csv']
    titles = [' Period Mean', ' Period CV', ' Amplitude Mean', ' Amplitude CV']
    for index in range(4):
        [data, center] = collect_heatmap_data(filenames[index], parameter_ranges)
        make_heatmap(data, center, ['CV of delay', 'Mean of delay'])
        print_heatmap_to_file(model_name + titles[index])
        plt.close()
