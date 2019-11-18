"""
graphs.py
Plots relevant graphs and saves them to the output folder. See README.md for more info.
"""

import pandas as pd
import seaborn
import matplotlib.pyplot as plt
import utils
from dateutil.parser import parse
import csv
import numpy as np
from pandas.plotting import register_matplotlib_converters


def calculateSimpleMovingAverage(window, cleaned_up_activities):
    """
    Calculates the simple moving average of the dataset given the selected window size
    :param window: selected window size to compute averages
    :param cleaned_up_activities: the preprocessed dataset
    :return: the list of moving averages predicted by the model
    """
    moving_avg = []
    for i in range(len(cleaned_up_activities)):
        if window <= i:
            moving_avg.append(sum([x for x in cleaned_up_activities[i - window:i]]) / window)
    return moving_avg


def plot_matrix(matrix):
    """
    Plots the confusion matrix passed in as a list of lists
    :param matrix: the confusion matrix
    :return: the plot of the confusion matrix
    """
    fig, ax = plt.subplots()
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2),
           yticks=np.arange(2),
           xticklabels=["Low", "High"], yticklabels=["Low", "High"],
           title="Average Heart Rate Prediction Confusion Matrix",
           ylabel='Actual label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if False else 'd'
    thresh = max([item for lst in matrix for item in lst]) / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(matrix[i][j], fmt),
                    ha="center", va="center",
                    color="white" if matrix[i][j] > thresh else "black")
    fig.tight_layout()
    return ax


def create_and_save_graphs():
    register_matplotlib_converters()
    activities = utils.loadCSVInfo()
    header = activities[0]
    del activities[0]
    activities = utils.cleanUpData(activities)
    cleaned_up_data = utils.dropAllLines(header, activities)
    cleaned_up_activities = cleaned_up_data[1]

    date_list = [parse(i[0]) for i in cleaned_up_activities]
    hr_list = [int(i[1]) for i in cleaned_up_activities]
    hr_reverse = hr_list[::-1]

    moving_avg_hr = calculateSimpleMovingAverage(15, hr_reverse)
    moving_avg_hr.reverse()

    data_dict = {'date': date_list[:-15], 'Avg Heart Rate': hr_list[:-15], 'Predicted Heart Rate (Moving Average)': moving_avg_hr}
    df = pd.DataFrame(data_dict, columns=['date', 'Avg Heart Rate', 'Predicted Heart Rate (Moving Average)'])
    df = df.set_index('date')

    data_plot = seaborn.lineplot(data=df)
    plt.yticks(np.arange(0, max(hr_list), 20))
    data_plot.set(xlabel="Date (YYYY-MM)", ylabel="Average Heart Rate", title="Average Heart Rate per Workout (Rides/Virtual Rides)")
    plt.setp(data_plot.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
    plt.gcf().set_size_inches(8, 6)
    plt.savefig("output/data_and_MA_predictions.png")
    print("Avg. heart rate and moving average predictions saved to output/data_and_MA_predictions.png")
    plt.cla()

    with open("Accuracies_for_Window_Size_Variations.csv", 'rU') as file:
        window_acc_data = [rec for rec in csv.reader(file, delimiter=':')]

    window_list = [int(i[0]) for i in window_acc_data]
    acc_list = [float(i[1]) for i in window_acc_data]
    window_size_plot = seaborn.lineplot(x=window_list[:-1], y=acc_list[:-1], label="Logistic Regression")
    window_size_plot.set(xlabel="Window Size", ylabel="Accuracy", title="Accuracy of Window Sizes")
    plt.axvline(15, 0, max(acc_list), ls='--', c='green', label="Chosen Window Size (15)")

    with open("ma_window_accuracies.csv", 'rU') as file:
        data = [rec for rec in csv.reader(file, delimiter=' ')]

    window_list = [int(i[1]) for i in data]
    acc_list = [float(i[0]) for i in data]
    seaborn.lineplot(x=window_list, y=acc_list, label="Moving Average")
    plt.legend(loc='lower left')
    plt.savefig("output/window_size_accuracies.png")
    print("Window size accuracies saved to output/window_size_accuracies.png")
    plt.cla()

    plot_matrix([[119, 42], [49, 84]])
    plt.savefig("output/CM_MA.png")
    print("Confusion matrix for moving average saved to output/CM_MA.png")
    plt.cla()

    plot_matrix([[120, 13], [21, 140]])
    plt.savefig("output/CM_LR.png")
    print("Confusion matrix for logistic regression saved to output/CM_LR.png")
    plt.cla()


if __name__ == "__main__":
    create_and_save_graphs()
