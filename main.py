# author        : Esteban, Sarah, Patrick
# course        : CS-691 Data Mining
# name          : main.py
# date          : 2019
# usage         : python3 main.py
# python_version: 3.7
# notes         : Data Mining Project
# ==============================================================================
import config
import utils


def main():
    activities = utils.loadCSVInfo()
    header = activities[0]
    del activities[0]

    activities = utils.cleanUpData(activities)
    cleaned_up_data = utils.dropAllLines(header, activities)
    cleaned_up_headers = cleaned_up_data[0]
    cleaned_up_activities = cleaned_up_data[1]
    cleaned_up_activities.reverse()

    if config.DEBUG:
        print(cleaned_up_headers)
        print(cleaned_up_activities)
        print(len(cleaned_up_activities))

    all_hr = [int(x[1]) for x in cleaned_up_activities]
    all_hr_labels = [0 if value <= config.HR_THRESHOLD else 1 for value in all_hr]

    print(all_hr)
    print(all_hr_labels)

    best_window = calculateBestWindow(all_hr)
    print(best_window)

    simple_moving_average = calculateSimpleMovingAverage(best_window, all_hr)
    simple_moving_average_labels = [0 if value <= config.HR_THRESHOLD else 1 for value in simple_moving_average]

    #print(simple_moving_average)
    print(simple_moving_average_labels)

    value = getPredictedAccuracyLabels(all_hr_labels, simple_moving_average_labels, best_window)
    print(value)

    values = calculateBestWindow2(all_hr)

    with open('a.csv', 'w') as f:
        for i in range(0, len(values)):
            f.write(str(values[i][0]) + ', ' + str(values[i][1]))
            f.write('\n')



def calculateBestWindow(all_hr):
    simple_moving_avg_accuracy = []
    for window in range(1, len(all_hr)):
        simple_avg_hr = calculateSimpleMovingAverage(window, all_hr)
        moving_hr_accuracy = getAccuracy(all_hr, simple_avg_hr, window)
        simple_moving_avg_accuracy.append((moving_hr_accuracy, window))

        # cumulative_avg_hr = calculateCumulativeMovingAverage(window, all_hr)
        # cumulative_hr_accuracy = getAccuracy(all_hr, cumulative_avg_hr, window)
        # c_acc.append(cumulative_hr_accuracy)

    simple_moving_avg_accuracy = utils.mergesort(simple_moving_avg_accuracy)

    return simple_moving_avg_accuracy[0][1]


def calculateBestWindow2(all_hr):
    simple_moving_avg_accuracy = []
    for window in range(1, len(all_hr)):
        simple_avg_hr = calculateSimpleMovingAverage(window, all_hr)

        all_hr_labels = [0 if value <= config.HR_THRESHOLD else 1 for value in all_hr]
        simple_moving_average_labels = [0 if value <= config.HR_THRESHOLD else 1 for value in simple_avg_hr]

        moving_hr_accuracy = getPredictedAccuracyLabels(all_hr_labels, simple_moving_average_labels, window)
        simple_moving_avg_accuracy.append((moving_hr_accuracy, window))

        # cumulative_avg_hr = calculateCumulativeMovingAverage(window, all_hr)
        # cumulative_hr_accuracy = getAccuracy(all_hr, cumulative_avg_hr, window)
        # c_acc.append(cumulative_hr_accuracy)

    simple_moving_avg_accuracy = utils.mergesort(simple_moving_avg_accuracy)

    print(simple_moving_avg_accuracy)

    return simple_moving_avg_accuracy


def getAccuracy(original_set, predicted_set, window):
    distances = []
    index = 0
    for i in range(window, len(original_set)):
        distances.append(abs(original_set[i] - predicted_set[index]))
        index += 1
    return sum(distances) / len(distances)


def getPredictedAccuracyLabels(original_set, predicted_set, window):
    index = 0
    right_predictions = 0
    for i in range(window, len(original_set)):
        if original_set[i] == predicted_set[index]:
            right_predictions += 1
        index += 1
    return right_predictions / len(predicted_set)


def calculateSimpleMovingAverage(window, cleaned_up_activities):
    moving_avg = []
    for i in range(len(cleaned_up_activities)):
        if window <= i:
            moving_avg.append(sum([x for x in cleaned_up_activities[i - window:i]]) / window)
    return moving_avg


def calculateCumulativeMovingAverage(window, cleaned_up_activities):
    moving_avg = []
    for i in range(len(cleaned_up_activities)):
        if window <= i:
            moving_avg.append(sum([x for x in cleaned_up_activities[0:i]]) / i)
    return moving_avg


if __name__ == "__main__":
    main()
