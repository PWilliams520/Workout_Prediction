# author        : Esteban, Sarah, Patrick
# course        : CS-691 Data Mining
# name          : main.py
# date          : 2019
# usage         : python3 main.py
# python_version: 3.7
# notes         : Data Mining Project
# ==============================================================================
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

    print(cleaned_up_headers)
    print(cleaned_up_activities)
    print(len(cleaned_up_activities))

    all_hr = [int(x[1]) for x in cleaned_up_activities]
    print(all_hr)

    m_acc = []
    c_acc = []
    for window in range(1, len(all_hr)):
        simple_avg_hr = calculateSimpleMovingAverage(window, all_hr)
        cumulative_avg_hr = calculateCumulativeMovingAverage(window, all_hr)

        # print(moving_avg_hr)
        # print(cumulative_avg_hr)

        moving_hr_accuracy = getAccuracy(all_hr, simple_avg_hr, window)
        cumulative_hr_accuracy = getAccuracy(all_hr, cumulative_avg_hr, window)
        m_acc.append(moving_hr_accuracy)
        c_acc.append(cumulative_hr_accuracy)

    print("len(m_acc):", len(m_acc))
    minimo = min(m_acc)
    print(minimo)
    indice = m_acc.index(minimo)
    print(indice)

    minimo = min(c_acc)
    print(minimo)
    indice = c_acc.index(minimo)
    print(indice)

    print(m_acc[indice])
    print(c_acc[indice])


def getAccuracy(original_set, predicted_set, window):
    distances = []
    index = 0
    for i in range(window, len(original_set)):
        distances.append(abs(original_set[i] - predicted_set[index]))
        index += 1
    if window == 304:
        print(distances)
    return sum(distances) / len(distances)


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
