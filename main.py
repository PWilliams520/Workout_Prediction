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

    print(cleaned_up_headers)
    print(cleaned_up_activities)
    print(len(cleaned_up_activities))

    all_hr = [int(x[1]) for x in cleaned_up_activities]
    print(all_hr)

    moving_avg_hr = calculateMovingAverage(config.MOVING_AVG_WINDOW, all_hr)
    print("moving_avg_hr:\n", moving_avg_hr)
    print(len(moving_avg_hr))


def calculateMovingAverage(window, cleaned_up_activities):
    moving_avg = []
    for i in range(len(cleaned_up_activities)):
        if i < window:
            moving_avg.append(-1)
        else:
            moving_avg.append(sum([x for x in cleaned_up_activities[i - window:i]]) / window)
    return moving_avg


if __name__ == "__main__":
    main()
