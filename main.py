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
    clean_up_data = utils.dropAllLines(header, activities)
    clean_up_headers = clean_up_data[0]
    clean_up_activities = clean_up_data[1]

    print(clean_up_headers)
    print(clean_up_activities)
    print(len(clean_up_activities))


if __name__ == "__main__":
    main()
