# author        : Esteban, Sarah, Patrick
# course        : CS-691 Data Mining
# name          : main.py
# date          : 2019
# usage         : python3 main.py
# python_version: 3.7
# notes         : Data Mining Project
# ==============================================================================
import csv

import config


def loadCSVInfo():
    rows = []
    with open(config.ACTIVITY_FILE_NAME, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            rows.append(row)
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            rows[i][j] = rows[i][j]
    return rows


def cleanUpData(activities):
    ride_activities = []
    for activity in activities:
        if activity[2] == config.ACTIVITY_TYPE_TARGET1 or activity[2] == config.ACTIVITY_TYPE_TARGET2:
            ride_activities.append(activity)
    return ride_activities


def dropAllLines(header, activities):
    clean_up_headers = []
    clean_up_activities = []
    indexes = []
    index = 0
    for data in header:
        if data == config.ACTIVITY_FEATURE_TARGET1 or data == config.ACTIVITY_FEATURE_TARGET2:
            indexes.append(index)
            clean_up_headers.append(header[index])
        index += 1
    for activity in activities:
        sub_activity = []
        for index in indexes:
            sub_activity.append(activity[index])
        clean_up_activities.append(sub_activity)

    return clean_up_headers, clean_up_activities
