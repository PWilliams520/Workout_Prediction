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
    with open(config.ACTIVITY_FILE_NAME, 'r', encoding='utf8') as csv_file:
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
        valid_hr = True
        sub_activity = []
        for index in indexes:
            if activity[index] != config.INVALID_FIELD:
                sub_activity.append(activity[index])
            else:
                valid_hr = not valid_hr
        if valid_hr:
            clean_up_activities.append(sub_activity)

    return clean_up_headers, clean_up_activities


def mergesort(arr):
    n = len(arr)
    if n == 1:
        return arr
    arr1 = arr[0:n // 2]
    arr2 = arr[n // 2:]
    arr1 = mergesort(arr1)
    arr2 = mergesort(arr2)
    return mergesort_aux(arr1, arr2)


def mergesort_aux(arr1, arr2):
    result = []
    n1 = len(arr1)
    n2 = len(arr2)
    while n1 > 0 and n2 > 0:
        if arr1[0] > arr2[0]:
            result.append(arr2[0])
            del arr2[0]
            n2 -= 1
        else:
            result.append(arr1[0])
            del arr1[0]
            n1 -= 1
    while n1 > 0:
        result.append(arr1[0])
        del arr1[0]
        n1 -= 1
    while n2 > 0:
        result.append(arr2[0])
        del arr2[0]
        n2 -= 1
    return result
