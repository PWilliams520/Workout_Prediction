# author        : Esteban, Sarah, Patrick
# course        : CS-691 Data Mining
# name          : main.py
# date          : 2019
# usage         : python3 main.py
# python_version: 3.7
# notes         : Data Mining Project
# ==============================================================================
import pandas as pd

import config


def main():
    df = pd.read_csv(config.ACTIVITY_FILE_NAME)

    df = df.rename(columns={"Avg HR (bpm)": "AvgHR"})

    df = df[df.AvgHR != '-']
    types = ['Ride', 'VirtualRide']

    df = df[df.Type.isin(types)]
    df = df.reset_index(drop=True)

    df = df[['Date', 'AvgHR']]

    df = df.iloc[::-1]

    print(df)

    rolling_mean = df.AvgHR.rolling(window=304).mean()

    exponential = df.AvgHR.ewm(span=50, adjust=False).mean()



    pd.set_option('display.max_columns', None)  # or 1000
    pd.set_option('display.max_rows', None)  # or 1000

    print(rolling_mean)

    #print(exponential)


if __name__ == "__main__":
    main()
