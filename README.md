# DataMiningProject
[Repo for CSC-691 Data Mining project](https://github.com/stbamb/DataMiningProject.git)

Patrick Williams, Esteban Murillo, Sarah Parsons

November 18, 2019

## References:

* ### Moving Average Code:

main.py

The simple moving average algorithm takes two parameters; a window and a list of observations. In our case these observations correspond to HR values. Next, we focus on a specific amount of observations at at time, this amount is dictated by our window parameter. We calculate the average of the observations within the window range and make that calculation our next prediction. Naturally, this window keeps moving until we have reached the end of our observation list and our full prediction set.

* In order to run the code, execute the 'main.py' file. Make sure to have the 'activities.csv' file in the same directory. In order to change code behavior modify variables in the 'config.py' file accordingly.  

* ### Logistic Regression Code:

DMProject.ipynb

The logistic regression algorithm uses a predefined feature set (Distance, Elevation, HRSS, and Avg Pace) for a given window size to predict average heart rate for a set of observations (based on the size of the test set). The model is fit to the training set first, then used to predict average heart rate for each test instance. 

* Please run the Jupyter notebook (DMProject.ipynb) to view the results from the Logistic Regression model built for the dataset provided in activities.csv.


* ### Code for Graphs:

Graph_Data.py

Running this file will automatically generate the following graphs and save them to the output folder:
* A graph of the average heart rate for each workout over time. Overlaid is a graph of the predicted heart rate for each workout using the simple moving average model, with a window size of 15.
* A graph of the binary (High/Low) heart rate prediction accuracies of each window size for both the simple moving average and logistic regression models. Threshold for High/Low classification is 154 bpm.
* Confusion matrices of both models for all binary predictions.

* ### Report:

Please see Data_Mining_Project_Milestone_3___CSC691.pdf for the final report.
