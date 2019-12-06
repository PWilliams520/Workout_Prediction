# DataMiningProject
[Repo for CSC-691 Data Mining project](https://github.com/stbamb/DataMiningProject.git)

Patrick Williams, Esteban Murillo, Sarah Parsons

December 6, 2019

## References:

* ### Moving Average Code:

main.py

The simple moving average algorithm takes two parameters; a window and a list of observations. In our case these observations correspond to HR values. Next, we focus on a specific amount of observations at at time, this amount is dictated by our window parameter. We calculate the average of the observations within the window range and make that calculation our next prediction. Naturally, this window keeps moving until we have reached the end of our observation list and our full prediction set.

* In order to run the code, execute the 'main.py' file. Make sure to have the 'activities.csv' file in the same directory. In order to change code behavior modify variables in the 'config.py' file accordingly.  

* ### Classification Code:

classification.py

The Logistic Regression and Random Forest algorithms use a predefined feature set (Distance, Elevation, HRSS, and Avg Pace) for a given window size to predict high or low average heart rate for a set of observations (based on the size of the test set). The model is fit to the training set first, then used to predict high or low average heart rate for each test instance. 

* Please run the classification.py script to view the results from the Logistic Regression and Random Forest models built for the dataset provided in activities.csv.

* ### Regression Code:

regression.py

The Gradient Boosting Regression, Random Forest Regression, and Linear Regression algorithms use a predefined feature set (Distance, Elevation, HRSS, and Avg Pace) for a given window size to predict average heart rate for a set of observations (based on the size of the test set). The model is fit to the training set first, then used to predict average heart rate for each test instance. 

* Please run the regression.py script to view the results from the Gradient Boosting Regression and Random Forest Regression models built for the dataset provided in activities.csv.

* ### Clustering Code:

k_means.py

This file automatically runs 2-Means clustering on the Updated_Data.csv file included in the repo. The features used for clustering are Workout Duration, Distance, and Calories Burned.

The output clusters are written to the file 2_means_results.csv. The user can then open this .csv file and filter the cluster column to observe the results.

* ### Code for Graphs:

Graph_Data.py

Running this file will automatically generate the following graphs and save them to the output folder:
* A graph of the average heart rate for each workout over time. Overlaid is a graph of the predicted heart rate for each workout using the simple moving average model, with a window size of 15.
* A graph of the binary (High/Low) heart rate prediction accuracies of each window size for both the simple moving average and logistic regression models. Threshold for High/Low classification is 154 bpm.
* Confusion matrices of both models for all binary predictions.


* ### Report:

Please see Data_Mining_Project_Milestone_4_CSC691.pdf for the final report.

* ### TeX and bib Files:

Please see Data_Mining_Project_Milestone_4_CSC691.zip for the TeX and bib files.
