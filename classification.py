# Jupyter notebook for classification models (Logistic Regression and Random Forest) - Phase 1
# CSC691 Final Project
# PaceMakers: Predicting Average Heart Rate for Bike Rides
# Patrick, Esteban, Sarah
# 12/2/19


# Reference for building Logistic Regression model: https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
import pandas as pd
from dateutil.parser import parse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import statsmodels
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
import csv
import warnings
import math
from sklearn.metrics import confusion_matrix
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000


def prep_data(df, type=None):
    df = df[['Avg HR (bpm)','Date','Type','Distance (km)','Avg Pace (/km)','Calories','HRSS','Elevation Gain (m)']]
    df = df.rename(columns={"Avg HR (bpm)": "AvgHR"})
    df = df[df.AvgHR != '-']
    types = ['Ride', 'VirtualRide']
    df = df[df.Type.isin(types)]
    df = df.reset_index(drop=True)

    # Convert features to numbers
    df["AvgHR"] = pd.to_numeric(df["AvgHR"])
    df["Calories"] = pd.to_numeric(df["Calories"])
    df["HRSS"] = pd.to_numeric(df["HRSS"])

    # Convert Avg Pace to seconds, parse Date and Time as separate columns
    for i in range(df.shape[0]):
        (m, s) = str(df.loc[i,'Avg Pace (/km)']).split(':')
        df.loc[i,'Avg Pace (/km)']= (int(m) * 60) + int(s)
        dt = parse(df.loc[i,'Date'])
        df.loc[i,'Date'] = dt.date()
        df.loc[i,'Time'] = dt.time()
    # Convert Avg Pace to number
    df['Avg Pace (/km)'] = pd.to_numeric(df['Avg Pace (/km)'])

    # Create binary labels for High 'AvgHR' (1) and Low 'AvgHR' (0) based on threshold of 154 bpm
    for j in range(df.shape[0]):
        if type != 'virtual':
            if int(df.loc[j,'AvgHR']) > 154:
                #print(df.loc[j,'AvgHR'])
                df.loc[j,'AvgHR_bin'] = 1
            else:
                df.loc[j,'AvgHR_bin'] = 0
        else:
            # print('VIRTUAL')
            if int(df.loc[j,'AvgHR']) > 163:
                df.loc[j,'AvgHR_bin'] = 1
            else:
                df.loc[j, 'AvgHR_bin'] = 0

    return df

def train_test(df, window_size):
    # Calculate sample size of each class
    count_no_sub = len(df[df['AvgHR_bin']==1])
    count_sub = len(df[df['AvgHR_bin']==0])
    pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
    print("percentage of High AvgHR is", '%.2f' %(pct_of_no_sub*100))
    pct_of_sub = count_sub/(count_no_sub+count_sub)
    print("percentage of Low AvgHR is", '%.2f' %(pct_of_sub*100))

    # Define train and test set, perform RFE
    # window_size = 294
    # removed Calories as feature since p-value was 0.07 > 0.05 (from below Logit function), as recommended
    df_vars = ['Distance (km)', 'Avg Pace (/km)', 'HRSS', 'Elevation Gain (m)','AvgHR_bin']
    df_final = df[df_vars]
    df_final_vars=df_final.columns.values.tolist()
    y=df_final.AvgHR_bin
    X=[i for i in df_final_vars if i not in y]

    X = df_final.loc[:, df_final.columns != 'AvgHR_bin']
    y = df_final.loc[:, df_final.columns == 'AvgHR_bin']

    # Configure train and test sets
    X_train = X.iloc[window_size:]
    y_train = y.iloc[window_size:]

    X_test = X.iloc[:window_size]
    y_test = y.iloc[:window_size]
    # print("X test: ", X_test, "y test: ", y_test)

    # Perform RFE (recursive feature elimination) to determine ranking of features
    logreg = LogisticRegression()
    rfe = RFE(logreg, 20)
    rfe = rfe.fit(X_train, y_train.values.ravel())
    print("RFE support: ", rfe.support_)
    print("RFE ranking: ", rfe.ranking_)

    try:
    # Implement Logit model to determine p-values and coefficients for each feature
        logit_model = sm.Logit(y, X)
        result = logit_model.fit()
        print("Logit results: ", result.summary2())
    except:
        print('Insufficient data for Logit model implementation.')

    return X, y

def model(window_size, max_window, X, y, clf):
    print('# Predictions: ', window_size)
    print('Window Size: ', max_window - window_size)

    X_train = X.iloc[window_size:]
    y_train = y.iloc[window_size:]
    X_test = X.iloc[:window_size]
    y_test = y.iloc[:window_size]

    # train set
    print('train set size: ', y_train.shape[0])
    # test set
    print('test set size: ', y_test.shape[0])

    actuals = pd.DataFrame(y_test)
    actuals = actuals.rename(columns={'AvgHR_bin': 'Actuals'})
    preds = np.zeros(X_test.shape[0])
    # print('preds (zeros): ', preds)

    #logreg = LogisticRegression()

    for i in range(0, y_test.shape[0]):
        #print('Iteration: ', i)
        #print('X train shape: ', X_train.shape[0])
        #print('X test shape: ', y_test.shape[0])
        clf.fit(X_train, y_train.values.ravel())
        # Predict test set
        y_pred = clf.predict(np.array(X_test.iloc[-1]).reshape(1, -1))
        #print('actual: ', y_test.loc[0, 'AvgHR_bin'], '\n pred: ', y_pred, '\n')
        preds[i] = y_pred
        # print('Accuracy of logistic regression classifier on test set {}: {:.2f}'.format(i, logreg.score(X_test, y_test)))

        # X_train = pd.concat([X_test.iloc[0], X_train]).reset_index(drop = True)
        # print('new X train: ', X_train.head())

        X_test_inst = pd.DataFrame(data=[X_test.iloc[-1]],
                                   columns=["Distance (km)", "Avg Pace (/km)", "HRSS", "Elevation Gain (m)"])
        y_test_inst = pd.DataFrame(data=[y_test.iloc[-1]], columns=["AvgHR_bin"])
        # print("X test inst: ", X_test_inst)
        X_train = X_train.drop(X_train.index[-1]).reset_index(drop=True)
        X_train = pd.concat([X_test_inst, X_train]).reset_index(drop=True)
        # print("X train: \n", X_train)

        y_train = y_train.drop(y_train.index[-1])
        y_train = pd.concat([y_test_inst, y_train])
        y_train = y_train.reset_index(drop=True)
        # print("y train \n", y_train)

        X_test = X_test.drop(X_test.index[-1])
        X_test = X_test.reset_index(drop=True)
        # print("X test: \n", X_test)

        y_test = y_test.drop(y_test.index[-1])
        y_test = y_test.reset_index(drop=True)

    preds_act_df = pd.DataFrame(preds, columns=['Predictions'])
    preds_act_df = preds_act_df.join(actuals.iloc[::-1].reset_index(drop=True))
    # print('actuals and preds: \n', preds_act_df)
    accuracy = metrics.accuracy_score(preds_act_df.Actuals.ravel(), preds_act_df.Predictions.ravel())
    print('accuracy for window size {}: {}'.format(max_window - window_size, '%.3f' % (accuracy)))

    return accuracy, actuals, preds

def find_optimal_window(clf, max_window):
    name_of_clf = clf_name
    #accuracies = np.zeros(max_window-1)
    #window_sizes = np.zeros(max_window-1)
    with open('Accuracies_for_Window_Size_Variations_' + name_of_clf + '.csv', 'w') as f:
        for window_size in range(1,max_window):
            try:
                accuracy, actuals, preds = model(window_size, max_window, X, y, clf)
                #accuracies[window_size-1] = accuracy
                #window_sizes[window_size-1] = max_window - window_size

                # Output accuracies for each window size to Accuracies_for_Window_Size_Variations.csv file
                f.write(str(max_window - window_size) + ': ' + str(accuracy))
                f.write('\n')
            except ValueError:
                print('Window size {} does not have a sufficient amount of items in each class to predict. Moving to window size {}.'.format(window_size, window_size+1))

    print("The accuracies for each window size have been stored in 'Accuracies_for_Window_Size_Variations_{}.csv' in your local working directory with accuracies recorded in descending order.".format(clf_name))

def read_windows():
    # Read in accuracies for each window size from Accuracies_for_Window_Size_Variations.csv file and sort by accuracy to determine best window size
    mylist = pd.DataFrame(columns=['Window', 'Accuracy'])
    with open('Accuracies_for_Window_Size_Variations_' + clf_name + '.csv', 'r') as csvfile:
        for i, row in enumerate(csv.reader(csvfile, delimiter='\n')):
            mylist.loc[i, 'Window'] = row[0].split(':')[0]
            mylist.loc[i, 'Accuracy'] = row[0].split(':')[1]
    print(mylist.sort_values('Accuracy', 0, ascending=False))

def read_look_aheads():
    mylist = pd.DataFrame(columns=[])
    with open('Accuracies_for_Look_Ahead_Variations_' + clf_name + '.csv', 'r') as csvfile:
        for i, row in enumerate(csv.reader(csvfile, delimiter='\n')):
            mylist.loc[i, 'Look Ahead'] = row[0].split(':')[0]
            mylist.loc[i, 'Accuracy'] = row[0].split(':')[1]
    print(mylist.sort_values('Accuracy', 0, ascending=False))

def results(actuals, preds):
    confusion_matrix1 = confusion_matrix(actuals.iloc[::-1].reset_index(drop=True), preds)
    print(confusion_matrix1)

    tn, fp, fn, tp = confusion_matrix(actuals.iloc[::-1].reset_index(drop = True), preds).ravel()
    print('true neg: ', tn, '\nfalse pos: ', fp, '\nfalse neg: ', fn, '\ntrue pos: ',tp)

    # Precision, recall, f1-score, support (# of test instances per class)
    print(classification_report(actuals.iloc[::-1].reset_index(drop=True), preds))

def find_optimal_lookahead(window_size, max_window, X, y, clf):
    average_accuracies = np.zeros(window_size)
    #predictions = []
    for m in range(1, window_size + 1):
        avg_accuracy, predictions, actuals = model_lookahead(window_size, max_window, X, y, m, clf)
        average_accuracies[m-1] = avg_accuracy

    avg_acc_df = pd.DataFrame(data=average_accuracies, columns=['Avg Accuracy'])
    print('Averages: ', avg_acc_df)
    with open('Accuracies_for_Look_Ahead_Variations_' + clf_name + '.csv', 'w') as f:
        f.write("Look Ahead : Accuracy \n")
        for i in range(0, len(average_accuracies)):
            f.write(str(i + 1) + ': ' + str(average_accuracies[i]))
            f.write('\n')
    print("The accuracies for each look ahead value have been stored in 'Accuracies_for_Look_Ahead_Variations_{}.csv' in your local working directory.".format(clf_name))

def model_lookahead(window_size, max_window, X, y, look_ahead, clf):
    print('# Predictions: ', window_size)
    print('Window Size: ', max_window - window_size)
    print('Look Ahead Value: ', look_ahead)
    print('--------------------------------------')

    # train set
    X_train = X.iloc[window_size:]
    y_train = y.iloc[window_size:]
    print('train size: ', y_train.shape[0])
    # test set
    X_test = X.iloc[:window_size]
    y_test = y.iloc[:window_size]
    print('test size: ', y_test.shape[0])

    # actuals = pd.DataFrame(y_test)
    accuracies = np.zeros(math.ceil(y_test.shape[0] / look_ahead))
    computations = np.zeros(math.ceil(y_test.shape[0] / look_ahead))
    predictions = []
    roc_values = np.zeros(math.ceil(y_test.shape[0] / look_ahead))
    rf_probs = []
    actuals = []

    for i in range(0, math.ceil(y_test.shape[0] / look_ahead)):
        preds = np.zeros(min(look_ahead, y_test.shape[0]))
        rf_prob = np.zeros(min(look_ahead, y_test.shape[0]))
        actuals_look_ahead = np.zeros(min(look_ahead, y_test.shape[0]))
        computations[i] = min(look_ahead, y_test.shape[0])
        print('Iteration: ', i)
        print('----------------')
        print('X train shape: ', X_train.shape[0])
        print('X test shape: ', y_test.shape[0])
        clf.fit(X_train, y_train.values.ravel())

        # Predict test set
        for j in range(1, min(look_ahead + 1, y_test.shape[0] + 1)):
            y_pred = clf.predict(np.array(X_test.iloc[-j]).reshape(1, -1))
            # print('actual: ', y_test.iloc[-j], '\n pred: ', y_pred, '\n')
            preds[j - 1] = y_pred
            predictions.append(y_pred)
            actuals_look_ahead[j - 1] = y_test.iloc[-j]
            rf_prob[j - 1] = clf.predict_proba(np.array(X_test.iloc[-j]).reshape(1, -1))[:, 1]
            rf_probs.append(rf_prob[j-1])

        X_test_inst = pd.DataFrame(columns=["Distance (km)", "Avg Pace (/km)", "HRSS", "Elevation Gain (m)"])
        y_test_inst = pd.DataFrame(columns=["AvgHR_bin"])
        for k in range(1, min(look_ahead + 1, y_test.shape[0] + 1)):
            X_test_inst = X_test_inst.append(X_test.iloc[-k])
            y_test_inst = y_test_inst.append(y_test.iloc[-k])
        #print("X test inst: ", X_test_inst)

        X_train = X_train.drop(X_train.index[-1]).reset_index(drop=True)
        X_train = pd.concat([X_test_inst, X_train]).reset_index(drop=True)
        # print("X train: \n", X_train)

        y_train = y_train.drop(y_train.index[-1])
        y_train = pd.concat([y_test_inst, y_train]).reset_index(drop=True)
        # print("y train \n", y_train)

        X_test = X_test.drop(X_test_inst.index.values)
        X_test = X_test.reset_index(drop=True)
        # print("X test: \n", X_test)

        y_test = y_test.drop(y_test_inst.index.values)
        y_test = y_test.reset_index(drop=True)

        print("predictions: ", preds)
        print("actuals: ", actuals_look_ahead)

        accuracy = metrics.accuracy_score(preds, actuals_look_ahead)
        accuracies[i] = accuracy
        # print("rf probs: ", rf_probs)
        # try:
        #     roc_value = roc_auc_score(actuals_look_ahead, rf_probs)
        #     print('roc for window size {} with look ahead value {}: {}'.format(max_window - window_size, look_ahead,
        #                                                                        '%.3f' % (roc_value)))
        #     roc_values[i] = roc_value
        # except ValueError:
        #     print("Insufficient number of predictions to compute roc value.")

        print('\naccuracy for iteration={} with look ahead value={}: {}'.format(i, look_ahead, '%.3f' % (accuracy)))
        for x in range(len(actuals_look_ahead)):
            actuals.append(actuals_look_ahead[x])

    roc_value = roc_auc_score(actuals, rf_probs)
    print('roc for window size={} with look ahead value={}: {}'.format(max_window - window_size, look_ahead,
                                                                       '%.3f' % (roc_value)))

    # print('accuracies: ', accuracies)
    avg_accuracy = 0
    # avg_roc = 0
    for g in range(len(accuracies)):
        avg_accuracy += (computations[g] / window_size) * accuracies[g]
        # if len(roc_values) > 0:
        #     avg_roc += (computations[g] / window_size) * roc_values[g]
        #     print('overall weighted roc for window size={}, look ahead={}: {}\n'.format(window_size, look_ahead,
        #                                                                                 '%.3f' % (avg_roc)))
    print('overall weighted accuracy for window size={}, look ahead={}: {}'.format(max_window - window_size, look_ahead, '%.3f' % (avg_accuracy)))

    return avg_accuracy, predictions, actuals

def save_results(window_size, predictions, actuals):
    # Output predictions and actuals for final window size to csv file
    with open(clf_name + '.csv', 'w') as f:
        f.write("Prediction, Actual \n")
        for i in range(0,window_size):
            f.write(str(predictions[i]) + ', ' + str(actuals[i]))
            f.write('\n')



# Please update the path to the activities.csv file from the repository
df = pd.read_csv('activities.csv')
# Prepare full dataset
df_bin = prep_data(df)

# Separate Ride and VirtualRide types
types_ride = ['Ride']
types_virtual = ['VirtualRide']

df_virtual = df[df.Type.isin(types_virtual)]
df_ride = df[df.Type.isin(types_ride)]

df_bin_ride = prep_data(df_ride)
df_bin_virtual = prep_data(df_virtual, type='virtual')


type_selection = input("Please specify if you prefer to analyze the 'full' dataset, 'virtual' ride dataset, or 'outdoor' ride dataset. \n")
if type_selection == 'full':
    df_selection = df_bin
    max_window = df_bin.shape[0]
    print(max_window)
elif type_selection == 'outdoor':
    df_selection = df_bin_ride
    max_window = df_bin_ride.shape[0]
else:
    df_selection = df_bin_virtual
    max_window = df_bin_virtual.shape[0]


if type_selection == 'outdoor':
    optimal_window = 120
    look_ahead = 120
elif type_selection == 'virtual':
    optimal_window = 4
    look_ahead = 2
else:
    optimal_window = 128
    look_ahead = 118


# Define X and y for test and train set
X, y = train_test(df_selection, optimal_window)

clf_name = 'RF' #input('Please specify the classification model you wish to test (LR or RF): \n'))

if clf_name == 'LR':
    print('Analysis for Logistic Regression\n-------------------------------------')
    if type_selection == 'outdoor':
        optimal_window = 120
        look_ahead = 120
    elif type_selection == 'virtual':
        optimal_window = 4
        look_ahead = 2
    else:
        optimal_window = 128
        look_ahead = 118

    clf = LogisticRegression()

    # Find optimal window size for LR
    all_iterations = input("Would you like to iterate over all possible window sizes to find the optimal window size (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if all_iterations =='Y':
        find_optimal_window(clf, max_window)
        read_windows()


    # Run LR for optimal window size
    LRinput = input('Please indicate Y/N to proceed with running LR with the optimal window size specified: {} \n'.format(max_window - optimal_window))
    if LRinput == 'Y':
        print('Running LR with optimal window size {}...'.format(optimal_window))
        accuracy, actuals, preds = model(optimal_window, max_window, X, y, clf)
        results(actuals, preds)

    # Find optimal look ahead value for LR
    all_iterations_lookahead = input("Would you like to iterate over all possible look ahead values to find the optimal look ahead value (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if all_iterations_lookahead == 'Y':
        find_optimal_lookahead(optimal_window,max_window, X,y,clf)
        read_look_aheads()

    # Run LR for optimal window size and optimal look ahead value
    LR_lookinput = input('Please indicate Y/N to proceed with running LR with the optimal window size and look ahead value: {}, {} \n'.format(max_window - optimal_window, look_ahead))
    if LR_lookinput == 'Y':
        print('Running LR with optimal window size and optimal look ahead value...')
        avg_accuracy, predictions, actuals = model_lookahead(optimal_window, max_window, X, y, look_ahead, clf)

        # Save predictions and actuals for LR to LR.csv file
        save_results(optimal_window, predictions, actuals)

if clf_name == 'RF':
    print('Analysis for Random Forest\n -------------------------------------')
    # optimal_window = 18
    clf = RandomForestClassifier(n_estimators=100,
                               bootstrap = True,
                               max_features = 'sqrt')
    # look_ahead = 18
    if type_selection == 'outdoor':
        optimal_window = 120
        look_ahead = 120
    elif type_selection == 'virtual':
        optimal_window = 7
        look_ahead = 4
    else:
        optimal_window = 18
        look_ahead = 118


    # Find optimal window size for LR
    all_iterations = input("Would you like to iterate over all possible window sizes to find the optimal window size (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if all_iterations == 'Y':
        find_optimal_window(clf, max_window)
        read_windows()

    # Run RF for optimal window size
    RFinput = input('Please indicate Y/N to proceed with running RF with the optimal window size specified: {} \n'.format(max_window - optimal_window))
    if RFinput == 'Y':
        accuracy, actuals, preds = model(optimal_window, max_window, X, y, clf)
        results(actuals, preds)

    # Find optimal look ahead value for RF
    all_iterations_RFlookahead = input(
        "Would you like to iterate over all possible look ahead values to find the optimal look ahead value (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if all_iterations_RFlookahead == 'Y':
        find_optimal_lookahead(optimal_window,max_window,X,y,clf)
        read_look_aheads()

    # Run RF for optimal window size of 281 and optimal look_ahead of 18
    RF_lookinput = input(
        'Please indicate Y/N to proceed with running RF with the optimal window size and look ahead value: {}, {} \n'.format(
            max_window - optimal_window, look_ahead))
    if RF_lookinput == 'Y':
        avg_accuracy, predictions, actuals = model_lookahead(optimal_window, max_window, X, y, look_ahead, clf)
        # Save predictions and actuals for LR to RF.csv file
        save_results(optimal_window, predictions, actuals)
