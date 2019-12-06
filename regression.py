# Jupyter notebook for regression models (Gradient Boosting, Random Forest Regression)
# CSC691 Final Project
# PaceMakers: Predicting Average Heart Rate for Bike Rides
# Patrick, Esteban, Sarah
# 12/2/19


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
from sklearn import ensemble
from sklearn.metrics import mean_squared_error, accuracy_score
warnings.simplefilter(action='ignore', category=FutureWarning)
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000


def prep_data(df):
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
        if int(df.loc[j,'AvgHR']) > 154:
            #print(df.loc[j,'AvgHR'])
            df.loc[j,'AvgHR_bin'] = 1
        else:
            df.loc[j,'AvgHR_bin'] = 0

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
    df_vars = ['Distance (km)', 'Avg Pace (/km)', 'HRSS', 'Elevation Gain (m)','AvgHR']
    df_bin = ['AvgHR_bin']

    df_final = df[df_vars]
    df_final_bin = df[df_bin]

    df_final_vars=df_final.columns.values.tolist()
    df_final_bin_vars=df_final_bin.columns.values.tolist()

    y=df_final.AvgHR
    X=[i for i in df_final_vars if i not in y]
    X_bin = [i for i in df_final_bin_vars if i not in y]

    X = df_final.loc[:, df_final.columns != 'AvgHR']
    y = df_final.loc[:, df_final.columns == 'AvgHR']
    X_bin = df_final_bin.loc[:, df_final_bin.columns == 'AvgHR_bin']

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

    # Implement Logit model to determine p-values and coefficients for each feature
    # logit_model = sm.Logit(y, X)
    # result = logit_model.fit()
    # print("Logit results: ", result.summary2())

    return X, X_bin, y

def model(window_size, X, y, clf):
    print('# Predictions: ', window_size)
    print('Window Size: ', 309 - window_size)

    X_train = X.iloc[window_size:]
    y_train = y.iloc[window_size:]
    X_test = X.iloc[:window_size]
    y_test = y.iloc[:window_size]
    X_bin_test = X_bin.iloc[:window_size]


    # train set
    print('train set size: ', y_train.shape[0])
    # print(y_train)
    # test set
    print('test set size: ', y_test.shape[0])
    # print(y_test)

    actuals = pd.DataFrame(y_test)
    actuals = actuals.rename(columns={'AvgHR': 'Actuals'})
    preds = np.zeros(X_test.shape[0])
    preds_bin = np.zeros(X_test.shape[0])
    actuals_bin = np.zeros(X_test.shape[0])
    # print('preds (zeros): ', preds)

    for i in range(0, y_test.shape[0]):
        #print('Iteration: ', i)
        #print('X train shape: ', X_train.shape[0])
        #print('X test shape: ', y_test.shape[0])
        clf.fit(X_train, y_train.values.ravel())
        # Predict test set
        y_pred = clf.predict(np.array(X_test.iloc[-1]).reshape(1, -1))
        #print('actual: ', y_test.loc[0, 'AvgHR_bin'], '\n pred: ', y_pred, '\n')
        preds[i] = y_pred
        if y_pred > 154:
            preds_bin[i] = 1
        else:
            preds_bin[i] = 0
        actuals_bin[i] = df_bin.loc[X_bin_test.iloc[-1],'AvgHR_bin']
        mse = mean_squared_error(y_test.iloc[-1], y_pred)

        # print('Accuracy of logistic regression classifier on test set {}: {:.2f}'.format(i, logreg.score(X_test, y_test)))

        # X_train = pd.concat([X_test.iloc[0], X_train]).reset_index(drop = True)
        # print('new X train: ', X_train.head())

        X_test_inst = pd.DataFrame(data=[X_test.iloc[-1]],
                                   columns=["Distance (km)", "Avg Pace (/km)", "HRSS", "Elevation Gain (m)"])
        y_test_inst = pd.DataFrame(data=[y_test.iloc[-1]], columns=["AvgHR"])
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
    # print('actuals bin: ', actuals_bin)
    # print('preds bin: ', preds_bin)
    # print('predictions: ', preds)
    accuracy = metrics.accuracy_score(actuals_bin, preds_bin)
    print('accuracy for window size {}: {}'.format(309 - window_size, '%.3f' % (accuracy)))
    final_mse = metrics.mean_squared_error(preds_act_df.Actuals.ravel(), preds_act_df.Predictions.ravel())
    print('mse for window size {}: {}'.format(window_size, '%.3f' % (final_mse)))

    return accuracy, actuals_bin, preds_bin, final_mse

def find_optimal_window(clf, max_window):
    accuracies = np.zeros(max_window-1)
    mses = np.zeros(max_window-1)

    for window_size in range(1,max_window):
        accuracy, actuals, preds, mse = model(window_size, X, y, clf)
        accuracies[window_size-1] = accuracy
        mses[window_size-1] = mse

    print('Accuracies: ', accuracies)
    print('MSEs: ', mses)

    # Output accuracies for each window size to Accuracies_for_Window_Size_Variations.csv file
    with open('MSEs_for_Window_Size_Variations_' + clf_name + '.csv', 'w') as f:
        for i in range(0, len(mses)):
            f.write(str(308 - i) + ': ' + str(mses[i]))
            f.write('\n')

def read_windows():
    # Read in accuracies for each window size from Accuracies_for_Window_Size_Variations.csv file and sort by accuracy to determine best window size
    mylist = pd.DataFrame(columns=['Window', 'MSE'])
    with open('MSEs_for_Window_Size_Variations_' + clf_name + '.csv', 'r') as csvfile:
        for i, row in enumerate(csv.reader(csvfile, delimiter='\n')):
            mylist.loc[i, 'Window'] = row[0].split(':')[0]
            mylist.loc[i, 'MSE'] = row[0].split(':')[1]
    print(mylist.sort_values('MSE', 0, ascending=False))

def read_look_aheads():
    mylist = pd.DataFrame(columns=[])
    with open('Accuracies_for_Look_Ahead_Variations_' + clf_name + '.csv', 'r') as csvfile:
        for i, row in enumerate(csv.reader(csvfile, delimiter='\n')):
            mylist.loc[i, 'Look Ahead'] = row[0].split(':')[0]
            mylist.loc[i, 'Accuracy'] = row[0].split(':')[1]
    print(mylist.sort_values('Accuracy', 0, ascending=False))
    print('\n')
    mylist2 = pd.DataFrame(columns=[])
    with open('MSEs_for_Look_Ahead_Variations_' + clf_name + '.csv', 'r') as csvfile:
        for i, row in enumerate(csv.reader(csvfile, delimiter='\n')):
            mylist2.loc[i, 'Look Ahead'] = row[0].split(':')[0]
            mylist2.loc[i, 'MSE'] = row[0].split(':')[1]
    print(mylist2.sort_values('MSE', 0, ascending=False))

def results(actuals, preds):
    confusion_matrix1 = confusion_matrix(actuals, preds)
    print(confusion_matrix1)

    tn, fp, fn, tp = confusion_matrix(actuals, preds).ravel()
    print('true neg: ', tn, '\nfalse pos: ', fp, '\nfalse neg: ', fn, '\ntrue pos: ',tp)

    # Precision, recall, f1-score, support (# of test instances per class)
    print(classification_report(actuals, preds))

def find_optimal_lookahead(window_size, X, y, clf):
    accuracies = np.zeros(window_size)
    mses = np.zeros(window_size)
    for m in range(1, window_size + 1):
        accuracy, predictions, actuals, mse = model_lookahead(window_size, X, y, m, clf)
        accuracies[m-1] = accuracy
        mses[m-1] = mse

    avg_mse_df = pd.DataFrame(data=mses, columns=['MSE'])
    print('MSEs: \n', avg_mse_df)
    with open('MSEs_for_Look_Ahead_Variations_' + clf_name + '.csv', 'w') as f:
        # f.write("Look Ahead : Accuracy \n")
        for i in range(0, len(mses)):
            f.write(str(i + 1) + ': ' + str(mses[i]))
            f.write('\n')
    acc_df = pd.DataFrame(data=accuracies, columns=['Accuracy'])
    print('Average Accuracies: \n', accuracies)
    with open('Accuracies_for_Look_Ahead_Variations_' + clf_name + '.csv', 'w') as f:
        # f.write("Look Ahead : Accuracy \n")
        for i in range(0, len(accuracies)):
            f.write(str(i + 1) + ': ' + str(accuracies[i]))
            f.write('\n')

def model_lookahead(window_size, X, y, look_ahead, clf):
    print('# Predictions: ', window_size)
    print('Window Size: ', 309 - window_size)
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
    X_bin_test = X_bin.iloc[:window_size]

    actuals = []
    predictions = []
    actuals_bin = []
    preds_bin = []

    accuracies = np.zeros(math.ceil(y_test.shape[0] / look_ahead))
    computations = np.zeros(math.ceil(y_test.shape[0] / look_ahead))
    mses = np.zeros(math.ceil(y_test.shape[0] / look_ahead))

    for i in range(0, math.ceil(y_test.shape[0] / look_ahead)):
        print('# iterations: ', math.ceil(y_test.shape[0] / look_ahead))
        preds = np.zeros(min(look_ahead, y_test.shape[0]))
        # rf_probs = np.zeros(min(look_ahead, y_test.shape[0]))
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
            actuals_look_ahead[j - 1] = y_test.iloc[-j]

            if y_pred > 154:
                preds_bin.append(1)
            else:
                preds_bin.append(0)
            predictions.append(preds[j-1])
            actuals_bin.append(df_bin.loc[X_bin_test.iloc[-1], 'AvgHR_bin'].values[0])
            # rf_probs[j - 1] = clf.predict_proba(np.array(X_test.iloc[-j]).reshape(1, -1))[:, 1]


        X_test_inst = pd.DataFrame(columns=["Distance (km)", "Avg Pace (/km)", "HRSS", "Elevation Gain (m)"])
        y_test_inst = pd.DataFrame(columns=["AvgHR"])
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
        for x in range(len(actuals_look_ahead)):
            actuals.append(actuals_look_ahead[x])
        mse = metrics.mean_squared_error(preds, actuals_look_ahead)

        # roc_value = roc_auc_score(actuals_look_ahead, rf_probs)

        print('mse for iteration {} at window size={} with look ahead value={}: {}'.format(i, 309 - window_size, look_ahead, '%.3f' % (mse)))
        mses[i] = mse

    print("\npredict bins: ", preds_bin)
    print("actual bins: ", actuals_bin)
    print('predictions: ', predictions)
    print('actuals: ', actuals)
    accuracy = metrics.accuracy_score(preds_bin, actuals_bin)
    print('overall accuracy for window size={}, look ahead value={}: {}'.format(309-window_size, look_ahead, '%.3f' % (accuracy)))
    print('mses: ', mses)

    #avg_accuracy = 0
    avg_mse = 0
    for g in range(len(accuracies)):
        # avg_accuracy += (computations[g] / window_size) * accuracies[g]
        avg_mse += (computations[g] / window_size) * mses[g]
    #print('\noverall weighted accuracy for window size={}, look ahead={}: {}'.format(window_size, look_ahead, '%.3f' % (avg_accuracy)))
    print('overall weighted mse for window size={}, look ahead={}: {}\n'.format(309-window_size, look_ahead, '%.3f' % (avg_mse)))

    return accuracy, predictions, actuals, avg_mse

def save_results(window_size, predictions, actuals):
    # Output predictions and actuals for final window size to csv file
    with open(clf_name + '.csv', 'w') as f:
        f.write("Prediction, Actual \n")
        for i in range(0,window_size):
            f.write(str(predictions[i]) + ', ' + str(actuals[i]))
            f.write('\n')




# Please update the path to the activities.csv file from the repository
df = pd.read_csv('activities.csv')
# Prepare data
df_bin = prep_data(df)

# max_window = df_bin.shape[0]

# Define X, binary X, and y for test and train set
X, X_bin, y = train_test(df_bin, 128)

clf_name = 'RFR' #str(input('Please specify the classification model you wish to test (LR or RF): \n'))

if clf_name == 'GB':
    print('Analysis for Gradient Boosting\n -------------------------------------')
    optimal_window = 7
    params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    look_ahead = 7

    # Find optimal window size for GB
    all_iterations = input(
        "Would you like to iterate over all possible window sizes to find the optimal window size (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if all_iterations == 'Y':
        find_optimal_window(clf,277)
        read_windows()

    # Run GB for optimal window size of 128
    GBinput = input(
        'Please indicate Y/N to proceed with running GB with the optimal window size specified: {} \n'.format(
            309 - optimal_window))
    if GBinput == 'Y':
        print('Running GB with the optimal window size {}...'.format(optimal_window))
        accuracy, actuals, preds, mse = model(optimal_window, X, y, clf)
        results(actuals, preds)

    # Find optimal look ahead value for GB
    all_iterations_lookahead = input(
        "Would you like to iterate over all possible look ahead values to find the optimal look ahead value (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if all_iterations_lookahead == 'Y':
        find_optimal_lookahead(optimal_window,X,y,clf)
        read_look_aheads()

    # Run LR for optimal window size of 128 and optimal look_ahead of 118
    LR_lookinput = input(
        'Please indicate Y/N to proceed with running GB with the optimal window size and look ahead value: {}, {} \n'.format(
            309 - optimal_window, look_ahead))
    if LR_lookinput == 'Y':
        avg_accuracy, predictions, actuals, mse = model_lookahead(optimal_window, X, y, look_ahead, clf)
        # Save predictions and actuals for LR to LR.csv file
        save_results(optimal_window, predictions, actuals)

if clf_name == 'RFR':
    print('Analysis for Random Forest Regression\n -------------------------------------')
    optimal_window = 7
    clf = ensemble.RandomForestRegressor(n_estimators=100)
    look_ahead = 7

    # Find optimal window size for RFR
    RFRall_iterations = input(
        "Would you like to iterate over all possible window sizes to find the optimal window size (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if RFRall_iterations == 'Y':
        find_optimal_window(clf,277)
        read_windows()

    # Run RFR for optimal window size of 302
    RFRinput = input(
        'Please indicate Y/N to proceed with running GB with the optimal window size specified: {} \n'.format(
            309 - optimal_window))
    if RFRinput == 'Y':
        accuracy, actuals, preds, mse = model(optimal_window, X, y, clf)
        results(actuals, preds)

    # Find optimal look ahead value for RFR
    RFRall_iterations_lookahead = input(
        "Would you like to iterate over all possible look ahead values to find the optimal look ahead value (Y/N)? If so, please note this may take an extensive amount of time. \n")
    if RFRall_iterations_lookahead == 'Y':
        find_optimal_lookahead(optimal_window,X,y,clf)
        read_look_aheads()

    # Run RF for optimal window size of 302 and optimal look_ahead of 7
    LR_lookinput = input(
        'Please indicate Y/N to proceed with running GB with the optimal window size and look ahead value: {}, {} \n'.format(
            309 - optimal_window, look_ahead))
    if LR_lookinput == 'Y':
        avg_accuracy, predictions, actuals, mse = model_lookahead(optimal_window, X, y, look_ahead, clf)
        # Save predictions and actuals for LR to RF.csv file
        save_results(optimal_window, predictions, actuals)
