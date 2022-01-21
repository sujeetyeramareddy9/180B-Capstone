import pandas as pd
import numpy as np
from numpy.lib.function_base import cov
from numpy.linalg.linalg import norm
from numpy.random.mtrand import beta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import random

def get_model_for_spread(training_schedule, test_schedule, teams, week):
    X_train = []
    y_train = []
    features_to_include = teams.columns
    training = training_schedule.copy()[['Winner/tie', 'Loser/tie', 'HomeWin', 'PtsW', 'PtsL']]

    for row in training.iterrows():
        w = row[1][0]
        l = row[1][1]
        pt_w = row[1][3]
        pt_l = row[1][4]
        home_win = row[1][2]
        if home_win == 1:
            w_vec = list(teams.loc[w, features_to_include]) + [1]
            l_vec = list(teams.loc[l, features_to_include]) + [0]
        if home_win == 0:
            w_vec = list(teams.loc[w, features_to_include]) + [0]
            l_vec = list(teams.loc[l, features_to_include]) + [1]

        X_train.append([1] + w_vec + l_vec)
        X_train.append([1] + l_vec + w_vec)
        y_train.append(pt_w-pt_l)
        y_train.append(pt_l-pt_w)

    inferential_model = LinearRegression()
    standardize = StandardScaler()
    # X_train = standardize.fit_transform(X_train)
    inferential_model.fit(X_train, y_train)

    X_test = []
    y_test = []
    features_to_include = teams.columns
    testing = test_schedule.copy()[test_schedule['Week']<week][['Winner/tie', 'Loser/tie', 'HomeWin', 'PtsW', 'PtsL']]
    
    for row in testing.iterrows():
        w = row[1][0]
        l = row[1][1]
        pt_w = row[1][3]
        pt_l = row[1][4]
        home_win = row[1][2]
        if home_win == 1:
            w_vec = list(teams.loc[w, features_to_include]) + [1]
            l_vec = list(teams.loc[l, features_to_include]) + [0]
        if home_win == 0:
            w_vec = list(teams.loc[w, features_to_include]) + [0]
            l_vec = list(teams.loc[l, features_to_include]) + [1]
        X_test.append([1] + w_vec + l_vec)
        X_test.append([1] + l_vec + w_vec)
        y_test.append(pt_w-pt_l)
        y_test.append(pt_l-pt_w)

    # X_test = standardize.fit_transform(X_test)
    y_pred = inferential_model.predict(X_test)
    print("Test MAE: " + str(np.mean(np.abs(np.array(y_test) - np.array(y_pred)))))

    relationship_model = LinearRegression()
    y_pred = [[1, i] for i in y_pred]
    relationship_model.fit(y_pred, y_test)

    return inferential_model, relationship_model
    


def predict_spread(test_schedule, teams, week, model):
    X_val = []
    features_to_include = teams.columns
    validation = test_schedule[test_schedule['Week']==week][['Winner/tie', 'Loser/tie']]

    for row in validation.iterrows():
        a = row[1][0]
        h = row[1][1]
        a_vec = list(teams.loc[a, features_to_include]) + [0]
        h_vec = list(teams.loc[h, features_to_include]) + [1]
        X_val.append([1] + a_vec + h_vec)
    
    standardize = StandardScaler()
    # X_val = standardize.fit_transform(X_val)

    y_val = model.predict(X_val)
    validation['prediction'] = y_val
    return X_val, y_val, validation