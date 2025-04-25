import os
import time

import numpy as np
from rank import ListNet
import pandas as pd
import json

start = time.time()

with open('../../settings.json', 'r') as f:
    settings = json.load(f)

shots = settings["GENERAL_SETTINGS"]["shots"]
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"].lower()
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()

gender_dict = {}
gender_dict = {'female': 1 if protected_group.lower() == 'female' else 0,
               'male': 0 if protected_group.lower() == 'female' else 1}


def get_files(directory):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            temp.append(os.path.join(dirpath, file))

    return temp


# Load train data
# data = pd.read_csv("data/LAW_train_data_for_LLM.csv")

def rank_with_ltr(Model, test_url, save_url):
    """
    Rank the test data using the trained model
    :param Model: LTR model
    :param test_url: path to ranking data
    :param save_url: path to save the ranked data
    :return:
    """
    test_data = pd.read_csv(test_url)
    test_data['Gender'] = test_data['Gender'].replace(gender_dict)
    rank_data = test_data.drop(['doc_id'], axis=1)
    # drop the last column
    X_test = rank_data.iloc[:, :-1].values

    # convert values to float
    X_test = X_test.astype(np.float32)
    # predict with loaded model
    m = Model.predict(X_test)
    m = pd.DataFrame(m)
    test_data['predictions'] = m
    test_data = test_data.sort_values(by=['predictions'], ascending=False)

    # save to csv
    test_data.to_csv(save_url, index=False)
    return


def model_train(train_url, epoch):
    """
    looks like we don''t need this
    :param train_url:
    :param epoch:
    :return:
    """
    Model = ListNet.ListNet()
    data = pd.read_csv(train_url)
    data['Gender'] = data['Gender'].replace(gender_dict)
    #data = data.drop(['doc_id'], axis=1)
    X = data.iloc[:, :-1].values

    # Min-Max Scaling
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X = (X - X_min) / (X_max - X_min)
    y = data.iloc[:, -1].values
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if "sqrt" in data.columns:
        tv_ratio = 0.5
    else:
        tv_ratio = 0.95
    Model.fit(X, y, n_epoch=epoch, tv_ratio=tv_ratio, optimizerAlgorithm="AdaGrad")
    return Model


def train_and_rank_with_ltr(test_url, save_url, train_url, epoch=1000):
    """
    Train and rank the test data
    :param test_url: path to ranking data
    :param save_url: path to save the ranked data
    :param train_url: path to training data
    :param epoch: number of epochs in training
    :return:
    """
    #####################################
    # Train
    #####################################

    Model = ListNet.ListNet()
    data = pd.read_csv(train_url)
    data['Gender'] = data['Gender'].replace(gender_dict)
    X = data.iloc[:, :-1].values

    # Min-Max Scaling
    X_min = np.min(X, axis=0)
    X_max = np.max(X, axis=0)
    X = (X - X_min) / (X_max - X_min)
    y = data.iloc[:, -1].values
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    if "sqrt" in data.columns:
        tv_ratio = 0.5
    else:
        tv_ratio = 0.95
    Model.fit(X, y, n_epoch=epoch, tv_ratio=tv_ratio, optimizerAlgorithm="AdaGrad")

    # create a folder to save the ranked files
    # save_folder = '../../Datasets/' + experiment_name + '/Ranked/DCS/prompt_NA/rank_size_20/shot_NA/'
    save_folder = save_url
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    ##############################################
    # Rank
    ##############################################

    # select all files in the test folder
    all_files = get_files(test_url)
    for file in all_files:
        test_path = file
        rank_with_ltr(Model.model, test_path, save_folder + f'ranked_data_' + os.path.basename(file))


#####################################
# 1. Pre-rank : For LLM rank
#####################################

# train_link = f"../../Datasets/{experiment_name}/{experiment_name}_train_data.csv"
# test_link = f"../../Datasets/{experiment_name}/Tests"
# save_link = f"../../Datasets/{experiment_name}/Ranked/ListNet/"
# epochs = 500
#
# train_and_rank_with_ltr(test_link, save_link, train_link, epoch=epochs)
# end = time.time()
# print("Time taken: ", end - start)

#############################################

# 2. Pre-rank : For Different protected group sizes
train_link = f"../../Datasets/{experiment_name}/{experiment_name}_train_data.csv"
test_link = f"../../Datasets/{experiment_name}/Tests/Different_Dis_Percent"
save_link = f"../../Datasets/{experiment_name}/Different_Dis_Percent/Ranked/ListNet/"
epochs = 500

train_and_rank_with_ltr(test_link, save_link, train_link, epoch=epochs)
end = time.time()
print("Time taken: ", end - start)