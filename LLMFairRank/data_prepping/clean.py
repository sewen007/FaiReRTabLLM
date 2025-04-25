import json
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

with open('./settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
dadv_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
additional_columns = settings["READ_FILE_SETTINGS"]["ADDITIONAL_COLUMNS"]


def Clean():
    if dadv_group == 'female':
        adv_group = 'male'
    else:
        adv_group = 'female'

    if experiment_name == 'LAW':
        data = pd.read_excel('./LAW.xlsx')
        gender_dict = {2: 0, 1: 1}
        data['Gender'] = data['sex'].replace(gender_dict)
    else:
        data = pd.read_csv(f'./{experiment_name}.csv')

    if experiment_name == 'COMPASSEX':
        # multiply score by -1
        data['score'] = data['score'] * -1
        gender_dict = {'M': 1, 'F': 0}
        data['Gender'] = data['Gender'].replace(gender_dict)
    if experiment_name == 'NBAWNBA':
        gender_dict = {'M': 0, 'F': 1}
        data['Gender'] = data['Gender'].replace(gender_dict)
    if experiment_name == 'LOAN':
        gender_dict = {'Male': 'male', 'Female': 'female'}
        data['Gender'] = data['Gender'].replace(gender_dict)

    if experiment_name == 'BostonMarathon':
        # Convert time to seconds
        data['score'] = pd.to_timedelta(data['score']).dt.total_seconds()
        # Normalize (min-max scaling) and invert
        min_time, max_time = data['score'].min(), data['score'].max()
        data['normalized_time'] = 1 - (data['score'] - min_time) / (max_time - min_time)
        data['score'] = data['normalized_time'].round(5)
        print(data['score'])
        gender_dict = {'M': 0, 'F': 1}
        data['Gender'] = data['Gender'].replace(gender_dict)

    # remove rows with missing score or gender values
    data = data.dropna(subset=[score_column, 'Gender'])

    # create doc_id to start from 1 instead of 0
    # if doc_id column is not present, create it
    if 'doc_id' not in data.columns:
        data['doc_id'] = (data.index + 1).astype(int)
    seed = 43

    # select relevant columns
    if protected_feature != '':
        gt_data = data[['doc_id', protected_feature]].copy()
        # ensure data['doc_id'] is integer
    else:
        gt_data = data[['doc_id']].copy()
    gt_data = gt_data.dropna(subset=['doc_id'])
    gt_data['doc_id'] = gt_data['doc_id'].astype(int)
    for column in additional_columns:
        gt_data.loc[:, column] = data[column]
    gt_data.loc[:, score_column] = data[score_column]

    # sort data based on ZFYA column
    gt_data = gt_data.sort_values(by=score_column, ascending=False)

    """ DIRECTORY MANAGEMENT """
    data_path = Path("../LLMFairRank/Datasets/" + experiment_name + '/')

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # save full data to csv
    gt_data.to_csv(str(data_path) + f"/{experiment_name}_data.csv", index=False)

    # split data into  20% test and 80% train using sklearn test_train_split
    train_data, test_data = train_test_split(gt_data, test_size=0.2, random_state=seed, shuffle=True)

    # sort test and train data based on score column
    test_data = test_data.sort_values(by=score_column, ascending=False)
    train_data = train_data.sort_values(by=score_column, ascending=False)

    # save test and train data to csv
    test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data.csv", index=False)

    if protected_feature == '':
        # save test and train data to csv
        train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data.csv", index=False)
        test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data_for_LLM.csv", index=False)
        train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data_for_LLM.csv", index=False)

        exit()
    # prepare test data for LLM
    demo_dict = {0: adv_group, 1: dadv_group}
    test_data[protected_feature] = test_data[protected_feature].replace(demo_dict)
    test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data_for_LLM.csv", index=False)

    # prepare train data for LLM
    train_data[protected_feature] = train_data[protected_feature].replace(demo_dict)
    train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data_for_LLM.csv", index=False)

    # clean train data for training (not LLM)
    train_data = train_data.drop(['doc_id'], axis=1)
    train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data.csv", index=False)

#######################################################################################
# temp prep
# for data gotten from previous hoitr experiments

# no need to set the advantaged and disadvantaged groups
# read test and train data
# test_data = pd.read_csv(f'../../LLMFairRank/Datasets/{experiment_name}/Testing_{experiment_name}.csv')
# train_data = pd.read_csv(f'../../LLMFairRank/Datasets/{experiment_name}/Training_{experiment_name}.csv')
#
#
# # no need to do the gender dict swap
# # no need to create doc_id. Rather delete q_id column
# test_data = test_data.drop(['q_id'], axis=1)
# train_data = train_data.drop(['q_id'], axis=1)
#
# # data is already sorted based on score column
#
# """ DIRECTORY MANAGEMENT """
# data_path = Path("../../LLMFairRank/Datasets/" + experiment_name + '/')
#
# # data is already split and saved in full format
#
# # drop columns that has "Name" in it
# train_data = train_data.drop(train_data.filter(like='Name').columns, axis=1)
# test_data = test_data.drop(test_data.filter(like='Name').columns, axis=1)
#
# # save test data as str(data_path) + f"/{experiment_name}_test_data.csv
# test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data.csv", index=False)
#
# if protected_feature == '':
#     # save test and train data to csv
#     train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data.csv", index=False)
#     test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data_for_LLM.csv", index=False)
#     train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data_for_LLM.csv", index=False)
#
#     exit()
# # prepare test data for LLM
# demo_dict = {0: adv_group, 1: dadv_group}
# test_data[protected_feature] = test_data[protected_feature].replace(demo_dict)
# test_data.to_csv(str(data_path) + f"/{experiment_name}_test_data_for_LLM.csv", index=False)
#
# # prepare train data for LLM
# train_data[protected_feature] = train_data[protected_feature].replace(demo_dict)
# train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data_for_LLM.csv", index=False)
#
# # clean train data for training (not LLM)
# train_data = train_data.drop(['doc_id'], axis=1)
# train_data.to_csv(str(data_path) + f"/{experiment_name}_train_data.csv", index=False)
