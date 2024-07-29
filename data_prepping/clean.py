import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
dadv_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
additional_columns = settings["READ_FILE_SETTINGS"]["ADDITIONAL_COLUMNS"]

if dadv_group == 'female':
    adv_group = 'male'
else:
    adv_group = 'female'

if experiment_name == 'LAW':
    data = pd.read_excel('../LAW.xlsx')
else:
    data = pd.read_csv(f'../{experiment_name}.csv')
# read xlsx data
if experiment_name == 'LAW':
    data = pd.read_excel('../LAW.xlsx')
    # rename sex to Gender and replace 2 with 0
    gender_dict = {2: 0, 1: 1}
    data['Gender'] = data['sex'].replace(gender_dict)

# create doc_id to start from 1 instead of 0
data['doc_id'] = data.index + 1
seed = 43


# select relevant columns
if protected_feature != '':
    gt_data = data[['doc_id', protected_feature]].copy()
else:
    gt_data = data[['doc_id']].copy()
for column in additional_columns:
    gt_data.loc[:, column] = data[column]
gt_data.loc[:, score_column] = data[score_column]

# sort data based on ZFYA column
gt_data = gt_data.sort_values(by=score_column, ascending=False)

""" DIRECTORY MANAGEMENT """
data_path = Path("../../LLMFairRank/Datasets/" + experiment_name + '/')

if not os.path.exists(data_path):
    os.makedirs(data_path)

# save full data to csv
gt_data.to_csv(str(data_path) + f"/{experiment_name}_data.csv", index=False)

# split data into  20% test and 80% train using sklearn test_train_split
test_data, train_data = train_test_split(gt_data, test_size=0.2, random_state=seed, shuffle=True)

# sort test and train data based on ZFYA column
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
