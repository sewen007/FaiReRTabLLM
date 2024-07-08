import os
from pathlib import Path

import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split

experiment_name = "LAW"

# read xlsx data
gt_data = pd.read_excel('law_data.xlsx')
seed = 123

# adjust unique_id to start from 1 instead of 0
gt_data['unique_id'] = gt_data.index + 1

dict = {2: 0, 1: 1}
gt_data['Gender'] = gt_data['sex'].replace(dict)

gt_data = gt_data[['unique_id', 'Gender', 'UGPA', 'LSAT', 'ZFYA']]

# sort data based on ZFYA column
gt_data = gt_data.sort_values(by='ZFYA', ascending=False)

""" DIRECTORY MANAGEMENT """
data_path = Path("../LLMFairRank/Datasets/" + experiment_name + '/')

if not os.path.exists(data_path):
    os.makedirs(data_path)

# save full data to csv
gt_data.to_csv(str(data_path) + "/LAW_data_for_LLM.csv", index=False)

# split data into  20% test and 80% train using sklearn test_train_split
test_data, train_data = train_test_split(gt_data, test_size=0.2, random_state=seed)

# sort test and train data based on ZFYA column
test_data = test_data.sort_values(by='ZFYA', ascending=False)
train_data = train_data.sort_values(by='ZFYA', ascending=False)


# clean train data for training
train_data = train_data.drop(['unique_id'], axis=1)

# save test and train data to csv
test_data.to_csv(str(data_path) + "/LAW_test_data.csv", index=False)
train_data.to_csv(str(data_path) + "/LAW_train_data.csv", index=False)

# prepare test data for LLM
dict = {0: "female", 1: "male"}
test_data['Gender'] = test_data['Gender'].replace(dict)
test_data.to_csv(str(data_path) + "/LAW_test_data_for_LLM.csv", index=False)


