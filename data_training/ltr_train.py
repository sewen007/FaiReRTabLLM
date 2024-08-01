import csv
import json
import os
import pickle

import pandas as pd
from fairsearchdeltr import Deltr

with open('../settings.json', 'r') as f:
    settings = json.load(f)

experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
dadv_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"]
num_iterations = settings["TRAINING_SETTINGS"]["HYPERPARAMETERS"]["NUM_ITERATIONS"]
l_r = settings["TRAINING_SETTINGS"]["HYPERPARAMETERS"]["LR"]

if dadv_group == 'female':
    adv_group = 'male'
else:
    adv_group = 'female'
demo_dict = {adv_group: 0, dadv_group: 1}

shot = 1


def train(train_size, full_train=False):
    if full_train:
        train_data = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_train_data_for_LLM.csv")
    else:
        # check if a path exists
        if os.path.exists(f"../Datasets/{experiment_name}/Train/' + 'rank_size_{train_size}_shot_1.csv"):
            train_data = pd.read_csv(
                f"../Datasets/{experiment_name}/Train/' + 'rank_size_{train_size}_shot_1.csv")
        else:  # randomly sample from the training data based on the train size
            train_df = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_train_data_for_LLM.csv")
            train_data = train_df.sample(n=train_size, random_state=1)

    train_data[protected_feature] = train_data[protected_feature].replace(demo_dict)

    # insert qid column in the first column and all rows show contain number 1
    train_data.insert(0, 'q_id', 1)

    gamma = 0

    dtr = Deltr(protected_feature, gamma, num_iterations, learning_rate=l_r, lambdaa=0.001, init_var=0.01,
                standardize=True)
    dtr.train(train_data)

    loss_dir_path = '../LNLoss/' + experiment_name
    if not os.path.exists(loss_dir_path):
        os.makedirs(loss_dir_path)
    if full_train:
        LOSS_PATH = '../LNLoss/' + experiment_name + "/(num_iterations=" + str(num_iterations) + ",gamma=" + str(
            gamma) + ")" + 'shot_' + str(shot) + '_size_' + str(len(train_data)) + "_full_train.csv"
    else:
        LOSS_PATH = '../LNLoss/' + experiment_name + "/(num_iterations=" + str(num_iterations) + ",gamma=" + str(
            gamma) + ")" + 'shot_' + str(shot) + '_size_' + str(train_size) + ".csv"

    with open(LOSS_PATH, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            ['iteration', 'loss', 'loss_exposure', 'loss_standard', 'omega'])
        i = 0

        for train_step in dtr.log:
            csvwriter.writerow([str(i), str(train_step.loss), str(train_step.loss_exposure),
                                str(train_step.loss_standard), train_step.omega])
            i += 1
    file_folder = '../Models/' + experiment_name
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)
    FILE_PATH = r'../Models/' + experiment_name + "/(num_iterations=" + str(num_iterations) + ",gamma=" + str(
        gamma) + ")" + 'shot_' + str(shot) + '_size_' + str(train_size) + '_' + experiment_name + '.obj'
    file = open(FILE_PATH, "wb")
    pickle.dump(dtr, file)
    print("SAVED MODEL TO PATH: " + FILE_PATH)


train_data = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_train_data_for_LLM.csv")
full_train_size = len(train_data)
for size in sample_sizes:
    train(size)
train(full_train_size, full_train=True)
