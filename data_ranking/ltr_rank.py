import json
import os
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

with open('../settings.json', 'r') as f:
    settings = json.load(f)

# sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
dadv_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"]
num_iterations = settings["TRAINING_SETTINGS"]["HYPERPARAMETERS"]["NUM_ITERATIONS"]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
score_col = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
experiment_name = 'LAW'
model_name = 'ListNet'
if dadv_group == 'female':
    adv_group = 'male'
else:
    adv_group = 'female'
demo_dict = {adv_group: 0, dadv_group: 1}


def get_files(directory):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            match = re.search(experiment_name, file)
            if match:
                # temp.append(directory + '/' + file)
                temp.append(os.path.join(dirpath, file))
    return temp


# size in models represent training size for the model
all_models = get_files(f'../Models/')
models = [model for model in all_models if str(num_iterations) in model]


def rank_ltr(size=50):
    """
    the size here is the size of the test set to be ranked
    :param size:
    :param shot:
    :return:
    """
    write_path = f'../Datasets/{experiment_name}/Ranked/{model_name}'

    gt = f'../Datasets/{experiment_name}/Tests/rank_size_{size}.csv'
    rank(models, size, write_path, gt)


def rank(models, size, write_path=f'../Datasets/{experiment_name}/Ranked/{model_name}',
         gt_path=f'../Datasets/{experiment_name}/Tests/rank_size_50.csv'):
    write_path = Path(write_path) / f'rank_size_{size}/shot_NA'
    if not os.path.exists(write_path):
        os.makedirs(write_path)
    for model in models:
        train_size = re.search(r'size_(\d+)', model).group(1)
        print('gt_path:', gt_path)
        print('model:', model)
        test_data = pd.read_csv(gt_path, index_col=False)
        filehandler = open(model, 'rb')
        DELTR = pickle.load(filehandler)

        test_data[protected_feature] = test_data[protected_feature].replace(demo_dict)

        # insert qid column in the first column and all rows show contain number 1
        test_data.insert(0, 'q_id', 1)

        numeric_cols = test_data.select_dtypes(include=[np.number]).columns.to_list()
        formatted_data = test_data[numeric_cols]
        print("data will be ranking using", numeric_cols)

        print('formatted data', formatted_data)

        result = DELTR.rank(formatted_data, has_judgment=True)

        result["GT_score"] = result['doc_id'].map(test_data.set_index('doc_id')[score_col])

        print(result)

        result.to_csv(str(write_path) + f'/{train_size}_ranked.csv', index=False)


rank_ltr(50)
