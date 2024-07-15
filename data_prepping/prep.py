import json
import os.path
import random
import time

import pandas as pd

start_time = time.time()

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
rank_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]


def prep_LLM_data(size=50, sh=1):
    """ this function splits the data into test and train data for the LLMs and LTR models"""
    # read test and train datas separately
    test_data = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_test_data_for_LLM.csv")
    test_df = test_data.sample(n=size, random_state=1)
    # create Test folder if it does not exist
    if not os.path.exists(f"../Datasets/{experiment_name}/Tests"):
        os.makedirs(f"../Datasets/{experiment_name}/Tests")
    # create Train folder if it does not exist
    if not os.path.exists(f"../Datasets/{experiment_name}/Train"):
        os.makedirs(f"../Datasets/{experiment_name}/Train")
    test_df.to_csv(f"../Datasets/{experiment_name}/Tests/rank_size_{size}.csv", index=False)
    if sh == 0:
        return
    train_data = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_train_data_for_LLM.csv")
    random_indices = sorted(random.sample(range(len(train_data)), int(size)))
    train_df = train_data.iloc[random_indices]
    train_df.to_csv(f"../Datasets/{experiment_name}/Train/rank_size_{size}_shot_{sh}.csv", index=False)


for rank_size in rank_sizes:
    for shot in shots:
        prep_LLM_data(rank_size, shot)

# for shot in shots:
#     prep_LLM_data(5, shot)
