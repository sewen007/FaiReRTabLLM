# this script prepares the data (already split) for the LLM ranking. It contains several functions that can be used to
# prep the data for different experiments.

import json
import os.path
import random
import shutil
import time
import detconstsort as dcs
import numpy as np
import pandas as pd
from data_analysis import calculate_metrics as cm
from data_analysis import skew as sk
from data_viz import plot_skew

start_time = time.time()

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
rank_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()
non_protected_group = settings["READ_FILE_SETTINGS"]["ADV_GROUP"].lower()

# random.seed(42)
# np.random.seed(42)

gender_dict = {}
gender_dict = {'female': 1 if protected_group.lower() == 'female' else 0,
               'male': 0 if protected_group.lower() == 'female' else 1}
test_data = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_test_data_for_LLM.csv")

if protected_feature == 'Gender':
    # split data into two dataframes based on Gender column and shuffle them
    test_data_1 = test_data[test_data[protected_feature] == 'female']
    test_data_2 = test_data[test_data[protected_feature] == 'male']
    # change random state to 42 for reproducibility in LLM-LTR. and 2 for reranking with score only
    # test_data_1 = test_data_1.sample(frac=1, random_state=2).reset_index(drop=True)
    # test_data_2 = test_data_2.sample(frac=1, random_state=2).reset_index(drop=True)

# def prep_LLM_data(size=50, sh=1):
#     """ this function creates the train data for the shots for LLM"""
#     # create Train folder if it does not exist
#     if not os.path.exists(f"../Datasets/{experiment_name}/Train"):
#         os.makedirs(f"../Datasets/{experiment_name}/Train")
#     if sh == 0:
#         return
#     train_data = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_train_data_for_LLM.csv")
#
#     # shuffle data
#     random_indices = random.sample(range(len(train_data)), int(10 * size))
#     train_df = train_data.iloc[random_indices]
#
#     # if sh is 1, then we take the first size rows, if sh is 2, we take the next size rows and so on
#     train_df = train_df.iloc[((sh - 1) * size):(sh * size)]
#
#     # sort by score
#     train_df = train_df.sort_values(by=score_column, ascending=False)
#     train_df.to_csv(f"../Datasets/{experiment_name}/Train/rank_size_{size}_shot_{sh}.csv", index=False)
random_sate = 123
train_data = pd.read_csv(f"../Datasets/{experiment_name}/{experiment_name}_train_data_for_LLM.csv")
train_data_1 = train_data[train_data[protected_feature] == 'female']
# train_data_1 = train_data_1.sample(frac=1, random_state=random_sate).reset_index(drop=True)
# random_indices_1 = random.sample(range(len(train_data_1)), int(10 * size))
# train_data_1 = train_data_1.iloc[random_indices_1]

train_data_2 = train_data[train_data[protected_feature] == 'male']


# train_data_2 = train_data_2.sample(frac=1, random_state=random_sate).reset_index(drop=True)


def prep_LLM_data(size=50, sh=1, create_fair_data=False, create_fair_data_for_reranking=False):
    """ this function creates the train data for the shots for LLM. The data is ranked used DetConstSort"""
    # do not create data for shot 0
    if sh == 0:
        return

    # create Scored data folder if it does not exist
    scored_save_path = f"../Datasets/{experiment_name}/Train/Scored/shot_{sh}"
    if not os.path.exists(scored_save_path):
        os.makedirs(scored_save_path)

    # split data into 2 groups if using sex as a protected feature and randomize them
    if protected_feature == 'Gender':
        # if sh is 1, then we take the first size rows, if sh is 2, we take the next size rows and so on
        # avoiding the first one

        sho = sh + 1

        train_df = pd.concat([train_data_1.iloc[((sho - 1) * size // 2):(sho * size // 2)],
                              train_data_2.iloc[((sho - 1) * size // 2):(sho * size // 2)]]).reset_index(drop=True)
        # sort by score and reset index
        # gt_df = train_df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
        gt_df = train_df.sort_values(by=score_column, ascending=False)
        # save the data before adding fairness
        # train_df_path = f"../Datasets/{experiment_name}/Train/"
        # scored_df = train_df[[train_df.columns[0], protected_feature, score_column]]
        gt_df.to_csv(f"{scored_save_path}/ranked_data_rank_size_{size}_shot_{sh}.csv", index=False)
        gt_df.to_csv(f"{scored_save_path}/ground_truth_rank_size_{size}_shot_{sh}.csv", index=False)
        # scored_dir = f"../Datasets/{experiment_name}/Train/Scored"
        cm.calculate_metrics_per_shot_llm(scored_save_path)
        skew_path = scored_save_path.replace("Datasets", "Results")
        if not os.path.exists(skew_path):
            os.makedirs(skew_path)
        # plot skew graph
        plot_skew(f"{skew_path}/skew.csv")

    if create_fair_data:
        # create Train folder if it does not exist
        if not os.path.exists(f"../Datasets/{experiment_name}/Train/Fair"):
            os.makedirs(f"../Datasets/{experiment_name}/Train/Fair")
        # get LTR-ranked data
        # check if LTR-rank folder is empty
        # if not os.path.exists(f"{train_df_path}/LTR_ranked"):
        #     print("LTR_ranked folder is empty. Please run LTR ranking first")
        #     return
        # rank using DetConstSort
        dcs.infer_with_detconstsort(f"{scored_save_path}/LTR_ranked/rank_size_{size}_shot_{sh}.csv")

    if create_fair_data_for_reranking:
        fair_rank_save_path = f"../Datasets/{experiment_name}/Train/Fair_Reranking/"
        # if not os.path.exists(fair_rank_save_path):
        #     os.makedirs(fair_rank_save_path)

        # rank using DetConstSort
        fair_rank_save_path_with_shot = f"{fair_rank_save_path}/shot_{sh}"
        dcs.infer_with_detconstsort(f"{scored_save_path}/ranked_data_rank_size_{size}_shot_{sh}.csv", post_process=True)
        fair_df = pd.read_csv(f"{fair_rank_save_path}/shot_{sh}/ranked_data_rank_size_{size}_shot_{sh}.csv")
        gt_fair_df = fair_df.sort_values(by='predictions', ascending=False).reset_index(drop=True)
        gt_fair_df.to_csv(f"{fair_rank_save_path}/shot_{sh}/ground_truth_rank_size_{size}_shot_{sh}.csv")
        # cm.calculate_metrics_per_shot_llm(fair_rank_save_path)
        # skew_path = fair_rank_save_path.replace("Datasets", "Results")
        # if not os.path.exists(skew_path):
        #     os.makedirs(skew_path)
        # # plot skew graph
        # plot_skew(f"{skew_path}/skew.csv")
        cm.calculate_metrics_per_shot_llm(fair_rank_save_path_with_shot)
        fair_skew_path = fair_rank_save_path_with_shot.replace("Datasets", "Results")
        if not os.path.exists(fair_skew_path):
            os.makedirs(fair_skew_path)
        # plot skew graph
        plot_skew(f"{fair_skew_path}/skew.csv")


def create_test_data(size=50, number=5, equal_distribution=False):
    """ this function creates n unique test data for the shots for LLM"""

    test_df = test_data.sample(n=size, random_state=42)
    # create Test folder if it does not exist
    test_folder = f"../Datasets/{experiment_name}/Tests"
    os.makedirs(test_folder, exist_ok=True)

    test_file_path = f"{test_folder}/rank_size_{size}.csv"

    # check if the file already exists to prevent overwriting
    if os.path.exists(test_file_path):
        print(f"Warning: {test_file_path} already exists. The file will not be overwritten.")
    else:
        test_df.to_csv(test_file_path, index=False)
        print(f"Test data saved to {test_file_path}")


def create_test_data_for_reranking(size=50, number=5, equal_distribution=True):
    """This function creates n unique test data for the shots for LLM reranking"""
    dcs_save_path = f"../Datasets/{experiment_name}/Ranked/DetConstSort"
    if not os.path.exists(dcs_save_path):
        os.makedirs(dcs_save_path)
    for i in range(number):
        # create empty dataframe
        new_test_df = pd.DataFrame(columns=test_data.columns)
        save_dir = f"../Datasets/{experiment_name}/Tests/Reranking_{i}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f"{save_dir}/ranked_data_rank_size_{size}_{i}.csv"
        # randomly pick an index for the test data and select a row from the test data_a and test data_b
        if equal_distribution:
            check_size = size // 2
        while len(new_test_df) < size:
            while len(new_test_df[new_test_df[protected_feature] == 'female']) < check_size:
                index_a = random.randint(0, len(test_data_1) - 1)
                # Append rows using pd.concat
                new_test_df = pd.concat([new_test_df, test_data_1.iloc[[index_a]]], ignore_index=True)
                new_test_df = pd.concat([new_test_df, test_data_2.iloc[[index_a]]], ignore_index=True)

                # sort by score
                new_test_df = new_test_df.sort_values(by=score_column, ascending=False)

                # get the ranked unique ids and group ids
                if 'Student ID' in new_test_df.columns:
                    ranked_unique_ids = new_test_df["Student ID"].tolist()
                else:
                    ranked_unique_ids = new_test_df["doc_id"].tolist()
                ranked_unique_ids = [int(id) for id in ranked_unique_ids]
                group_ids = new_test_df[protected_feature].tolist()
                group_ids = [1 if id == protected_group else 0 for id in group_ids]

                # measure the skew, if the skew for protected data is greater 1.0 or skew for non-protected data is less
                # Ensure pos is within bounds using try-except
                try:
                    # Calculate skew
                    skew_0 = sk(np.array(ranked_unique_ids), np.array(group_ids), non_protected_group, i)
                    skew_1 = sk(np.array(ranked_unique_ids), np.array(group_ids), protected_group, i)
                    print('i:', i)
                    print('skew_0:', skew_0)
                    print('skew_1:', skew_1)
                except Exception as e:
                    # Handle the specific exception for out-of-bounds error
                    if "Pos is not within the bounds of the arrays" in str(e):
                        print(f"Skipping iteration {i} due to error: {e}")
                        continue  # Skip the current iteration and move to the next one
                    else:
                        # Raise other exceptions that are not related to 'pos'
                        raise

                # skew_0 = sk(np.array(ranked_unique_ids), np.array(group_ids), non_protected_group, i)
                # skew_1 = sk(np.array(ranked_unique_ids), np.array(group_ids), protected_group, i)
                if skew_0 > 1.0 or skew_1 < 1.0:
                    new_test_df = new_test_df.drop(index=new_test_df.index[-1])
                    new_test_df = new_test_df.append(test_data_1.iloc[index_a])
                    continue

        # save the data
        new_test_df.to_csv(save_path, index=False)
        #create ground truth ranking
        gt_df = new_test_df.sort_values(by=score_column, ascending=False)
        # save the data
        gt_df.to_csv(save_path, index=False)
        gt_df.to_csv(f"{save_dir}/ground_truth_rank_size_{size}_{i}.csv", index=False)

        # check skew
        cm.calculate_metrics_per_shot_llm(save_dir)
        skew_path = save_dir.replace("Datasets", "Results")
        if not os.path.exists(skew_path):
            os.makedirs(skew_path)
        # plot skew graph
        plot_skew(f"{skew_path}/skew.csv")
        dcs.infer_with_detconstsort(f"{save_dir}/ranked_data_rank_size_{size}_{i}.csv", post_process=True,
                                    test_data=True)
        # read the DetConstSort ranked data
        ranked_data = pd.read_csv(
            f"{dcs_save_path}/prompt_NAD/rank_size_{size}/shot_NAD/ranked_data_rank_size_{size}_{i}.csv")
        # rank by score
        ranked_data = ranked_data.sort_values(by='GT_score', ascending=False)
        # save as ground truth
        ranked_data.to_csv(
            f"{dcs_save_path}/prompt_NAD/rank_size_{size}/shot_NAD/ranked_data_rank_size_{size}_{i}.csv_ground_truth.csv",
            index=False)
    flatten_directories(f"../Datasets/{experiment_name}/Tests/")
    # create Initial folder in Ranked
    if not os.path.exists(f"../Datasets/{experiment_name}/Ranked/Initial/prompt_NA/rank_size_{size}/shot_NA"):
        os.makedirs(f"../Datasets/{experiment_name}/Ranked/Initial/prompt_NA/rank_size_{size}/shot_NA")
    # copy all ground truth files to Initial folder
    for file in os.listdir(f"../Datasets/{experiment_name}/Tests/"):
        shutil.copy(f"../Datasets/{experiment_name}/Tests/{file}",
                    f"../Datasets/{experiment_name}/Ranked/Initial/prompt_NA/rank_size_{size}/shot_NA/{file}")
def create_non_unique_test_data(size=50, number=5):
    """ this function creates 5 non-unique test data for LLM ranking with same distribution"""
    # split data into 2 groups if using sex as a protected feature
    # create Test folder if it does not exist
    save_dir = f"../Datasets/{experiment_name}/Tests/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_dir = f"../Datasets/{experiment_name}/Tests/"
    # create 5 test data with same distribution, half and half for each group
    for i in range(number):
        test_df = pd.concat([test_data_1.iloc[:size // 2], test_data_2.iloc[:size // 2]])
        test_df = test_df.sample(frac=1, random_state=i).reset_index(drop=True)
        test_df.to_csv(f"{save_dir}/rank_size_{size}_{i}.csv", index=False)


def prep_different_dis_percent_data(size=50):
    """
    prepares data with different percentage of protected group
    :return:
    """
    safe_folder = f"../Datasets/{experiment_name}/Tests/Different_Dis_Percent"
    os.makedirs(safe_folder, exist_ok=True)
    num_samples = size
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # create 10 datasets, each with different percentage of protected group starting from 10 percent to 100 percent
    for pct in percentages:
        num_females = int(num_samples * pct)
        num_males = num_samples - num_females

        # Sample the required number of males and females
        sampled_females = test_data_1.sample(n=num_females, random_state=random.randint(1, 1000))
        sampled_males = test_data_2.sample(n=num_males, random_state=random.randint(1, 1000))

        # Combine the sampled males and females
        combined_dataset = pd.concat([sampled_females, sampled_males]).sample(frac=1).reset_index(drop=True)
        combined_dataset.to_csv(f"{safe_folder}/rank_size_size_{int(pct * 100)}.csv", index=False)


def create_non_unique_test_data_for_reranking_for_fairness(size=50):
    """ this function creates 1 non-unique test data for LLM reranking with same distribution"""
    if not os.path.exists(f"../Datasets/{experiment_name}/Tests"):
        os.makedirs(f"../Datasets/{experiment_name}/Tests")
    # create 1 test data with same distribution, half and half for each group
    test_df = pd.concat([test_data_1.iloc[:size // 2], test_data_2.iloc[:size // 2]])
    test_df = test_df.sample(frac=1, random_state=22).reset_index(drop=True)
    # leave only first column, score and protected feature. drop the rest
    test_df = test_df[[test_df.columns[0], protected_feature, score_column]]
    # rank by score
    test_df = test_df.sort_values(by=score_column, ascending=False)
    save_dir = f"../Datasets/{experiment_name}/Tests/"
    save_path = f"{save_dir}/ground_truth_ranked_by_score_for_fair_reranking_rank_size_{size}.csv"
    test_df.to_csv(save_path, index=False)
    cm.calculate_metrics_per_shot_llm(save_dir)

    skew_dir = save_dir.replace("Datasets", "Results")
    # plot skew graph
    plot_skew(f'{skew_dir}/test_skew.csv')


def metric_tests(df):
    """
    This function calculates the average exposure and plots skew for data when called
    :return:
    """


def flatten_directories(parent_dir):
    # Iterate through all subdirectories in the parent directory
    for subdir in [d for d in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, d))]:
        subdir_path = os.path.join(parent_dir, subdir)

        # Move all files and subdirectories to the parent directory
        for item in os.listdir(subdir_path):
            item_path = os.path.join(subdir_path, item)
            shutil.move(item_path, parent_dir)

        # Remove the now-empty subdirectory
        os.rmdir(subdir_path)


########################################################################################
# 1. Create test data for LLM-LTR fair ranking
# for rank_size in rank_sizes:
#     create_non_unique_test_data(rank_size, 5)
#     for shot in shots:
#         prep_LLM_data(rank_size, shot, create_fair_data=True)

# 2. Create shots for LLM-LTR ranking
# for shot in shots:
#     prep_LLM_data(5, shot)
# prep_different_dis_percent_data(20)

#################################### For reranking #################################################

# 3. Create test data for LLM reranking for fairness
# create_non_unique_test_data_for_reranking_for_fairness(20)
# this is the most recent one. It included the DetConstSort ranking and Initial ranking for the test data
create_test_data_for_reranking(size=20, number=10, equal_distribution=True)

# 4. Create shots for LLM reranking for fairness
# for shot in shots:
#     prep_LLM_data(20, shot, create_fair_data_for_reranking=True)
