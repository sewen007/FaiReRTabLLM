# this script prepares the data (already split) for the LLM ranking. It contains several functions that can be used to
# prep the data for different experiments.

import json
import os.path
import random
import shutil
import time
from .detconstsort import detconstsort as dcs, infer_with_detconstsort as iwd
import numpy as np
import pandas as pd
from ..data_analysis import calculate_metrics as cm
from ..data_analysis import skew as sk
from ..data_viz import plot_skew

start_time = time.time()

with open('./settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
rank_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()
non_protected_group = settings["READ_FILE_SETTINGS"]["ADV_GROUP"].lower()

gender_dict = {}
gender_dict = {'female': 1 if protected_group.lower() == 'female' else 0,
               'male': 0 if protected_group.lower() == 'female' else 1}

random_sate = 123


def prep_LLM_data(size=50, sh=1, create_fair_data=False, create_fair_data_for_reranking=False):
    """ this function creates the train data for the shots for LLM. The data is ranked used DetConstSort"""

    # do not create data for shot 0
    test_data = pd.read_csv(f"./Datasets/{experiment_name}/{experiment_name}_test_data_for_LLM.csv")
    train_data = pd.read_csv(f"./Datasets/{experiment_name}/{experiment_name}_train_data_for_LLM.csv")
    train_data_1 = train_data[train_data[protected_feature] == protected_group]
    train_data_0 = train_data[train_data[protected_feature] == non_protected_group]
    if sh == 0:
        return

    # create Scored data folder if it does not exist
    scored_save_path = f"./Datasets/{experiment_name}/Train/size_{size}/Scored/shot_{sh}"
    if not os.path.exists(scored_save_path):
        os.makedirs(scored_save_path)

    # split data into 2 groups if using sex as a protected feature and randomize them
    if protected_feature == 'Gender':

        ave_exp = 0.8
        ave_exp_dict = {'LOAN': 0.73, 'NBAWNBA': 0.68, 'LAW': 0.73, 'BostonMarathon': 0.75}

        ave_exp_limit = ave_exp_dict[experiment_name]

        while ave_exp >= ave_exp_limit:
            train_df = pd.DataFrame(columns=test_data.columns)
            # generate unique random indices
            random_indices_1 = random.sample(range(len(train_data_1)), size // 2)
            random_indices_0 = random.sample(range(len(train_data_0)), size // 2)
            # select the rows with the random indices
            train_df = pd.concat([train_df, train_data_1.iloc[random_indices_1]]).reset_index(drop=True)
            train_df = pd.concat([train_df, train_data_0.iloc[random_indices_0]]).reset_index(drop=True)

            # sort by score and reset index
            # gt_df = train_df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
            gt_df = train_df.sort_values(by=score_column, ascending=False)

            # save the data before adding fairness

            gt_df.to_csv(f"{scored_save_path}/ranked_data_rank_size_{size}_shot_{sh}.csv", index=False)
            gt_df.to_csv(f"{scored_save_path}/ground_truth_rank_size_{size}_shot_{sh}.csv", index=False)
            # scored_dir = f"../Datasets/{experiment_name}/Train/Scored"
            cm.calculate_metrics_per_shot_llm(scored_save_path, rank_size=size)
            # check exposure

            result_folder = f"{scored_save_path}".replace("Datasets", "Results")
            result = pd.read_csv(f"{result_folder}/metrics.csv")
            skew_path = scored_save_path.replace("Datasets", "Results")
            if not os.path.exists(skew_path):
                os.makedirs(skew_path)
            # plot skew graph
            plot_skew(f"{skew_path}/skew.csv", size=size)
            # get the Average Exposure
            ave_exp = result["Average Exposure"].iloc[0]
    # save current random indices, random_indices_1 and random_indices_0
    with open(f"{scored_save_path}/random_indices.json", 'w') as f:
        json.dump({'random_indices_1': random_indices_1, 'random_indices_0': random_indices_0}, f)

    # if create_fair_data:
    #     # create Train folder if it does not exist
    #     if not os.path.exists(f"../Datasets/{experiment_name}/Train/Fair"):
    #         os.makedirs(f"../Datasets/{experiment_name}/Train/Fair")
    #
    #     dcs.infer_with_detconstsort(f"{scored_save_path}/LTR_ranked/rank_size_{size}_shot_{sh}.csv")

    if create_fair_data_for_reranking:
        fair_rank_save_path = f"./Datasets/{experiment_name}/Train/size_{size}/Fair_Reranking/"
        if not os.path.exists(fair_rank_save_path):
            os.makedirs(fair_rank_save_path)

        p = [{protected_group: round(i, 1), non_protected_group: round(1 - i, 1)} for i in
             [x / 10 for x in range(6, 10)]]
        # p = [{protected_group:0., non_protected_group:0.5}]
        # rank using DetConstSort
        fair_rank_save_path_with_shot = f"{fair_rank_save_path}/shot_{sh}"
        for pe in p:
            iwd(f"{scored_save_path}/ranked_data_rank_size_{size}_shot_{sh}.csv",
                post_process=True, p_value=pe)
            fair_df = pd.read_csv(f"{fair_rank_save_path}/shot_{sh}/ranked_data_rank_size_{size}_shot_{sh}.csv")
            gt_fair_df = fair_df.sort_values(by='predictions', ascending=False).reset_index(drop=True)
            gt_fair_df.to_csv(f"{fair_rank_save_path}/shot_{sh}/ground_truth_rank_size_{size}_shot_{sh}.csv")
            cm.calculate_metrics_per_shot_llm(fair_rank_save_path_with_shot, rank_size=size)
            # get the Average Exposure
            result_folder = fair_rank_save_path_with_shot.replace("Datasets", "Results")
            result = pd.read_csv(f"{result_folder}/metrics.csv")
            # get the Average Exposure
            avg_exp = result["Average Exposure"].iloc[0]
            if 0.9 <= avg_exp <= 1.03:
                # if avg_exp == 1.0:
                # save pe and avg_exp to a file
                with open(f"{result_folder}/pe_avg_exp.csv", 'a') as f:
                    f.write(f"p={pe}, average_exposure={avg_exp}\n")
                # end the loop
                break

        fair_skew_path = fair_rank_save_path_with_shot.replace("Datasets", "Results")
        if not os.path.exists(fair_skew_path):
            os.makedirs(fair_skew_path)
        # plot skew graph
        plot_skew(f"{fair_skew_path}/skew.csv", size=size)


def create_test_data(size=50, number=5, equal_distribution=False):
    """ this function creates n unique test data for the shots for LLM"""
    test_data = pd.read_csv(f"./Datasets/{experiment_name}/{experiment_name}_test_data_for_LLM.csv")
    test_df = test_data.sample(n=size, random_state=42)
    # create Test folder if it does not exist
    test_folder = f"./Datasets/{experiment_name}/Tests"
    os.makedirs(test_folder, exist_ok=True)

    test_file_path = f"{test_folder}/rank_size_{size}.csv"

    # check if the file already exists to prevent overwriting
    if os.path.exists(test_file_path):
        print(f"Warning: {test_file_path} already exists. The file will not be overwritten.")
    else:
        test_df.to_csv(test_file_path, index=False)
        print(f"Test data saved to {test_file_path}")


def check_skew(df):
    position = len(df)
    skew_0 = 0
    skew_1 = 0

    # k = len(df)
    # sort by score
    df = df.sort_values(by=score_column, ascending=False)

    # get the ranked unique ids and group ids
    if 'Student ID' in df.columns:
        ranked_unique_ids = df["Student ID"].tolist()
    else:
        ranked_unique_ids = df["doc_id"].tolist()
    ranked_unique_ids = [int(id) for id in ranked_unique_ids]
    group_ids = df[protected_feature].tolist()
    group_ids = [1 if id == protected_group else 0 for id in group_ids]
    print('group_ids:', group_ids)

    # measure the skew, if the skew for protected data is greater 1.0 or skew for non-protected data is less
    # Ensure pos is within bounds using try-except
    try:
        # Calculate skew
        skew_0 = sk(np.array(ranked_unique_ids), np.array(group_ids), 0, position)
        skew_1 = sk(np.array(ranked_unique_ids), np.array(group_ids), 1, position)
        print('current length:', len(df))

        print('skew_0:', skew_0)
        print('skew_1:', skew_1)
    except Exception as e:
        # Handle the specific exception for out-of-bounds error
        if "Pos is not within the bounds of the arrays" in str(e):
            print(f"Skipping iteration {len(df)} due to error: {e}")
    return skew_0, skew_1


def map_to_target(num):
    # Map 0-9 to 1-4 cyclically
    target_values = [1, 2, 3, 4]
    return target_values[num % len(target_values)]


def create_test_data_for_reranking(size=50, number=5, equal_distribution=True):
    """This function creates n unique test data for the shots for LLM reranking"""

    test_data = pd.read_csv(f"./Datasets/{experiment_name}/{experiment_name}_test_data_for_LLM.csv")
    test_set = f"./Datasets/{experiment_name}/{experiment_name}_test_data.csv"
    full_size = len(pd.read_csv(test_set))

    if protected_feature == 'Gender':
        if size == full_size:
            pass
        else:
            # split data into two dataframes based on Gender column and shuffle them
            test_data_1 = test_data[test_data[protected_feature] == protected_group]
            test_data_0 = test_data[test_data[protected_feature] == non_protected_group]

            # order by score
            test_data_1_by_score = test_data_1.sort_values(by=score_column, ascending=False)
            test_data_0_by_score = test_data_0.sort_values(by=score_column, ascending=False)
            top_k = 10
            if size == 10:
                top_k = 4
            # pick top 10 of non-protected group and last 10 of protected group
            test_data_0_top_10 = test_data_0_by_score.head(top_k)
            # test_data_1_last_10 = test_data_1_by_score.tail(10)

            # change test_data_0 to be the rest of data after top 10 and shuffle
            test_data_0 = test_data_0_by_score.iloc[top_k:]
            test_data_0 = test_data_0.sample(frac=1, random_state=2).reset_index(drop=True)

            # change test_data_1 to be the rest of data before last 10
            test_data_1 = test_data_1_by_score.iloc[:-top_k]
            test_data_1 = test_data_1.sample(frac=1, random_state=2).reset_index(drop=True)
    if size == full_size:
        pass
    else:
        check_size_0 = len(test_data_0)
        check_size_1 = len(test_data_1)
        if equal_distribution:
            check_size_0 = size // 2
            check_size_1 = check_size_0

    for i in range(number):
        # create Reranking folder if it does not exist
        save_dir = f"./Datasets/{experiment_name}/Tests/size_{size}/Reranking_{i}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = f"{save_dir}/ranked_data_rank_size_{size}_{i}.csv"
        # create empty dataframe
        new_test_df = pd.DataFrame(columns=test_data.columns)
        if size == full_size:
            new_test_df = test_data
        else:
            rand_1 = random.randint(0, len(test_data_0_top_10) - 1)
            rand_2 = random.randint(0, len(test_data_1) - 1)
            # randomly pick from top 10 of non-protected group and last 10 of protected group
            new_test_df = pd.concat(
                [new_test_df, test_data_0_top_10.iloc[[rand_1]]], ignore_index=True)
            new_test_df = pd.concat(
                [new_test_df, test_data_1.iloc[[rand_2]]],
                ignore_index=True)
            # check that skew is within bounds
            skew_0, skew_1 = check_skew(new_test_df)
            # count_protected = 0
            if skew_0 < 1 or skew_1 > 1:
                while skew_0 < 1 or skew_1 > 1:
                    new_test_df = new_test_df.drop(index=new_test_df.index[-1])
                    # count_protected += 1
                    # pick another row from the protected group
                    rand_3 = random.randint(0, len(test_data_1) - 1)
                    new_test_df = pd.concat(
                        [new_test_df, test_data_1.iloc[[rand_3]]],
                        ignore_index=True)
                    skew_0, skew_1 = check_skew(new_test_df)
                # save random indices
                with open(f"{save_dir}/shot_{i}/random_indices.json", 'w') as f:
                    json.dump({'random_indices_1': [rand_1], 'random_indices_0': [rand_2]}, f)

            # now we have 2 rows in the new_test_df
            for l in range(1, check_size_1):
                # add non-protected group member
                rand_4 = random.randint(0, len(test_data_0) - 1)
                new_test_df = pd.concat([new_test_df, test_data_0.iloc[[rand_4]]],
                                        ignore_index=True)
                skew_0, skew_1 = check_skew(new_test_df)
                rand_5 = 'NAN'
                if skew_1 > 1 or skew_0 < 1:
                    while skew_1 > 1 or skew_0 < 1:
                        # drop the last row
                        new_test_df = new_test_df.drop(index=new_test_df.index[-1])
                        # pick another row from the protected group
                        # selected_rows = set()
                        rand_5 = random.randint(0, len(test_data_0) - 1)
                        new_test_df = pd.concat(
                            [new_test_df, test_data_0.iloc[[rand_5]]],
                            ignore_index=True)
                        skew_0, skew_1 = check_skew(new_test_df)
                rand_6 = random.randint(0, len(test_data_1) - 1)
                new_test_df = pd.concat([new_test_df, test_data_1.iloc[[rand_6]]], ignore_index=True)
                skew_0, skew_1 = check_skew(new_test_df)
                rand_7 = 'NAN'
                if skew_0 < 1 or skew_1 > 1:
                    while skew_0 < 1 or skew_1 > 1:
                        # drop the last row
                        new_test_df = new_test_df.drop(index=new_test_df.index[-1])
                        # pick another row from the non-protected group
                        rand_7 = random.randint(0, len(test_data_1) - 1)
                        new_test_df = pd.concat(
                            [new_test_df, test_data_1.iloc[[rand_7]]],
                            ignore_index=True)
                        skew_0, skew_1 = check_skew(new_test_df)
                # save random indices for l

                with open(f"{save_dir}/{i}_random_indices_{l}.json", 'w') as f:
                    json.dump({'random_indices_4': [rand_4], 'random_indices_5': [rand_5], 'random_indices_6': [rand_6],
                               'random_indices_7': [rand_7]}, f)
            new_test_df = new_test_df.sort_values(by=score_column, ascending=False)

        # create DetConstSort folder if it does not exist
        dcs_save_path = f"./Datasets/{experiment_name}/Ranked/DetConstSort"
        if not os.path.exists(dcs_save_path):
            os.makedirs(dcs_save_path)

        # reset 'doc_id' column
        new_test_df['doc_id'] = range(1, len(new_test_df) + 1)
        # save the data
        new_test_df.to_csv(save_path, index=False)
        print('savefile:', save_path)
        # create ground truth ranking
        gt_df = new_test_df.sort_values(by=score_column, ascending=False)
        # save the data
        gt_df.to_csv(save_path, index=False)
        gt_df.to_csv(f"{save_dir}/ground_truth_rank_size_{size}_{i}.csv", index=False)

        # check skew
        cm.calculate_metrics_per_shot_llm(save_dir, rank_size=size)
        skew_path = save_dir.replace("Datasets", "Results")
        if not os.path.exists(skew_path):
            os.makedirs(skew_path)
        # plot skew graph
        plot_skew(f"{skew_path}/skew.csv")
        # get p from corresponding shots

        if size == full_size:
            p = None
        else:
            # map i to shot value
            shot_value = map_to_target(i)
            print('p_folder before change:', save_dir)

            p_folder = save_dir.replace("Datasets", "Results")
            # p_folder = p_folder.replace(str(i), str(shot_value))
            p_folder = p_folder.replace("Tests", "Train")
            p_folder = p_folder.replace("Reranking_" + str(i), "Fair_Reranking/shot_" + str(shot_value))

            # get p from pe_avg_exp.csv
            with open(f"{p_folder}/pe_avg_exp.csv", 'r') as f:
                s = f.readlines()[0]
                start = s.index("{")  # Find the opening brace
                end = s.index("}") + 1  # Find the closing brace and include it
                dict_str = s[start:end]

                # Convert the string to an actual dictionary
                import ast
                p = ast.literal_eval(dict_str)

                # save p for reference
                with open(f"{save_dir}/p_value_{i}.txt", 'w') as g:
                    g.write(str(p))
        print('we outchea')
        iwd(f"{save_dir}/ranked_data_rank_size_{size}_{i}.csv", post_process=True,
            test_data=True, p_value=p)

        # read the DetConstSort ranked data
        ranked_data = pd.read_csv(
            f"{dcs_save_path}/prompt_NAD/rank_size_{size}/shot_NAD/ranked_data_rank_size_{size}_{i}.csv")
        # rank by score
        ranked_data = ranked_data.sort_values(by='GT_score', ascending=False)
        # save as ground truth
        ranked_data.to_csv(
            f"{dcs_save_path}/prompt_NAD/rank_size_{size}/shot_NAD/ranked_data_rank_size_{size}_{i}.csv_ground_truth.csv",
            index=False)
    flatten_directories(f"./Datasets/{experiment_name}/Tests/size_{size}")
    # create Initial folder in Ranked
    if not os.path.exists(f"./Datasets/{experiment_name}/Ranked/Initial/prompt_NA/rank_size_{size}/shot_NA"):
        os.makedirs(f"./Datasets/{experiment_name}/Ranked/Initial/prompt_NA/rank_size_{size}/shot_NA")
    src_dir = f"./Datasets/{experiment_name}/Tests/size_{size}"
    dst_dir = f"./Datasets/{experiment_name}/Ranked/Initial/prompt_NA/rank_size_{size}/shot_NA/"
    # copy all ground truth files to Initial folder
    # for file in os.listdir(f"./Datasets/{experiment_name}/Tests/"):
    #     shutil.copy(f"./Datasets/{experiment_name}/Tests/{file}",
    #                 f"./Datasets/{experiment_name}/Ranked/Initial/prompt_NA/rank_size_{size}/shot_NA/{file}")
    for file in os.listdir(src_dir):
        file_path = os.path.join(src_dir, file)
        if os.path.isfile(file_path):  # Ensures only files are copied
            shutil.copy(file_path, os.path.join(dst_dir, file))


#
# def create_non_unique_test_data(size=50, number=5):
#     """ this function creates 5 non-unique test data for LLM ranking with same distribution"""
#     # split data into 2 groups if using sex as a protected feature
#     # create Test folder if it does not exist
#     save_dir = f"../Datasets/{experiment_name}/Tests/"
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     save_dir = f"../Datasets/{experiment_name}/Tests/"
#     # create 5 test data with same distribution, half and half for each group
#     for i in range(number):
#         test_df = pd.concat([test_data_1.iloc[:size // 2], test_data_0.iloc[:size // 2]])
#         test_df = test_df.sample(frac=1, random_state=i).reset_index(drop=True)
#         test_df.to_csv(f"{save_dir}/rank_size_{size}_{i}.csv", index=False)

#
# def prep_different_dis_percent_data(size=50):
#     """
#     prepares data with different percentage of protected group
#     :return:
#     """
#     safe_folder = f"../Datasets/{experiment_name}/Tests/Different_Dis_Percent"
#     os.makedirs(safe_folder, exist_ok=True)
#     num_samples = size
#     percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     # create 10 datasets, each with different percentage of protected group starting from 10 percent to 100 percent
#     for pct in percentages:
#         num_females = int(num_samples * pct)
#         num_males = num_samples - num_females
#
#         # Sample the required number of males and females
#         sampled_females = test_data_1.sample(n=num_females, random_state=random.randint(1, 1000))
#         sampled_males = test_data_0.sample(n=num_males, random_state=random.randint(1, 1000))
#
#         # Combine the sampled males and females
#         combined_dataset = pd.concat([sampled_females, sampled_males]).sample(frac=1).reset_index(drop=True)
#         combined_dataset.to_csv(f"{safe_folder}/rank_size_size_{int(pct * 100)}.csv", index=False)


# def create_non_unique_test_data_for_reranking_for_fairness(size=50):
#     """ this function creates 1 non-unique test data for LLM reranking with same distribution"""
#     if not os.path.exists(f"../Datasets/{experiment_name}/Tests"):
#         os.makedirs(f"../Datasets/{experiment_name}/Tests")
#     # create 1 test data with same distribution, half and half for each group
#     test_df = pd.concat([test_data_1.iloc[:size // 2], test_data_0.iloc[:size // 2]])
#     test_df = test_df.sample(frac=1, random_state=22).reset_index(drop=True)
#     # leave only first column, score and protected feature. drop the rest
#     test_df = test_df[[test_df.columns[0], protected_feature, score_column]]
#     # rank by score
#     test_df = test_df.sort_values(by=score_column, ascending=False)
#     save_dir = f"../Datasets/{experiment_name}/Tests/"
#     save_path = f"{save_dir}/ground_truth_ranked_by_score_for_fair_reranking_rank_size_{size}.csv"
#     test_df.to_csv(save_path, index=False)
#     cm.calculate_metrics_per_shot_llm(save_dir)
#
#     skew_dir = save_dir.replace("Datasets", "Results")
#     # plot skew graph
#     plot_skew(f'{skew_dir}/test_skew.csv')


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


def Prep(size):
    test_set = f"./Datasets/{experiment_name}/{experiment_name}_test_data.csv"
    full_size = len(pd.read_csv(test_set))

    number = 10
    print('size:', size)
    if size == full_size:
        print('Size is full size')
        number = 1
    else:
        for shot in shots:
            prep_LLM_data(size, shot, create_fair_data_for_reranking=True)
    print('Done creatting shots')
    print('size:', size)
    create_test_data_for_reranking(size=size, number=number, equal_distribution=True)
# for shot in shots:
#     prep_LLM_data(20, shot, create_fair_data_for_reranking=True)
#
# # 4. Create test data for LLM reranking for fairness
# # create_non_unique_test_data_for_reranking_for_fairness(20)
# # this is the most recent one. It included the DetConstSort ranking and Initial ranking for the test data
# create_test_data_for_reranking(size=20, number=10, equal_distribution=True)


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

# 3. Create shots for LLM reranking for fairness
# for shot in shots:
#     prep_LLM_data(20, shot, create_fair_data_for_reranking=True)
#
# # 4. Create test data for LLM reranking for fairness
# # create_non_unique_test_data_for_reranking_for_fairness(20)
# # this is the most recent one. It included the DetConstSort ranking and Initial ranking for the test data
# create_test_data_for_reranking(size=20, number=10, equal_distribution=True)
