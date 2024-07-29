import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from data_analysis import kendall_tau, NDCG, NDKL, avgExp

# print working directory

with open('../settings.json', 'r') as f:
    settings = json.load(f)

start = time.time()

sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"]
dadv_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"]

feature_dict = {}

if dadv_group == 'female':
    adv_group = 'male'
else:
    adv_group = 'female'

feature_dict[dadv_group] = 1
feature_dict[adv_group] = 0


def calculate_metrics_per_shot_llm(shot_path, shot='shot_0', exp_name=experiment_name, rank_size='size_3'):
    """
    Calculate the Kendall Tau correlation coefficient between the ground truth and the inferred rankings
    :return: Kendall Tau correlation coefficient
    """
    path = shot_path + '/'
    ranked_folder = os.listdir(path)

    # list all ranked_files in the directory (not groundtruth)
    ranked_files = [f for f in ranked_folder if f.endswith('.csv') and 'ground_truth' not in f]

    # get the ground truth file
    if 'ListNet' in exp_name:
        ground_truth_file = None
    else:
        ground_truth_file = [f for f in ranked_folder if f.endswith('.csv') and 'ground_truth' in f][
            0]  # there should be only one ground truth file

    # create a new path by changing 'Datasets' to 'Results' in shot_path
    results_path = Path(shot_path.replace('Datasets', 'Results'))

    # check if the results path exists, if not create it
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    metrics_path = results_path / "metrics.csv"

    # open the metrics file
    with open(metrics_path, 'w', newline='') as f_metrics:
        writer = csv.writer(f_metrics)
        # write the header
        writer.writerow(["Run", "Kendall Tau", "NDKL", "Average Exposure"])  # Write the header once before the loop
        count = 0
        # for each file, read the inferred ranking and the ground truth ranking
        for file in ranked_files:
            file_path = path + file
            print(file_path)
            ranked_df = pd.read_csv(file_path)

            if 'ListNet' in exp_name:
                ground_truth_df = ranked_df.sort_values(by='GT_score', ascending=False)
                run_number = file.split('_')[0]
                ranked_unique_ids = ranked_df["doc_id"].tolist()
            else:
                # apply the feature_dict to the protected_feature column
                ranked_df[protected_feature] = ranked_df[protected_feature].map(feature_dict)
                ground_truth_df = pd.read_csv(path + ground_truth_file)
                ground_truth_df = ground_truth_df.sort_values(by=score_column, ascending=False)
                run_number = file.split('_')[-1].split('.')[0]
                if experiment_name == 'LAW':
                    # check ig the unique_id column exists
                    if 'Student ID' in ranked_df.columns:
                        ranked_unique_ids = ranked_df["Student ID"].tolist()
                    else:
                        ranked_unique_ids = ranked_df["doc_id"].tolist()
                else:
                    ranked_unique_ids = ranked_df["doc_id"].tolist()

            gt_unique_ids = ground_truth_df["doc_id"].tolist()
            group_ids = ranked_df[protected_feature].tolist()

            # Convert items in the lists to ints
            gt_unique_ids = [int(id) for id in gt_unique_ids]
            ranked_unique_ids = [int(id) for id in ranked_unique_ids]
            group_ids = [int(id) for id in group_ids]

            """ CALCULATE AND STORE METRICS"""
            kT_corr = kendall_tau(gt_unique_ids, ranked_unique_ids)
            ndkl = NDKL(np.array(gt_unique_ids), np.array(group_ids))
            avg_exp = avgExp.avg_exp(np.array(ranked_unique_ids), np.array(group_ids))
            exp_ratio = avg_exp[1] / avg_exp[0]
            # print(gt_unique_ids, ranked_unique_ids)
            writer.writerow([run_number, kT_corr[0], ndkl, exp_ratio])

            """ CALCULATE AND STORE NDCG"""
            print("Calculating NDCG...")
            GT_score = np.array(ranked_df.iloc[:, -1])
            GT_score_normalized = (GT_score - np.min(GT_score)) / (np.max(GT_score) - np.min(GT_score))
            ndcg_path = results_path / "ndcg.csv"
            ndcg_data = []
            for i in range(1, len(ranked_df) + 1):
                ndcg = NDCG(np.array(ranked_df.iloc[:, 0]), GT_score_normalized, i)
                ndcg_data.append([i, ndcg])

            if 'ListNet' in exp_name:
                if count == 0:
                    ndcg_header = ["Position", "NDCG_train_size_" + run_number]
                    with open(ndcg_path, 'w') as f_ndcg:
                        print("Writing to NDCG csv.")
                        writer_ndcg = csv.writer(f_ndcg)
                        # write the header
                        writer_ndcg.writerow(ndcg_header)

                        # write the data
                        writer_ndcg.writerows(ndcg_data)
                        count += 1
                else:
                    ndcg_df = pd.read_csv(ndcg_path)
                    ndcg_df["NDCG_train_size_" + run_number] = [item[1] for item in ndcg_data]
                    ndcg_df.to_csv(ndcg_path, index=False)

            else:

                if run_number == '1':
                    ndcg_header = ["Position"] + [f"NDCG_{number}" for number in range(1, int(run_number) + 1)]
                    # only write on the NDCG_1 column
                    with open(ndcg_path, 'w') as f_ndcg:
                        print("Writing to NDCG csv.")
                        writer_ndcg = csv.writer(f_ndcg)
                        # write the header
                        writer_ndcg.writerow(ndcg_header)

                        # write the data
                        writer_ndcg.writerows(ndcg_data)
                else:
                    ndcg_df = pd.read_csv(ndcg_path)
                    ndcg_df[f'NDCG_{run_number}'] = [item[1] for item in ndcg_data]
                    ndcg_df.to_csv(ndcg_path, index=False)


def CalculateResultMetrics():
    folder = Path(f"../Datasets/{experiment_name}/Ranked")
    # Get the list of experiments
    experiments = [f for f in os.listdir(folder) if os.path.isdir(folder / f)]
    for experiment in experiments:
        if 'llama' in experiment:
            experiment = 'meta-llama/Meta-Llama-3-8B-Instruct'
        experiment_path = folder / experiment
        prompts = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        for prompt in prompts:
            prompt_path = experiment_path / prompt
            # Get the list of sizes
            sizes = [f for f in os.listdir(prompt_path) if os.path.isdir(prompt_path / f)]
            for size in sizes:
                size_path = prompt_path / size
                # Get the list of shots
                shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
                for shot in shots:
                    shot_path = size_path / shot
                    print("Calculating metrics for shot", shot, "in experiment", experiment, "and size", size)
                    calculate_metrics_per_shot_llm(str(shot_path), str(shot), experiment, size)


CalculateResultMetrics()

end = time.time()
# print("time taken = ", end - start)
