import csv
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from data_analysis import kendall_tau, NDCG

# print working directory

with open('../settings.json', 'r') as f:
    settings = json.load(f)

start = time.time()

sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
experiment_name = "LAW"

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))


def collect_json_after_second_occurrence(file_path):
    """
    Collects the JSON array that comes after the second occurrence of 'json' in a text file.

    :param file_path: Path to the text file
    :return: JSON array (if found), None otherwise
    """
    with open(file_path, 'r') as f:
        content = f.read()

    # Find the index of the second occurrence of 'json'
    second_json_index = content.find('json', content.find('json') + 1)

    if second_json_index != -1:
        # Find the start of the JSON array (assuming the array starts with '[')
        array_start = content.find('[', second_json_index)

        if array_start != -1:
            # Find the end of the JSON array (assuming it ends with ']')
            array_end = content.find(']', array_start) + 1  # +1 to include the closing ']'

            if array_end != -1:
                # Extract the JSON array substring
                json_array_str = content[array_start:array_end]

                try:
                    # Parse the JSON array
                    json_array = json.loads(json_array_str)
                    if isinstance(json_array, list) and all(isinstance(item, dict) for item in json_array):
                        df = pd.DataFrame.from_records(json_array)
                    else:
                        print("json_array is not a list of dictionaries")
                        return None

                    return df

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON:", e)
                    return None
            else:
                print("End of JSON array not found.")
                return None
        else:
            print("Start of JSON array not found.")
            return None
    else:
        print("Second occurrence of 'json' not found.")
        return None


def calculate_metrics_per_shot_llm(shot_path, shot='shot_0', exp_name=experiment_name, rank_size='size_3'):
    """
    Calculate the Kendall Tau correlation coefficient between the ground truth and the inferred rankings
    :return: Kendall Tau correlation coefficient
    """
    shot = shot.split('_')[-1]
    sample_size = rank_size.split('_')[-1]
    # read all txt files in the data folder
    path = shot_path + '/'
    ranked_folder = os.listdir(path)

    # list all ranked_files in the directory (not groundtruth)
    ranked_files = [f for f in ranked_folder if f.endswith('.csv') and 'ground_truth' not in f]
    ground_truth_file = [f for f in ranked_folder if f.endswith('.csv') and 'ground_truth' in f][
        0]  # there should be only one ground truth file

    # create a new path by changing 'Datasets' to 'Results' in shot_path
    results_path = Path(shot_path.replace('Datasets', 'Results'))

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    metrics_path = results_path / "metrics.csv"

    with open(metrics_path, 'w', newline='') as f_metrics:
        writer = csv.writer(f_metrics)
        writer.writerow(["Run", "Kendall Tau"])  # Write the header once before the loop

        # for each file, read the inferred ranking and the ground truth ranking
        for file in ranked_files:
            file_path = path + file
            print(file_path)
            ranked_df = pd.read_csv(file_path)
            ground_truth_df = pd.read_csv(path + ground_truth_file)
            ground_truth_df = ground_truth_df.sort_values(by='ZFYA', ascending=False)

            run_number = file.split('_')[-1].split('.')[0]

            gt_unique_ids = ground_truth_df["doc_id"].tolist()
            ranked_unique_ids = ranked_df["Student ID"].tolist()

            # Convert items in the lists to floats
            gt_unique_ids = [int(id) for id in gt_unique_ids]
            ranked_unique_ids = [int(id) for id in ranked_unique_ids]

            """ CALCULATE AND STORE KENDALL TAU"""
            kT_corr = kendall_tau(gt_unique_ids, ranked_unique_ids)
            print(gt_unique_ids, ranked_unique_ids)
            writer.writerow([run_number, kT_corr[0]])

            """ CALCULATE AND STORE NDCG"""
            print("Calculating NDCG...")
            GT_score = np.array(ranked_df.iloc[:, -1])
            GT_score_normalized = (GT_score - np.min(GT_score)) / (np.max(GT_score) - np.min(GT_score))
            ndcg_path = results_path / "ndcg.csv"
            ndcg_data = []
            for i in range(1, len(ranked_df) + 1):
                ndcg = NDCG(np.array(ranked_df.iloc[:, 0]), GT_score_normalized, i)
                ndcg_data.append([i, ndcg])

            if run_number == '1':
                ndcg_header = ["Position"] + [f"NDCG_{number}" for number in range(1, int(run_number) + 1)]
                # only write on the NDCG_1 columna
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
        # Get the list of sizes
        sizes = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        for size in sizes:
            size_path = experiment_path / size
            # Get the list of shots
            shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
            for shot in shots:
                shot_path = size_path / shot
                print("Calculating metrics for shot", shot, "in experiment", experiment, "and size", size)
                if 'ListNet' in experiment:
                    pass
                else:
                    calculate_metrics_per_shot_llm(str(shot_path), str(shot), experiment, size)


def calculate_metrics_per_shot_llama(shot='shot_0', experiment='meta-llama/Meta-Llama-3-8B-Instruct', size='size_3'):
    """
    Calculate the Kendall Tau correlation coefficient between the ground truth and the inferred rankings
    :return: Kendall Tau correlation coefficient
    """
    shot = shot.split('_')[1]
    sample_size = size
    # read all txt files in the data folder
    path = '../Datasets/' + experiment_name + '/Ranked/' + experiment + '/' + sample_size + '/shot_' + shot + '/'
    ranked_folder = os.listdir(path)

    # list all txt_files in the directory
    inferred_files = [f for f in ranked_folder if f.endswith('.txt')]
    # count files without proper json
    count_err = 0
    # Open the CSV file outside the loop to ensure it's opened only once
    # results_path = Path("../LLMFairRank/Results/LAW/size_" + sample_size + "/shot_" + str(shot) + "/")
    results_path = Path(path.replace('Datasets', 'Results'))

    if not os.path.exists(results_path):
        os.makedirs(results_path)
    metrics_path = results_path / "metrics.csv"

    with open(metrics_path, 'w', newline='') as f_metrics:
        writer = csv.writer(f_metrics)
        writer.writerow(["Sample", "Kendall Tau"])  # Write the header once before the loop

        # for each file, read the inferred ranking and the ground truth ranking
        for file in inferred_files:
            file_path = path + file
            print(file_path)
            ranked_df = collect_json_after_second_occurrence(file_path)
            if ranked_df is None:
                count_err += 1
                continue

            sample_number = re.split(regex_pattern, file)[1]
            print("Sample number:", sample_number)
            # ground_truth_file = './Datasets/' + experiment_name + '/Splits/size_' + sample_size + '/test_' +
            # sample_number + '.csv'
            ground_truth_file = '../Datasets/' + experiment_name + '/Ranked/gpt-3.5-turbo/' + sample_size + '/shot_' + shot + '/ground_truth.csv'
            ground_truth_df = pd.read_csv(ground_truth_file)
            ground_truth_df = ground_truth_df.sort_values(by='ZFYA', ascending=False)

            gt_unique_ids = ground_truth_df["Student ID"].tolist()
            inferred_unique_ids = ranked_df["Student ID"].tolist()

            # print(gt_unique_ids, inferred_unique_ids)

            # Convert items in the lists to floats
            gt_unique_ids = [int(id) for id in gt_unique_ids]
            inferred_unique_ids = [int(id) for id in inferred_unique_ids]

            kT_corr = kendall_tau(gt_unique_ids, inferred_unique_ids)
            print('here', len(gt_unique_ids), len(inferred_unique_ids))

            print("Kendall Tau correlation coefficient for sample", sample_number, ":", kT_corr)

            # Write the data for each iteration on a new line in the CSV file
            writer.writerow([sample_number, kT_corr[0]])


def collate_metrics():
    """
    Collate the metrics from the different shots into a single CSV file
    :return:
    """
    results_folder = Path("../Results/")
    # Get the list of experiments
    experiments = [f for f in os.listdir(results_folder) if os.path.isdir(results_folder / f)]
    for experiment in experiments:
        experiment_path = results_folder / experiment
        # Get the list of sizes
        sizes = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        for size in sizes:
            size_path = experiment_path / size
            collated_metrics_path = str(size_path / "collated_metrics_") + str(size) + ".csv"
            shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
            # reorder the shots
            shots = sorted(shots, key=lambda x: int(x.split('_')[1]))

            with open(collated_metrics_path, 'w', newline='') as f_collated_metrics:
                writer = csv.writer(f_collated_metrics)
                # write header row corresponding to the different shots
                writer.writerow(["Sample"] + shots)
                # write sample column
                # Write the rows with sample numbers and empty values for shots
                n = 50  # number of samples
                for i in range(1, n + 1):
                    row = [i] + ["" for _ in shots]  # Empty strings for each shot column
                    writer.writerow(row)

            for shot in shots:
                shot_path = size_path / shot
                metrics_path = shot_path / "metrics.csv"
                # read file
                metric_df = pd.read_csv(metrics_path)
                # sort the dataframe by the sample column
                metric_df = metric_df.sort_values(by="Sample", ascending=True)
                collated_metric_df = pd.read_csv(collated_metrics_path)
                # Check if the "shot" column exists in the original DataFrame
                if shot in collated_metric_df.columns:
                    # Ensure the length of new values matches the length of the original DataFrame
                    if len(metric_df) == len(collated_metric_df):
                        # Update the "shot" column with values from the new DataFrame
                        collated_metric_df[shot] = metric_df["Kendall Tau"].tolist()
                    else:

                        print(
                            "The length of the collated metrics DataFrame does not match the length of the original "
                            "DataFrame.")
                        continue
                else:
                    print("The 'shot' column does not exist in the original CSV file.")
                    continue

                # Save the updated DataFrame back to the CSV file
                collated_metric_df.to_csv(str(collated_metrics_path), index=False)

    # print(experiments)
    # print("Collating metrics...")


CalculateResultMetrics()
# collate_metrics()
# CalculateResultMetrics(1)
# CalculateResultMetrics(2)
# CalculateResultMetrics(3)
# CalculateResultMetrics(7)
# CalculateResultMetrics(10)

# kendall tau correlation test
# test
# ranking_ids_1 = [1, 2, 3, 4, 5]
# reversed_ranking_ids_1 = [5, 4, 3, 2, 1]
# test = kendall_tau(ranking_ids_1, reversed_ranking_ids_1)
# print(ranking_ids_1, reversed_ranking_ids_1, ', Kendall Tau = ', test)
#
# test_2 = kendall_tau(reversed_ranking_ids_1, ranking_ids_1)
# print(reversed_ranking_ids_1, ranking_ids_1, ', Kendall Tau = ', test_2)
#
# random = np.random.randint(1, 100, 5)
# test_4 = kendall_tau(ranking_ids_1, random)
# print(ranking_ids_1, random, ', Kendall Tau = ', test_4)
#
# kendall_zero = [5, 3, 4, 1, 2]
# test_6 = kendall_tau(ranking_ids_1, kendall_zero)
# print(ranking_ids_1, kendall_zero, ', Kendall Tau = ', test_6)
#
# misarranged = [4, 3, 2, 5, 1]
# test_5 = kendall_tau(ranking_ids_1, misarranged)
# print(ranking_ids_1, misarranged, ', Kendall Tau = ', test_5)
#
# misarranged = [2, 3, 2, 5, 2]
# test_5 = kendall_tau(ranking_ids_1, misarranged)
# print(ranking_ids_1, misarranged, ', Kendall Tau = ', test_5)
#
# ######################################################
# misarranged = [4, 3, 2, 5, 4]
# test_5 = kendall_tau(ranking_ids_1, misarranged)
# print(ranking_ids_1, misarranged, ', Kendall Tau = ', test_5)
#
# misarranged = [1, 3, 2, 5, 1]
# test_5 = kendall_tau(ranking_ids_1, misarranged)
# print(ranking_ids_1, misarranged, ', Kendall Tau = ', test_5)
#
# misarranged = [1, 3, 2, 5, 4]
# test_5 = kendall_tau(ranking_ids_1, misarranged)
# print(ranking_ids_1, misarranged, ', Kendall Tau = ', test_5)
#
# test_3 = kendall_tau(ranking_ids_1, ranking_ids_1)
# print(ranking_ids_1, ranking_ids_1, ', Kendall Tau = ', test_3)


end = time.time()
# print("time taken = ", end - start)
