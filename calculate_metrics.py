import csv
import json
import os
import re
from pathlib import Path

import pandas as pd
from scipy.stats import kendalltau

experiment_name = "LAW"

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))


# def get_files(directory):
#     temp = []
#     for dirpath, dirnames, filenames in os.walk(directory):
#         for file in filenames:
#             match = re.search(experiment_name, file)
#             if match:
#                 # temp.append(directory + '/' + file)
#                 temp.append(os.path.join(dirpath, file))
#     return temp


def kT(X, Y):
    """
    calculate kendall tau correlation coefficient between two rankings
    :param X: rank 1. use either the unique_ids or the index of the ranking
    :param Y: rank 2. use either the unique_ids or the index of the ranking
    :return: kendall tau correlation coefficient or a message if X and Y are not the same length
    """
    if len(X) != len(Y):
        return None, "X and Y are not the same length"

    corr, p_value = kendalltau(X, Y, variant='c')
    return corr, None


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
                    df = pd.DataFrame.from_records(json_array)

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


def CalculateResultMetrics(shot=0):
    """
    Calculate the Kendall Tau correlation coefficient between the ground truth and the inferred rankings
    :return: Kendall Tau correlation coefficient
    """
    # read all txt files in the data folder
    path = 'Llama3Output/LAW/shot_' + str(shot) + '/'
    ranked_folder = os.listdir(path)

    # list all txt_files in the directory
    inferred_files = [f for f in ranked_folder if f.endswith('.txt')]
    # count files without proper json
    count_err = 0
    # Open the CSV file outside the loop to ensure it's opened only once
    results_path = Path("../LLMFairRank/Results/LAW/shot_" + str(shot) + "/")
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    metrics_path = results_path / "metrics.csv"

    with open(metrics_path, 'w', newline='') as f_metrics:
        writer = csv.writer(f_metrics)
        writer.writerow(["Sample", "Kendall Tau"])  # Write the header once before the loop

        # for each file, read the inferred ranking and the ground truth ranking
        for file in inferred_files:
            file_path = path + file
            ranked_df = collect_json_after_second_occurrence(file_path)
            if ranked_df is None:
                count_err += 1
                continue

            sample_number = re.split(regex_pattern, file)[1]
            ground_truth_file = './Datasets/' + experiment_name + '/Splits/test_' + sample_number + '.csv'
            ground_truth_df = pd.read_csv(ground_truth_file)
            ground_truth_df = ground_truth_df.sort_values(by='ZFYA', ascending=False)

            gt_unique_ids = ground_truth_df["unique_id"].tolist()
            inferred_unique_ids = ranked_df["student_id"].tolist()
            kT_corr = kT(gt_unique_ids, inferred_unique_ids)

            print("Kendall Tau correlation coefficient for sample", sample_number, ":", kT_corr)

            # Write the data for each iteration on a new line in the CSV file
            writer.writerow([sample_number, kT_corr[0]])


CalculateResultMetrics()
CalculateResultMetrics(1)
CalculateResultMetrics(2)
