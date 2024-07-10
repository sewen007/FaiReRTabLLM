# from data_analysis.calculate_metrics import kendall_tau
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation
import json
import os
import random
import re
import time
from pathlib import Path
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import pandas as pd

os.environ['TRANSFORMERS_CACHE'] = '/scratch/shared/models/'

start = time.time()

experiment_name = "LAW"
# variables for GPT
api_key = "sk-proj-BKhsxjRDcvt4ZBzlM63pT3BlbkFJlmacaHE8VVdLZwGhmV1P"
rank_size = 50
test_df = pd.read_csv('../Datasets/' + experiment_name + '/' + experiment_name + '_test_data_for_LLM.csv')
train_path = '../Datasets/' + experiment_name + '/' + experiment_name + '_train_data_for_LLM.csv'

with open('../settings.json', 'r') as f:
    settings = json.load(f)

sample_sizes = settings["GENERAL_SETTINGS"]["sample_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))


def row_converter(row):
    return "Student ID: " + str(row['unique_id']) + " (" + str(row['Gender']) + ", UGPA: " + str(row['UGPA']) + (
        ",LSAT: ") + str(row['LSAT']) + ")"


def RankWithLlamaMultipleSizesAndShots():
    for size in sample_sizes:
        for shot in shots:
            pass
            # RankWithLlama(shot, rank_size=size)


def rank_with_GPT(model_name, number_of_shots=0):
    # df = test_df.sample(n=rank_size, random_state=1)
    df = pd.read_csv(f"../Datasets/{experiment_name}/Tests/rank_size_'{rank_size}.csv")
    item = create_items(df, number_of_shots)
    # (1) Create permutation generation instruction
    messages = create_permutation_instruction(item=item, rank_start=0, rank_end=rank_size, model_name=model_name)
    print('messages = ', messages)

    # (2) Get ChatGPT predicted permutation
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)

    # (3) Use permutation to re-rank the passage
    new_item = receive_permutation(item, permutation, rank_start=0, rank_end=rank_size)

    # Extract information and store in a list of dictionaries
    extracted_ranked_data = [extract_info(item['content']) for item in new_item['hits']]

    # Convert list of tuples to DataFrame
    ranked_df = pd.DataFrame(extracted_ranked_data)

    # order gt_df by ZFYA
    gt_df = df.sort_values(by=['ZFYA'], ascending=False)

    # rename unique_id to Student ID in gt_df
    gt_df.rename(columns={'unique_id': 'Student ID'}, inplace=True)

    # get gt scores from df
    merged_df = pd.merge(ranked_df, gt_df[['Student ID', 'ZFYA']], on='Student ID', how='left')

    # create directory for results based on n and model name
    results_dir = Path(
        f'../Datasets-backup/{experiment_name}/Ranked/{model_name}/rank_size_{rank_size}/shot_{number_of_shots}')

    # Create the directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # result = kendall_tau(gt_df['Student ID'].tolist(), ranked_df['Student ID'].tolist())
    return gt_df, merged_df


# Define a function to extract information from the content string
def extract_info(content):
    student_id = re.search(r'Student ID: (\d+)', content).group(1)
    gender = re.search(r'Gender: (\w+)', content).group(1)
    ugpa = re.search(r'UGPA: ([\d.]+)', content).group(1)
    lsat = re.search(r'LSAT: ([\d.]+)', content).group(1)
    return {
        'Student ID': int(float(student_id)),
        'Gender': gender,
        'UGPA': float(ugpa),
        'LSAT': float(lsat)
    }


def create_items(df, number_of_shots=0):
    """creates items with format item = {
    'query': 'You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to '
             'predict their success in the program. The school wants to rank the applicants using their UGPA score '
             'and LSAT scores.  Without including explanations, rank the following applicants in a predicted order of '
             'success in the program. Return your ranked results in the following json only {student_id: the student\'s '
             ' ID,gender: the student gender}:',
    'hits': [
        {'content': 'Student ID: 18739 (Gender: female, UGPA: 3.7, LSAT: 44.0)'},
        {'content': 'Student ID: 8742 (Gender: male, UGPA: 3.2, LSAT: 20.0)'},
        {'content': 'Student ID: 17967 (Gender: male, UGPA: 3.2, LSAT: 34.5)'},
        {'content': 'Student ID: 13964 (Gender: male, UGPA: 3.7, LSAT: 32.0)'}
    ]
}"""
    n_rank_size = len(df)
    s_shots = number_of_shots
    if s_shots == 0:

        # Hardcoded query
        query = ('You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to'
                 'predict their success in the program. The school wants to rank the applicants using their UGPA score '
                 'and LSAT scores. Rank the following applicants in a predicted order of success in the program. Return'
                 'your ranked results in the following json only {student_id: the student\'s'
                 'ID,gender: the student gender}:')
    else:
        # note that the shot examples come from the training data
        query = prepare_template(s_shots, n_rank_size)

    # Creating the hits list from the DataFrame
    hits = []
    for index, row in df.iterrows():
        content = f"Student ID: {row['unique_id']} (Gender: {row['Gender']}, UGPA: {row['UGPA']}, LSAT: {row['LSAT']})"
        hits.append({'content': content})

    # Creating the final JSON object
    result = {
        'query': query,
        'hits': hits
    }
    return result


def prepare_template(shot=0, size=3):
    base_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                     "applicants to predict their success in the program. The school wants to rank the applicants "
                     "using their UGPA score and LSAT scores. ")
    instruct_template = (" Rank the following applicants in a predicted order of success in the program. "
                         "Return your ranked results in the following json only {""Student ID"": ""the students ID"",""gender"": ""the "
                         "student's gender""}:")
    shot_templates = []
    sampling_data = pd.read_csv(train_path)
    # based on ranking size, convert rows of data to sample formats
    # for i in range(shots+1):
    shot_template = base_template
    if shot == 0:
        shot_template += instruct_template
    else:
        for i in range(shot):
            shot_template += (pick_conjunction(i) + " example of ranked applicants in order of success in the "
                                                    "program is: ")
            # Randomly select  number of indices from the range of available indices
            random_indices = sorted(random.sample(range(len(sampling_data)), int(size)))

            # create new df with randomly selected rows
            new_df = sampling_data.iloc[random_indices]
            new_df.to_csv(
                '../Datasets/' + experiment_name + '/Train/rank_size_' + str(size) + '_shot_' + str(i + 1) + '.csv',
                index=False)
            # Create examples list with randomly selected rows
            examples = [row_converter(sampling_data.iloc[j]) for j in random_indices]

            enumerated_examples = []
            for k, item in enumerate(examples):
                enumerated_examples.append(str(k + 1) + ". " + item + " ")
            shot_template += ' '.join(enumerated_examples) + '. '

        shot_template += instruct_template
    shot_templates.append(shot_template)

    return str(shot_templates)


def pick_conjunction(i):
    conjunction_options = ["An", "Another", "Yet another", "And another"]
    if i == 0:
        return conjunction_options[0]
    elif i == 1:
        return conjunction_options[1]
    elif i == 2:
        return conjunction_options[2]
    else:
        return conjunction_options[3]


def RankWithGPT(model_name, shot_number=1, runs=5):
    results_dir = Path(
        f'../Datasets-backup/{experiment_name}/Ranked/{model_name}/rank_size_{rank_size}/shot_{shot_number}')
    gt_df = rank_with_GPT(model_name, shot_number)[0]
    gt_df.to_csv(os.path.join(results_dir, 'ground_truth.csv'), index=False)
    for i in range(runs):
        # result = rank_with_GPT(model_name, shot_number)[0]
        ranked_df = rank_with_GPT(model_name, shot_number)[1]
        ranked_df.to_csv(os.path.join(results_dir, 'ranked_data_' + str(i + 1) + '.csv'), index=False)
        # print(f"Run {i + 1} Kendall's Tau: {result}")


RankWithGPT("gpt-3.5-turbo")

end = time.time()

print("time taken = ", end - start)
