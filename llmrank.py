from pathlib import Path
from calculate_metrics import kendall_tau
import pandas as pd
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation
import json, re, os
import numpy as np
import json
import os
import random
import re
import time
from pathlib import Path

import pandas as pd

# os.environ['TRANSFORMERS_CACHE'] = '/scratch/shared/models/'
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch
from hf_login import CheckLogin

start = time.time()

experiment_name = "LAW"
# variables for GPT
api_key = "sk-proj-BKhsxjRDcvt4ZBzlM63pT3BlbkFJlmacaHE8VVdLZwGhmV1P"
rank_size = 50
test_df = pd.read_csv('Datasets/' + experiment_name + '/' + experiment_name + '_test_data_for_LLM.csv')

with open('settings.json', 'r') as f:
    settings = json.load(f)

sample_sizes = settings["GENERAL_SETTINGS"]["sample_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))


# read data
# test_data = pd.read_csv('./Datasets/' + experiment_name + '/LAW_test_data.csv')
#
# seed = 123
#
# # sample_size = 5
#
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     print("CUDA is available.")
# else:
#     device = torch.device("cpu")
#     print("CUDA is not available, exiting")
#     exit()
# device_map = "auto"
#
# # model = "meta-llama/Llama-2-7b-chat-hf"
# model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
#
# # for fine-tuning note this from docs: "Training the model in float16 is not recommended and is known to produce
# # nan; as such, the model should be trained in bfloat16."
# pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map=device_map)
#
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name)
#
# prompt_addon = "```json"
#
#
# def serialize(df, row_index):
#     """
#     Takes a row of the data and serializes it as a string
#     returns: string
#     """
#     result = 'The student ID is ' + str(df.iloc[row_index]['unique_id']) + '. The gender is ' + str(
#         df.iloc[row_index]['Gender']) + ". The UGPA score is " + str(
#         df.iloc[row_index]['UGPA']) + ". The LSAT score is " + str(df.iloc[row_index]['LSAT']) + "."
#     return result
#
#
# def serialize_option_2(df, row_index):
#     """
#     Takes a row of the data and serializes it as a string
#     returns: string
#     """
#     # result = Student ID: 9105 (Gender: male, UGPA: 3.1, LSAT: 41)
#     result = 'Student ID: ' + str(df.iloc[row_index]['unique_id']) + ' (gender: ' + str(
#         df.iloc[row_index]['Gender'] + ', UGPA: ' + str(df.iloc[row_index]['UGPA']) + ', LSAT: ' + str(
#             df.iloc[row_index]['LSAT']) + ')')
#     return result


# def serialize_to_list(df):
#     """
#     Takes all rows of the data and serializes them as a list of strings
#     :param df:
#     :return: list of strings
#     """
#     listed_gt_data = pd.DataFrame()
#     list_index = list(range(0, len(df)))
#     for index in list_index:
#         listed_gt_data.loc[index, 0] = serialize_option_2(df, index)
#     return listed_gt_data.values.tolist()
#
#
# stop_token_ids = [
#     tokenizer.convert_tokens_to_ids(x) for x in [
#         ['```', ']']
#     ]
# ]
#
# stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
#
#
# # define custom stopping criteria object (# https://www.youtube.com/watch?v=-OXI2CZ_QgU)
# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         for stop_ids in stop_token_ids:
#             # stop_ids = stop_ids.squeeze()  # Ensure stop_ids is 1-dimensional
#             if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
#                 return True
#         return False
#
#
# stopping_criteria = StoppingCriteriaList([StopOnTokens()])


# def get_llama_response(prompt: str) -> None:
#     sequences = pipe(
#         prompt,
#         do_sample=True,
#         top_k=10,
#         num_return_sequences=1,
#         stopping_criteria=stopping_criteria,
#         eos_token_id=tokenizer.eos_token_id
#     )
#     return sequences[0]['generated_text']
#     # print("Chatbot:", "Llama response:", sequences[0]['generated_text'])


# def RankWithLlama(shot=0, rank_size=3):
#     CheckLogin()
#
#     # read each csv file from sampled directory and add to list
#     folder_path = Path('./Datasets/' + experiment_name + '/Splits/size_' + str(rank_size) + '/')
#     sample_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#
#     # get template for prompt
#     prompt_template = prepare_template(shot)
#
#     """ DIRECTORY MANAGEMENT """
#     result_path = Path("../LLMFairRank/Llama3Output/LAW2/size_" + str(rank_size) + "/shot_" + str(shot) + "/")
#
#     if not os.path.exists(result_path):
#         os.makedirs(result_path)
#
#     for sample in sample_list:
#         sample_number = re.split(regex_pattern, sample)[1]
#         # serialize sample
#         sample_path = os.path.join(folder_path, sample)
#         sample = pd.read_csv(sample_path)
#         serialized_sample = serialize_to_list(sample)
#         save_path = str(result_path) + '/output_' + str(sample_number) + '.txt'
#         prompt = prompt_template + str(serialized_sample) + prompt_addon
#         response = get_llama_response(prompt)
#         # sort the sample by the ZFYA column
#         sample = sample.sort_values(by='ZFYA', ascending=False)
#         with open(save_path, 'w') as file:
#             # Write data to the file
#             file.write(str(response))
#             file.write("\n\nGround truth: \n")
#             file.write(str(sample))
#

def prepare_template(shots=0, rank_size=3):
    size = rank_size
    base_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                     "applicants to predict their success in the program. The school wants to rank the applicants "
                     "using their UGPA score and LSAT scores. ")
    instruct_template = (" Without including explanations, rank the following applicants in a predicted "
                         "order of success in the program. "
                         "Return your ranked results in the following json only {""Student ID"": ""the students ID"",""gender"": ""the "
                         "student's gender""}:")
    shot_templates = []
    sampling_data = pd.read_csv('Datasets/' + experiment_name + '/' + experiment_name + '_test_data_for_LLM.csv')
    # based on ranking size, convert rows of data to sample formats
    # for i in range(shots+1):
    shot_template = base_template
    if shots == 0:
        shot_template += instruct_template
    else:
        for i in range(shots):
            shot_template += (pick_conjuction(i) + " example of ranked applicants in order of success in the "
                                                   "program is: ")
            # Randomly select  number of indices from the range of available indices
            random_indices = sorted(random.sample(range(len(sampling_data)), int(size)))

            # Create examples list with randomly selected rows
            examples = [row_converter(sampling_data.iloc[j]) for j in random_indices]

            enumerated_examples = []
            for k, item in enumerate(examples):
                enumerated_examples.append(str(k + 1) + ". " + item + " ")
            shot_template += ' '.join(enumerated_examples) + '. '

        shot_template += instruct_template
    shot_templates.append(shot_template)

    return str(shot_templates)


def row_converter(row):
    return "Student ID: " + str(row['unique_id']) + " (" + str(row['Gender']) + ", UGPA: " + str(row['UGPA']) + (
        ",LSAT: ") + str(row['LSAT']) + ")"


def pick_conjuction(i):
    conjuction_options = ["An", "Another", "Yet another", "And another"]
    if i == 0:
        return conjuction_options[0]
    elif i == 1:
        return conjuction_options[1]
    elif i == 2:
        return conjuction_options[2]
    else:
        return conjuction_options[3]


def RankWithLlamaMultipleSizesAndShots():
    for size in sample_sizes:
        for shot in shots:
            pass
            # RankWithLlama(shot, rank_size=size)


def RankWithGPT(model_name):
    df = test_df.sample(n=rank_size, random_state=1)
    item = create_items(df)
    # (1) Create permutation generation instruction
    messages = create_permutation_instruction(item=item, rank_start=0, rank_end=rank_size, model_name=model_name)
    # print('messages = ', messages)

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
    results_dir = Path(f'Results/{experiment_name}/{model_name}/rank_size_{rank_size}')

    # Create the directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    result = kendall_tau(gt_df['Student ID'].tolist(), ranked_df['Student ID'].tolist())
    return result, gt_df, merged_df


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


def create_items(df=test_df):
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

    # Hardcoded query
    query = ('You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to '
             'predict their success in the program. The school wants to rank the applicants using their UGPA score '
             'and LSAT scores. Without including explanations, rank the following applicants in a predicted order of '
             'success in the program. Return your ranked results in the following json only {student_id: the student\'s'
             'ID,gender: the student gender}:')

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


def RankkWithGPTMultipleRuns(model_name):
    results_dir = Path(f'Results/{experiment_name}/{model_name}/rank_size_{rank_size}')
    gt_df = RankWithGPT(model_name)[1]
    gt_df.to_csv(os.path.join(results_dir, 'ground_truth.csv'), index=False)
    for i in range(5):
        result = RankWithGPT(model_name)[0]
        ranked_df = RankWithGPT(model_name)[2]
        ranked_df.to_csv(os.path.join(results_dir, 'ranked_data_' + str(i + 1) + '.csv'), index=False)
        print(f"Run {i + 1} Kendall's Tau: {result}")


# RankWithGPT("gpt-3.5-turbo")
RankkWithGPTMultipleRuns("gpt-3.5-turbo")

# test = prepare_template(3)

# RankWithLlamaMultipleSizesAndShots()

# print(test)
# RankWithLLM()
# RankWithLLM(1)
# RankWithLLM(2)
# RankWithLLM(3)
# RankWithLLM(7)
# RankWithLLM(10)

end = time.time()

print("time taken = ", end - start)
