# from data_analysis.calculate_metrics import kendall_tau
import io
import random

import google.api_core.exceptions
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
import pathlib

from vertexai.preview import tokenization

from hf_login import CheckLogin
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation, sliding_windows, permutation_pipeline
import json
import os
import re
import time
from pathlib import Path
import pandas as pd

# os.environ['TRANSFORMERS_CACHE'] = '/scratch/shared/models/'
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

start = time.time()
# torch.cuda.empty_cache()
# variables for GPT
api_key = "AIzaSyC2x3jP6uRHeXTJ2JtOtmEltExHSfs4G14"

with open('../settings.json', 'r') as f:
    settings = json.load(f)

sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]
item_type = settings["READ_FILE_SETTINGS"]["ITEM"].lower()
protected_feature = settings["READ_FILE_SETTINGS"]["PROTECTED_FEATURE"].lower()
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()
non_protected_group = settings["READ_FILE_SETTINGS"]["ADV_GROUP"].lower()

test_set = f"../Datasets/{experiment_name}/{experiment_name}_test_data.csv"

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))

if protected_feature == "gender":
    prot = 'sex'


def RankWithLLM(model_name, shot_number=1, size=50, prompt_id=1, ltr_ranked=False, varied_percentage=False,
                post_process=False):
    """
    This function ranks the data using a language model
    :param post_process:
    :param varied_percentage: experiment with different percentages of disadvantaged group
    :param ltr_ranked: True if the data is already ranked using LTR
    :param prompt_id: prompt 0 is the neutral prompt
    :param model_name: LLM model name
    :param shot_number: number of examples. Each example has the size of the rank
    :param size: size of the rank
    :return:
    """
    print('model_name = ', model_name)
    print('prompt_id = ', prompt_id)
    # specify the test folder. if pre_ranked experiment, use ListNet ranked data
    post_process = post_process
    if varied_percentage:
        if ltr_ranked:
            test_folder = f"../Datasets/{experiment_name}/Different_Dis_Percent/Ranked/ListNet/prompt_NA/rank_size_{size}/shot_NA"
            results_dir = Path(
                f'../Datasets/{experiment_name}/Different_Dis_Percent/Ranked/{model_name}_pre-ranked/prompt_{prompt_id}/rank_size_{size}/shot_{shot_number}')
        else:
            test_folder = f"../Datasets/{experiment_name}/Tests/Different_Dis_Percent"
            results_dir = Path(
                f'../Datasets/{experiment_name}/Different_Dis_Percent/Ranked/{model_name}/prompt_{prompt_id}/rank_size_{size}/shot_{shot_number}')
    else:
        if ltr_ranked:
            test_folder = f"../Datasets/{experiment_name}/Ranked/ListNet/prompt_NA/rank_size_{size}/shot_NA"
            results_dir = Path(
                f'../Datasets/{experiment_name}/Ranked/{model_name}_pre-ranked/prompt_{prompt_id}/rank_size_{size}/shot_{shot_number}')
        else:
            test_folder = f"../Datasets/{experiment_name}/Tests"
            results_dir = Path(
                f'../Datasets/{experiment_name}/Ranked/{model_name}/prompt_{prompt_id}/rank_size_{size}/shot_{shot_number}')
    # check test folder for all files with rank_size
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    test_files = [f for f in os.listdir(test_folder) if
                  f'ranked_data_rank_size_{size}' in f and os.path.isfile(os.path.join(test_folder, f))]
    for file in test_files:
        # sort by score to get the ground truth
        gt_df = pd.read_csv(os.path.join(test_folder, file))
        gt_df = gt_df.sort_values(by=[score_column], ascending=False)
        # save the ground truth
        gt_df.to_csv(os.path.join(results_dir, f'{os.path.basename(file)}_ground_truth.csv'), index=False)

        if "llama" in model_name:
            ranked_df = rank_with_llama(model_name, file, shot_number, size, prompt_id, test_folder, post_process)[1]

            ranked_df.to_csv(os.path.join(results_dir, f'ranked_data_{os.path.basename(file)}'), index=False)

        else:  # "gpt" in model_name:
            if "gpt" in model_name:
                ranked_df = \
                    rank_with_GPT_or_Gemini(model_name, file, shot_number, size, prompt_id, test_folder,
                                            model_type='gpt', post_process=post_process)[
                        1]

            else:
                ranked_df = rank_with_GPT_or_Gemini(model_name, file, shot_number, size, prompt_id, test_folder,
                                                    model_type='gemini', post_process=post_process)[1]
            # print('ranked_df = ', ranked_df)

            ranked_df.to_csv(os.path.join(results_dir, f'ranked_data_{os.path.basename(file)}'), index=False)


def rank_with_llama(model_name, file, number_of_shots=0, size=5, prompt_id=1,
                    test_folder=f'../Datasets/{experiment_name}/Tests', post_process=False):
    CheckLogin()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, exiting")
        exit()

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(os.path.join(test_folder, file))

    item = create_items(df, number_of_shots, prompt_id, post_process)

    # (1) Create permutation generation instruction
    messages = create_permutation_instruction(df, item=item, rank_start=0, rank_end=size, item_type=item_type,
                                              prompt_id=prompt_id,
                                              model_type='llama')
    # TODO sample should not be hardcoded. Code based on size
    sample = ('[20] > [4] > [14] > [3] > [10] > [9] > [11] > [5] > [17] > [1] > [6] > [16] > [15] > [19] > [7] > [12] '
              '> [2] > [8] > [13] > [18]')
    # count the number of tokens
    token_number = get_tokens_and_count(str(messages), tokenizer)
    sample_token_number = get_tokens_and_count(sample, tokenizer)

    total_tokens = token_number + sample_token_number

    if model_name == "meta-llama/Meta-Llama-3.1-8B-Instruct":
        if total_tokens > 128000:
            print('total tokens = ', total_tokens)
            print('tokens exceed 128000')
            return
    else:
        if total_tokens > 8000:
            print('total tokens = ', total_tokens)
            print('tokens exceed 8000')
            return
    results_dir = Path(
        f'../Datasets/{experiment_name}/Ranked/{model_name}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')

    # Create the directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # save the messages to txt
    with open(results_dir / 'messages.txt', 'w') as f:
        for message in messages:
            f.write(f"{message}")
            # f.write(f"{message['role']}: {message['content']}\n")

    template_prompt_pred = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    print('template after applying chat template = ', template_prompt_pred)
    template_prompt_pred += '<|start_header_id|>assistant<|end_header_id|>\n\n'
    #print('template after adding assistant role = ', template_prompt_pred)
    inputs_template_pred = tokenizer(template_prompt_pred, add_special_tokens=False, return_tensors='pt')
    # print('inputs_template_pred = ', inputs_template_pred)
    inputs_template_pred = inputs_template_pred.to(device)
    #
    # # generate an output using the template prompt and print only the model generated tokens
    outputs_template_pred = model.generate(**inputs_template_pred, pad_token_id=tokenizer.eos_token_id,
                                           return_dict_in_generate=True)
    generated_tokens_template_pred = outputs_template_pred.sequences[:,
                                     inputs_template_pred.input_ids.shape[1]:]  # for decoder only models
    generated_text_template_pred = tokenizer.decode(generated_tokens_template_pred[0], skip_special_tokens=True)
    print('generated_result = ', generated_text_template_pred)

    # Use permutation to re-rank the passage
    new_item = receive_permutation(item, generated_text_template_pred, rank_start=0, rank_end=size)
    # Extract information and store in a list of dictionaries
    gt_df, merged_df = extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots, size)

    return gt_df, merged_df
    # return df, df


def rank_with_GPT_or_Gemini(model_name, file, number_of_shots=0, size=20, prompt_id=1,
                            test_folder=f'../Datasets/{experiment_name}/Tests', model_type='gpt', post_process=False):
    # read the test data
    df = pd.read_csv(os.path.join(test_folder, file))

    # serialize the rows of the DataFrame
    item = create_items(df, number_of_shots, prompt_id, post_process)

    # add pre-prompt, fairness instruction and post-prompt
    messages = create_permutation_instruction(df, item=item, rank_start=0, rank_end=size, item_type=item_type,
                                              prompt_id=prompt_id, model_type=model_type)
    # messages = messages.replace("['", "").replace("']", "")
    print('messages = ', messages)

    # create directory for results based on n and model name
    results_dir = Path(
        f'../Datasets/{experiment_name}/Ranked/{model_name}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')

    # create the directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # save the prompt to a txt file
    with open(results_dir / 'messages.txt', 'w') as f:
        for message in messages:
            # Check if the message has 'content' before writing it
            if 'content' in message:
                # f.write(f"{message['role']}: {message['content']}\n")
                f.write(f"{message}")
            else:  # save the message as is, (no rol)
                f.write(f"{message}")

    # get LLM predicted permutation
    if model_type == 'gpt':
        permutation = run_llm(messages, api_key=api_key, model_name=model_name)
        # permutation = permutation_pipeline(messages, model_name=model_name)

    else:
        # pass
        for attempt in range(3):
            try:
                permutation = rank_with_gemini(key=GEMINI_API_KEY, messages=messages)
                # print('permutation = ', permutation)
            except google.api_core.exceptions.ResourceExhausted:
                delay = 20 + random.uniform(0, 3)
                time.sleep(delay)

    # use permutation to re-rank the passage
    new_item = receive_permutation(item, permutation, rank_start=0, rank_end=size)

    # extract information and store

    gt_df, merged_df = extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots, size)

    return gt_df, merged_df
    # return df, df


def rank_with_gemini(key="<api_key>", messages=None, model_name="gemini-1.5-flash"):
    # get gemini tokenizer
    tokenizer = tokenization.get_tokenizer_for_model(model_name)

    # count the number of tokens
    token_number = tokenizer.count_tokens(str(messages))
    print('number of tokens = ', token_number.total_tokens)
    model = genai.GenerativeModel(model_name)
    try:
        # key = os.getenv('GEMINI_API_KEY')
        genai.configure(api_key='AIzaSyAjNV30pZOZfZ9i__h45WHnwqrHZtELuZ4')
        response = model.generate_content(str(messages))
        # key = os.getenv('GEMINI_API_KEY')
        # genai.configure(api_key=key)

        print(response.text)
        return response.text
    except google.api_core.exceptions.ResourceExhausted:
        print('Resource exhausted. Trying again')
        time.sleep(20)
        return rank_with_gemini(key, messages, model_name)


def rank_entire_test(window_size=20, step=5, model_name='gpt-3.5-turbo', prompt_id=1, number_of_shots=0):
    df = pd.read_csv(test_set)
    size = len(df)
    item = create_items(df, 0, 1)
    # sliding windows
    new_item = sliding_windows(item, rank_start=0, rank_end=size, window_size=window_size, step=step,
                               model_name='gpt-3.5-turbo',
                               api_key=api_key)
    results_dir = Path(
        f'../Datasets/{experiment_name}/Ranked/{model_name}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')
    gt_df, ranked_df = extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots, size)
    ranked_df.to_csv(os.path.join(results_dir, 'ranked_data' + '.csv'), index=False)


def extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots=0, size=50):
    # Extract information and store in a list of dictionaries
    if prompt_id == 3:
        ranked_df = new_item['hits'][0]['content']
        ranked_df = pd.read_csv(io.StringIO(ranked_df), header=0, index_col=0)
        if experiment_name == 'LAW':
            ranked_df.columns = ['doc_id', 'Gender', 'UGPA', 'LSAT', 'ZFYA']
        # elif experiment_name == 'NBAWNBA':
        #     ranked_df.columns = ['doc_id', 'Gender', 'Career Points']
        else:
            ranked_df.columns = ['doc_id', 'Gender', 'score']
    else:
        extracted_ranked_data = [extract_info(item['content']) for item in new_item['hits']]
        print('extracted_ranked_data = ', extracted_ranked_data)
        ranked_df = pd.DataFrame(extracted_ranked_data)
    # Convert list of tuples to DataFrame

    # order gt_df by score_column
    gt_df = df.sort_values(by=[score_column], ascending=False)

    if 'doc_id' in gt_df.columns:
        ranked_df.rename(columns={'ID': 'doc_id'}, inplace=True)
        ranked_df['doc_id'] = ranked_df['doc_id'].astype(str)
        gt_df['doc_id'] = gt_df['doc_id'].astype(str)

        merged_df = pd.merge(ranked_df, gt_df[['doc_id', score_column]], on='doc_id', how='left')

    else:
        merged_df = pd.merge(ranked_df, gt_df[['doc_id', score_column]], on='doc_id', how='left')

    return gt_df, merged_df


# Define a function to extract information from the content string
# def extract_info(content):
#     if experiment_name == 'LAW':
#         student_id = re.search(r'Student ID: (\d+)', content).group(1)
#         gender = re.search(r'Gender: (\w+)', content).group(1)
#         ugpa = re.search(r'UGPA: ([\d.]+)', content).group(1)
#         lsat = re.search(r'LSAT: ([\d.]+)', content).group(1)
#         return {
#             'Student ID': int(float(student_id)),
#             'Gender': gender,
#             'UGPA': float(ugpa),
#             'LSAT': float(lsat)
#         }
#     elif experiment_name == 'dummy':  # experiment_name == 'dummy':
#         parts = content.split(',')
#         doc_id = re.search(r'Unique ID: (\d+)', parts[0]).group(1)
#         number = re.search(r'Number: ([\d.]+)', parts[1]).group(1)
#         return {
#             'doc_id': int(float(doc_id)),
#             'Number': float(number)
#         }
#     elif experiment_name == 'bostonmarathon':
#         parts = content.split(',')
#         doc_id = re.search(r'Unique ID: (\d+)', parts[0]).group(1)
#         gender = re.search(r'Gender: (\w+)', content).group(1)
#         k_5 = re.search(r'5K: (\w+)', content).group(1)
#         k_10 = re.search(r'10K: (\w+)', content).group(1)
#         k_15 = re.search(r'15K: (\w+)', content).group(1)
#         k_20 = re.search(r'20K: (\w+)', content).group(1)
#         half = re.search(r'Half: (\w+)', content).group(1)
#         k_30 = re.search(r'30K: (\w+)', content).group(1)
#         return {
#             'doc_id': int(float(doc_id)),
#             'Gender': gender,
#             '5K': k_5,
#             '10K': k_10,
#             '15K': k_15,
#             '20K': k_20,
#             'Half': half,
#             '30K': k_30
#         }

def extract_info(content):
    # Dictionary to store extracted values
    extracted_values = {}

    # Pattern to match key-value pairs in the content
    key_value_pattern = r'(\w+): ([\w.-]+)'

    # Extract all key-value pairs
    matches = re.findall(key_value_pattern, content)

    # Handle each key-value pair
    for key, value in matches:
        extracted_values[key] = value
        # Convert to appropriate type if necessary
        # if key in ['Student ID', 'Unique ID', 'doc_id']:
        #     extracted_values[key] = int(float(value))
        # elif key in ['UGPA', 'LSAT', 'Number']:
        #     extracted_values[key] = float(value)
        # else:
        #     extracted_values[key] = float(value)
    # print('extracted_values = ', extracted_values)

    return extracted_values


def create_items(df, number_of_shots=0, prompt_id=1, post_process=False):
    """creates items with format item = {
    'query': 'Re-rank the following list to incorporate fairness. ',
    'hits': [
        {'content': 'Student ID: 18739 (Gender: female, UGPA: 3.7, LSAT: 44.0)'},
        {'content': 'Student ID: 8742 (Gender: male, UGPA: 3.2, LSAT: 20.0)'},
        {'content': 'Student ID: 17967 (Gender: male, UGPA: 3.2, LSAT: 34.5)'},
        {'content': 'Student ID: 13964 (Gender: male, UGPA: 3.7, LSAT: 32.0)'}
    ]
}"""
    print('number_of_shots = ', number_of_shots)
    n_rank_size = len(df)
    s_shots = number_of_shots

    # get the query
    if post_process:
        query = prepare_fair_rerank_template(s_shots, n_rank_size, prompt_id)
    else:
        query = prepare_fair_rerank_template(s_shots, n_rank_size, prompt_id)
        # query = prepare_template(df, s_shots, n_rank_size, prompt_id)

    # Create the hit
    hits = []
    if prompt_id == 3:
        hits.append({'content': str(df)})
    else:  # serialize the rows of the DataFrame
        if experiment_name == 'LAW':
            for index, row in df.iterrows():
                if post_process:
                    content = f"Student ID: {row['doc_id']} (sex: {row['Gender']}, score: {row[score_column]})"
                else:
                    content = f"Student ID: {row['doc_id']} (sex: {row['Gender']}, UGPA: {row['UGPA']}, LSAT: {row['LSAT']})"
                hits.append({'content': content})
        else:
            for index, row in df.iterrows():
                content_parts = [f"ID: {row['doc_id']} "]
                for column in df.columns:
                    if column != 'doc_id':
                        if column == 'Gender':
                            content_parts.append(f"(sex: {row[column]}, ")
                        else:
                            content_parts.append(f"{column}: {row[column]}")
                # add a closing bracket
                content_parts += "),"
                content = "".join(content_parts)
                hits.append({'content': content})

    # Creating the final JSON object
    result = {
        'query': query,
        'hits': hits
    }
    return result


def fair_instruction(id_):
    if id_ == 2:
        return "."
    elif id_ == 4:
        return "in representation"
    elif id_ == 6:
        return "in exposure"
    elif id_ == 8:
        return f"with respect to {prot}"
    elif id_ == 10:
        return f"in representation with respect to {prot}"
    elif id_ == 12:
        return f"in exposure with respect to {prot}"
    elif id_ == 14:
        return f"with respect to {prot}, where {protected_group} is the disadvantaged group"
    elif id_ == 16:
        return f"in representation with respect to {prot}, where {protected_group} is the disadvantaged group"
    elif id_ == 18:
        return f"in exposure with respect to {prot}, where {protected_group} is the disadvantaged group"


def prepare_fair_rerank_template(shott, size=50, prompt_id=1):
    """
    "For example, .... Re-rank the following list to incorporate fairness."
    :param shott:
    :param size:
    :param prompt_id:
    :return:
    """
    instruct_template = ""
    # base_template = ("You are the admissions officer for a Law school and your goal is to rank applicants by "
    #                 "predicted success in the program, while ensuring fairness. ")
    base_template = ""
    # re_rank_template = "You are given a list that is already ranked based on scores"
    re_rank_template = "\n\nTASK\n" + "Re-rank the following list to incorporate fairness\n"
    if experiment_name != ' ':  # experiment_name == 'LAW':
        instruct_template = ""
        if prompt_id == 1 or prompt_id == 3 or prompt_id == 5:  # 3 is a tabulations prompt #5 is for LTR examples
            base_template = base_template

        elif prompt_id == 2:
            base_template = re_rank_template

        elif prompt_id == 4:
            base_template = re_rank_template + f" {fair_instruction(prompt_id)}"

        elif prompt_id == 6:
            base_template = re_rank_template + f" {fair_instruction(prompt_id)}"

        elif prompt_id == 8:  # fairness prompt
            base_template = re_rank_template + f" {fair_instruction(prompt_id)}"

        elif prompt_id == 10:
            base_template = re_rank_template + f" {fair_instruction(prompt_id)}"

        elif prompt_id == 12:
            base_template = re_rank_template + f" {fair_instruction(prompt_id)}"

        elif prompt_id == 14:
            base_template = re_rank_template + f"{fair_instruction(prompt_id)}"
        elif prompt_id == 16:
            base_template = re_rank_template + f" {fair_instruction(prompt_id)}"

        elif prompt_id == 18:
            base_template = re_rank_template + f" {fair_instruction(prompt_id)}"

        else:
            base_template = generate_base(prompt_id)
            instruct_template = " Rank the following applicants:"
        base_template += "\nDATA\n\nINPUT LIST: "

    shot_templates = []

    shot_template = instruct_template
    if prompt_id == 3:
        return shot_template + instruct_template
    if shott == 0:
        shot_template += base_template
        print(shot_template)
    else:
        for i in range(1, shott + 1):
            # example_template = (pick_conjunction(i - 1) + f" example, given this list of {item_type}s: ")

            example_template = "\n\nEXAMPLE " + str(i) + "\n\nINPUT LIST: "

            if prompt_id % 2 == 0:
                shot_sample = pd.read_csv(
                    f"../Datasets/{experiment_name}/Train/Fair_Reranking/shot_{i}/ground_truth_rank_size_{size}_shot_{i}.csv")

            else:
                shot_sample = pd.read_csv(
                    '../Datasets/' + experiment_name + '/Train/Scored/ranked_data_rank_size_' + str(
                        size) + '_shot_' + str(
                        i) + '.csv')
            # reset the index
            shot_sample.reset_index(drop=True, inplace=True)
            # shuffle the data
            # shuffled_sample = shot_sample.sample(frac=1).reset_index(drop=True)
            # Create examples list from the shot_sample
            examples = [row_converter(row, post_process=True) for index, row in shot_sample.iterrows()]

            formatted_examples = [f"[{i + 1}] {item}" for i, item in enumerate(examples)]

            # enumerated_examples = []
            # for k, item in enumerate(examples):
            #     enumerated_examples.append(str(k + 1) + ". " + str(item) + " ")
            # shot_template += ' '.join(enumerated_examples) + '. '

            # example_template += ' '.join(formatted_examples) + '. The fairly re-reranked list is: '
            example_template += ' '.join(formatted_examples) + "\n\n" + "OUTPUT LIST: "
            # Get the row numbers of shuffled_sample based on the order in shot_sample

            fair_ranked_sample = pd.read_csv(
                f"../Datasets/{experiment_name}/Train/Fair_Reranking/shot_{i}/ranked_data_rank_size_{size}_shot_{i}.csv")

            # create a mapping of items to their original indices in shot_example
            items_to_index = shot_sample['doc_id'].reset_index().set_index('doc_id')['index']

            # use the mapping to get the indices of items in fair_examples based on shot_example
            reordered_indices = fair_ranked_sample['doc_id'].map(items_to_index)

            # sort the indices to determine the order of the items in fair_examples
            # sorted_indices = reordered_indices.sort_values()

            # row_numbers = shot_sample['doc_id'].apply(lambda x: shot_sample[shot_sample['doc_id'] == x].index[0])
            # Convert the row numbers to the desired format
            # formatted_output = " > ".join(map(str, reordered_indices.values+1))
            formatted_output = " > ".join(map(lambda x: f"[{x}]", reordered_indices.values + 1))
            # formatted_output = ' > '.join([f"[{num + 1}]" for num in row_numbers])
            example_template += formatted_output
            shot_template = shot_template + example_template

        shot_template += base_template.replace("['", "").replace("']", "")
    shot_templates.append(shot_template)

    return shot_templates[0].replace("['", "").replace("']", "") + ""


def pick_conjunction(i):
    conjunction_options = [" For", " Another", " Yet another", " And another"]
    if i == 0:
        return conjunction_options[0]
    elif i == 1:
        return conjunction_options[1]
    elif i == 2:
        return conjunction_options[2]
    else:
        return conjunction_options[3]


def row_converter(row, post_process=False):
    if experiment_name == 'LAW':
        if post_process:
            return "Student ID: " + str(row['doc_id']) + " (" + "sex: " + str(row['Gender']) + ", score: " + str(
                row['predictions']) + ")"
        else:
            return "Student ID: " + str(row['doc_id']) + " (" + "sex: " + str(row['Gender']) + ", UGPA: " + str(
                row['UGPA']) + (
                ",LSAT: ") + str(row['LSAT']) + ")"

    # elif experiment_name == 'bostonmarathon':  # experiment_name == 'dummy':
    #     return "Unique ID: " + str(row['doc_id']) + " (" + str(row['Gender']) + ", Age: " + str(row['Age']) + (
    #         ",5K: ") + " (" + str(row['5K']) + ", 10K: " + str(row['10K']) + ", 15K: " + str(
    #         row['15K']) + ", 20K: " + str(row['20K']) + ", Half: " + str(row['Half']) + (
    #         ",30K: ") + str(row['30K']) + ")"
    else:
        return create_content(row)


def create_content(row):
    # content_parts = [f"Unique ID: {row['doc_id']}"]
    content_parts = []
    for column in row.index:
        content_parts = [f"ID: {row['doc_id']} "]
        #if column != 'doc_id':
        #if 'Gender' in column:
        content_parts.append(f"(sex: {row[column]}, ")
        #if 'score' in column:
        if experiment_name == 'NBAWNBA':
            content_parts.append(f"Career Points: {row['GT_score']}")
        else:
            content_parts.append(f"score: {row['GT_score']}")
    content_parts += "),"  # add a closing bracket
    content = "".join(content_parts)
    return content
    # for index, row in df.iterrows():
    #     content_parts = [f"ID: {row['doc_id']} "]
    #     for column in df.columns:
    #         if column != 'doc_id':
    #             if column == 'Gender':
    #                 content_parts.append(f"(sex: {row[column]}, ")
    #             else:
    #                 content_parts.append(f"{column}: {row[column]}")
    #     # add a closing bracket
    #     content_parts += "),"
    #     content = "".join(content_parts)


def generate_base(prompt_id):  # hardcoded for now. Considering having multiple queries
    """
    this should only give us the base template
    :param prompt_id:
    :return:
    """
    if experiment_name == 'LAW':

        return ("You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to "
                "predict their success in the program. The school wants to rank the applicants using their UGPA score "
                "and LSAT scores and gender. Rank the following applicants in a predicted order of success in the "
                "program:")
    else:
        return " "


def get_tokens_and_count(string, tokenizer):
    tokens = tokenizer.encode(string)
    return len(tokens)


############################################################################################################
# Run experiments
############################################################################################################

# for shot in shots:
#     RankWithGPT("gpt-3.5-turbo", shot, 1)

# RankWithGPT("gpt-3.5-turbo", 1, 1)

# rank_with_llama("meta-llama/Meta-Llama-3-8B-Instruct", 0, 5)
# RankWithLLM("gemini-1.5-flash", 4, 20, prompt_id=6, ltr_ranked=False, post_process=True)
# RankWithLLM("gemini-1.5-pro", 4, 20, prompt_id=6, ltr_ranked=False, post_process=True)

# RankWithLLM("gpt-3.5-turbo", 10, 1, 100, prompt_id=1)
ltr_rank_options = [False]
for option in ltr_rank_options:
    print('option = ', option)
    # prmpt_ids = [6, 8, 10, 12, 14, 16]
    # prmpt_ids = [24]
    # for prmpt_id in prmpt_ids:
    for prmpt_id in range(2, 19, 2):
        for shot in shots:
            #RankWithLLM("gemini-1.5-flash", shot, 20, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)
            #RankWithLLM("gemini-1.5-pro", shot, 20, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)
            RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", shot, 20, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)
            #RankWithLLM("meta-llama/Llama-3.3-70B-Instruct", shot, 20, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)
#################################################################################################

# ltr_rank_options = [True, False]
# for option in ltr_rank_options:
#     for shot in shots:
#         # RankWithLLM("gemini-1.5-flash", shot, 20, prompt_id=16, ltr_ranked=option)
#         # RankWithLLM("gemini-1.5-pro", shot, 20, prompt_id=16, ltr_ranked=option)
#         RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", shot, 20, prompt_id=16, ltr_ranked=option)

# prmpt_ids = [1]
# for prmpt_id in prmpt_ids:
#     for shot in shots:
#         print(os.getcwd())
#         RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", shot, 5, 20, prompt_id=prmpt_id)
#         RankWithLLM("meta-llama/Meta-Llama-3.1-8B-Instruct", shot, 5, 20, prompt_id=prmpt_id)

# for shot in shots:
#     RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", shot, 1, 50, prompt_id=1)

# RankWithLLM("gpt-3.5-turbo", 0, 5, 5)
# RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", 10, 5, 20, prompt_id=1)
# RankWithLLM("meta-llama/Meta-Llama-3.1-8B-Instruct", 10, 5, 20, prompt_id=1)
# RankWithLLM("gemini-1.5-flash", 0, 5, 20, prompt_id=1)

# sliding windows
# rank_entire_test(window_size=50, step=10, model_name='gpt-4o-mini', prompt_id=1, number_of_shots=0)

# rank_with_gemini(GEMINI_API_KEY)

end = time.time()

print("time taken = ", end - start)
