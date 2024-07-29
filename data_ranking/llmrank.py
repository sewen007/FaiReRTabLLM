# from data_analysis.calculate_metrics import kendall_tau
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from hf_login import CheckLogin
from rank_gpt import create_permutation_instruction, run_llm, receive_permutation
import json
import os
import re
import time
from pathlib import Path
import pandas as pd

os.environ['TRANSFORMERS_CACHE'] = '/scratch/shared/models/'

start = time.time()

# variables for GPT
api_key = "sk-proj-BKhsxjRDcvt4ZBzlM63pT3BlbkFJlmacaHE8VVdLZwGhmV1P"

with open('../settings.json', 'r') as f:
    settings = json.load(f)

sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
score_column = settings["READ_FILE_SETTINGS"]["SCORE_COL"]

delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))


def RankWithLLM(model_name, shot_number=1, runs=5, size=50, prompt_id=1):
    """
    This function ranks the data using a language model
    :param prompt_id: prompt 0 is the neutral prompt
    :param model_name: LLM model name
    :param shot_number: number of examples. Each example has the size of the rank
    :param runs: number of runs
    :param size: size of the rank
    :param neutral: determines if to use a neutral prompt
    :return:
    """
    if prompt_id == 0:
        neutral = True
    else:
        neutral = False
    results_dir = Path(
        f'../Datasets/{experiment_name}/Ranked/{model_name}/prompt_{prompt_id}/rank_size_{size}/shot_{shot_number}')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # gt_df = rank_with_llama(model_name, shot_number)[0]
    gt_df = pd.read_csv(f"../Datasets/{experiment_name}/Tests/rank_size_{size}.csv")
    gt_df = gt_df.sort_values(by=[score_column], ascending=False)
    gt_df.to_csv(os.path.join(results_dir, 'ground_truth.csv'), index=False)
    # print('gt_df = ', gt_df)
    for i in range(runs):
        if "llama" in model_name:
            ranked_df = rank_with_llama(model_name, shot_number, size, neutral, prompt_id)[1]
        else:  # "gpt" in model_name:
            ranked_df = rank_with_GPT(model_name, shot_number, size, neutral, prompt_id)[1]
        ranked_df.to_csv(os.path.join(results_dir, 'ranked_data_' + str(i + 1) + '.csv'), index=False)


def rank_with_llama(model_name, number_of_shots=0, size=5, neutral=False, prompt_id=1):
    CheckLogin()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, exiting")
        exit()
    #device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    df = pd.read_csv(f"../Datasets/{experiment_name}/Tests/rank_size_{size}.csv")

    item = create_items(df, number_of_shots, neutral, prompt_id)
    # (1) Create permutation generation instruction
    messages = create_permutation_instruction(item=item, rank_start=0, rank_end=size)
    template_prompt_pred = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    template_prompt_pred += '<|start_header_id|>prediction<|end_header_id|>\n\n'
    inputs_template_pred = tokenizer(template_prompt_pred, add_special_tokens=False, return_tensors='pt')
    inputs_template_pred = inputs_template_pred.to(device)

    # generate an output using the template prompt and print only the model generated tokens
    outputs_template_pred = model.generate(**inputs_template_pred, pad_token_id=tokenizer.eos_token_id,
                                           max_new_tokens=25, return_dict_in_generate=True)
    generated_tokens_template_pred = outputs_template_pred.sequences[:,
                                     inputs_template_pred.input_ids.shape[1]:]  # for decoder only models
    generated_text_template_pred = tokenizer.decode(generated_tokens_template_pred[0], skip_special_tokens=True)

    # Use permutation to re-rank the passage
    new_item = receive_permutation(item, generated_text_template_pred, rank_start=0, rank_end=size)
    # Extract information and store in a list of dictionaries
    gt_df, merged_df = extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots, size)

    return gt_df, merged_df


def rank_with_GPT(model_name, number_of_shots=0, size=5, neutral=False, prompt_id=1):
    # df = test_df.sample(n=rank_size, random_state=1)
    df = pd.read_csv(f"../Datasets/{experiment_name}/Tests/rank_size_{size}.csv")
    item = create_items(df, number_of_shots, neutral, prompt_id)
    # (1) Create permutation generation instruction
    messages = create_permutation_instruction(item=item, rank_start=0, rank_end=size)
    print('messages = ', messages)

    # (2) Get ChatGPT predicted permutation
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)

    # (3) Use permutation to re-rank the passage
    new_item = receive_permutation(item, permutation, rank_start=0, rank_end=size)

    gt_df, merged_df = extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots, size)

    return gt_df, merged_df


def extract_and_save_permutation(df, new_item, model_name, prompt_id, number_of_shots=0, size=50):
    # Extract information and store in a list of dictionaries
    extracted_ranked_data = [extract_info(item['content']) for item in new_item['hits']]

    # Convert list of tuples to DataFrame
    ranked_df = pd.DataFrame(extracted_ranked_data)

    # order gt_df by score_column
    gt_df = df.sort_values(by=[score_column], ascending=False)
    if experiment_name == 'LAW':
        # rename unique_id to Student ID in gt_df
        gt_df.rename(columns={'doc_id': 'Student ID'}, inplace=True)

    if experiment_name == 'LAW':
        #
        # get gt scores from df
        merged_df = pd.merge(ranked_df, gt_df[['Student ID', score_column]], on='Student ID', how='left')
    else:
        merged_df = pd.merge(ranked_df, gt_df[['doc_id', score_column]], on='doc_id', how='left')

    # create directory for results based on n and model name
    results_dir = Path(
        f'../Datasets/{experiment_name}/Ranked/{model_name}/prompt_{prompt_id}/rank_size_{size}/shot_{number_of_shots}')

    # Create the directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # result = kendall_tau(gt_df['Student ID'].tolist(), ranked_df['Student ID'].tolist())
    return gt_df, merged_df


# Define a function to extract information from the content string
def extract_info(content):
    if experiment_name == 'LAW':
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
    else:  # experiment_name == 'dummy':
        parts = content.split(',')
        doc_id = re.search(r'Unique ID: (\d+)', parts[0]).group(1)
        number = re.search(r'Number: ([\d.]+)', parts[1]).group(1)
        return {
            'doc_id': int(float(doc_id)),
            'Number': float(number)
        }


def create_items(df, number_of_shots=0, neutral=False, prompt_id=1):
    """creates items with format item = {
    'query': 'You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to '
             'predict their success in the program. The school wants to rank the applicants using their UGPA score '
             'and LSAT scores.  Without including explanations, rank the following applicants in a predicted order of '
             'success in the program.',
    'hits': [
        {'content': 'Student ID: 18739 (Gender: female, UGPA: 3.7, LSAT: 44.0)'},
        {'content': 'Student ID: 8742 (Gender: male, UGPA: 3.2, LSAT: 20.0)'},
        {'content': 'Student ID: 17967 (Gender: male, UGPA: 3.2, LSAT: 34.5)'},
        {'content': 'Student ID: 13964 (Gender: male, UGPA: 3.7, LSAT: 32.0)'}
    ]
}"""
    print('neutral = ', neutral)
    print('number_of_shots = ', number_of_shots)
    n_rank_size = len(df)
    s_shots = number_of_shots
    if s_shots == 0:
        if experiment_name == 'LAW':
            # Hardcoded query
            # if neutral is True, the query is neutral
            if neutral:  # prompt_id == 0:
                query = (
                    'You are the admissions ranking system for a Law school; their goal is to rank a list of '
                    'applicants to predict their success in the program. Rank the following applicants in a predicted '
                    'order of success in the program.')
            else:
                if prompt_id == 1:
                    query = (
                        'You are the admissions ranking system for a Law school; their goal is to rank a list of '
                        'applicants to predict their success in the program. The school wants to rank the applicants '
                        'using their UGPA score and LSAT scores. Rank the following applicants in a predicted order of '
                        'success in the program:')
                else:  # query is not 0 or 1
                    query = generate_base(prompt_id)
                    query = query + ' Rank the following applicants in a predicted order of success in the program:'

        else:  # ranking dummy data
            query = 'You are a ranking system for a list of numbers. Rank the following numbers in descending order:'
    else:
        # note that the shot examples come from the training data
        query = prepare_template(df, s_shots, n_rank_size, neutral, prompt_id)

    # Creating the hits list from the DataFrame
    hits = []
    if experiment_name == 'LAW':
        for index, row in df.iterrows():
            content = f"Student ID: {row['doc_id']} (Gender: {row['Gender']}, UGPA: {row['UGPA']}, LSAT: {row['LSAT']})"
            hits.append({'content': content})
    else:
        for index, row in df.iterrows():
            content_parts = [f"Unique ID: {row['doc_id']}"]
            for column in df.columns:
                if column != 'doc_id':
                    content_parts.append(f"{column}: {row[column]}")
            content = ", ".join(content_parts)
            hits.append({'content': content})

    # Creating the final JSON object
    result = {
        'query': query,
        'hits': hits
    }
    return result


def prepare_template(df, shott, size=50, neutral=False, prompt_id=1):
    instruct_template = "Rank the following in descending order:"
    if experiment_name != 'LAW':  # experiment_name == 'dummy':
        base_template = "You are a ranking system for a list of numbers. "
        instruct_template = "Rank the following numbers in descending order:"
    else:  # experiment_name == 'LAW':
        if neutral:  # prompt_id == 0:
            base_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                             "applicants to predict their success in the program. Rank the following applicants in a "
                             "predicted order of success in the program. ")
        elif prompt_id == 1:
            base_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                             "applicants to predict their success in the program. The school wants to rank the "
                             "applicants using their UGPA score and LSAT scores. ")
            instruct_template = " Rank the following applicants in a predicted order of success in the program.:"
        else:
            base_template = generate_base(prompt_id)
            instruct_template = " Rank the following applicants in a predicted order of success in the program.:"

    shot_templates = []

    shot_template = base_template
    if shott == 0:
        shot_template += instruct_template
    else:
        for i in range(shott):
            if experiment_name == 'LAW':
                shot_template += (pick_conjunction(i) + " example of ranked applicants in order of success in the "
                                                        "program is: ")
            else:  # experiment_name == 'dummy':
                shot_template += (pick_conjunction(i) + " example of ranked numbers in descending order is: ")

            shot_sample = pd.read_csv(
                '../Datasets/' + experiment_name + '/Train/' + 'rank_size_' + str(size) + '_shot_' + str(
                    i + 1) + '.csv')
            # Create examples list from the shot_sample
            examples = [row_converter(row) for index, row in shot_sample.iterrows()]

            enumerated_examples = []
            for k, item in enumerate(examples):
                enumerated_examples.append(str(k + 1) + ". " + str(item) + " ")
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


def row_converter(row):
    if experiment_name == 'LAW':
        return "Student ID: " + str(row['doc_id']) + " (" + str(row['Gender']) + ", UGPA: " + str(row['UGPA']) + (
            ",LSAT: ") + str(row['LSAT']) + ")"
    else:
        create_content(row)


def create_content(row):
    content_parts = [f"Unique ID: {row['doc_id']}"]
    for column in row.index:
        if column != 'doc_id':
            content_parts.append(f"{column}: {row[column]}")
    content = ", ".join(content_parts)
    return {'content': content}


def generate_base(prompt_id):  # hardcoded for now. Considering having multiple queries
    """
    this should only give us the base template
    :param prompt_id:
    :return:
    """
    return ("You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to "
            "predict their success in the program. The school wants to rank the applicants using their UGPA score "
            "and LSAT scores and gender. Rank the following applicants in a predicted order of success in the "
            "program:")


# for shot in shots:
#     RankWithGPT("gpt-3.5-turbo", shot, 1)

# RankWithGPT("gpt-3.5-turbo", 1, 1)

# rank_with_llama("meta-llama/Meta-Llama-3-8B-Instruct", 0, 5)
# prmpt_ids = [0, 1]
# for prmpt_id in prmpt_ids:
#     for shot in shots:
#         RankWithLLM("gpt-3.5-turbo", shot, 5, 50, prompt_id=prmpt_id)
prmpt_ids = [0, 1]
for prmpt_id in prmpt_ids:
    for shot in shots:
        RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", shot, 5, 50, prompt_id=prmpt_id)


# for shot in shots:
#     RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", shot, 5, 50)

# RankWithLLM("gpt-3.5-turbo", 0, 5, 5)

end = time.time()

print("time taken = ", end - start)
