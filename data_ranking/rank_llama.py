import json
import os
import re
import time
from pathlib import Path
import pandas as pd
from hf_login import CheckLogin

os.environ['TRANSFORMERS_CACHE'] = '/scratch/shared/models/'
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
import torch


start = time.time()

with open('../settings.json', 'r') as f:
    settings = json.load(f)
CheckLogin()

sample_sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]

experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
delimiters = "_", "/", "\\", "."
regex_pattern = '|'.join(map(re.escape, delimiters))

# read data
test_data = pd.read_csv('../Datasets/' + experiment_name + '/LAW_test_data.csv')

seed = 123

# sample_size = 5

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    device = torch.device("cpu")
    print("CUDA is not available, exiting")
    exit()
device_map = "auto"

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_sig = 'Meta-Llama-3-8B-Instruct'

# for fine-tuning note this from docs: "Training the model in float16 is not recommended and is known to produce
# nan; as such, the model should be trained in bfloat16."
pipe = pipeline("text-generation", model=model_name, torch_dtype=torch.float16, device_map=device_map)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

prompt_addon = "```json"


def serialize(df, row_index):
    """
    Takes a row of the data and serializes it as a string
    returns: string
    """
    result = 'The student ID is ' + str(df.iloc[row_index]['unique_id']) + '. The gender is ' + str(
        df.iloc[row_index]['Gender']) + ". The UGPA score is " + str(
        df.iloc[row_index]['UGPA']) + ". The LSAT score is " + str(df.iloc[row_index]['LSAT']) + "."
    return result


def serialize_option_2(df, row_index):
    """
    Takes a row of the data and serializes it as a string
    returns: string
    """
    # result = Student ID: 9105 (Gender: male, UGPA: 3.1, LSAT: 41)
    result = 'Student ID: ' + str(df.iloc[row_index]['unique_id']) + ' (gender: ' + str(
        df.iloc[row_index]['Gender'] + ', UGPA: ' + str(df.iloc[row_index]['UGPA']) + ', LSAT: ' + str(
            df.iloc[row_index]['LSAT']) + ')')
    return result


def serialize_to_list(df):
    """
    Takes all rows of the data and serializes them as a list of strings
    :param df:
    :return: list of strings
    """
    listed_gt_data = pd.DataFrame()
    list_index = list(range(0, len(df)))
    for index in list_index:
        listed_gt_data.loc[index, 0] = serialize_option_2(df, index)
    return listed_gt_data.values.tolist()


stop_token_ids = [
    tokenizer.convert_tokens_to_ids(x) for x in [
        ['```', ']']
    ]
]

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]


# define custom stopping criteria object (# https://www.youtube.com/watch?v=-OXI2CZ_QgU)
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            # stop_ids = stop_ids.squeeze()  # Ensure stop_ids is 1-dimensional
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False


stopping_criteria = StoppingCriteriaList([StopOnTokens()])


def get_llama_response(prompt: str) -> None:
    sequences = pipe(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        stopping_criteria=stopping_criteria,
        eos_token_id=tokenizer.eos_token_id
    )
    return sequences[0]['generated_text']
    # print("Chatbot:", "Llama response:", sequences[0]['generated_text'])


def RankWithLlama(shot=0, rank_size=50):
    CheckLogin()

    # read each csv file from test directory and add to list
    folder_path = Path('../Datasets/' + experiment_name + '/Tests/')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    sample_list = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    # get template for prompt
    prompt_template = prepare_template(shot)

    """ DIRECTORY MANAGEMENT """
    result_path = Path(f'../Datasets-backup/{experiment_name}/Ranked/{model_sig}/rank_size_{rank_size}/shot_{shot}')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for sample in sample_list:
        sample_number = re.split(regex_pattern, sample)[1]
        # serialize sample
        sample_path = os.path.join(folder_path, sample)
        sample = pd.read_csv(sample_path)
        serialized_sample = serialize_to_list(sample)
        save_path = str(result_path) + '/output_' + str(sample_number) + '.txt'
        prompt = prompt_template + str(serialized_sample) + prompt_addon
        response = get_llama_response(prompt)
        # sort the sample by the ZFYA column
        sample = sample.sort_values(by='ZFYA', ascending=False)
        with open(save_path, 'w') as file:
            # Write data to the file
            file.write(str(response))
            file.write("\n\nGround truth: \n")
            file.write(str(sample))


def prepare_template(shots=0, rank_size=50):
    size = rank_size
    base_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                     "applicants to predict their success in the program. The school wants to rank the applicants "
                     "using their UGPA score and LSAT scores. ")
    instruct_template = (" Without including explanations, rank the following applicants in a predicted "
                         "order of success in the program. "
                         "Return your ranked results in the following json only {""Student ID"": ""the students ID"",""gender"": ""the "
                         "student's gender""}:")
    shot_templates = []
    # sampling_data = pd.read_csv('../Datasets/' + experiment_name + '/' + experiment_name + '_test_data_for_LLM.csv')
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
            shot_sample = pd.read_csv(
                '../Datasets/' + experiment_name + '/Train/' + 'rank_size_' + str(size) + '_shot_' + str(
                    i + 1) + '.csv')

            examples = [row_converter(row) for index, row in shot_sample.iterrows()]

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
            RankWithLlama(shot, rank_size=50)


# test = prepare_template(3)

#RankWithLlamaMultipleSizesAndShots()

RankWithLlama(shot=0, rank_size=50)

end = time.time()

print("time taken = ", end - start)
