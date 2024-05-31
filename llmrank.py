import pandas as pd
import numpy as np
import transformers
from transformers import AutoTokenizer, pipeline
import torch
import hf_login

# read data
gt_data = pd.read_csv('LAW_data_for_LLM.csv')

# Random selection of data
seed = 123

data_to_sample = gt_data

sample_size = 4  # min(10, len(trained_data))  # Adjust the sample size as needed

# prompt_sample = data_to_sample.sample(n=sample_size, replace=True, random_state=seed)

# prompt_sample = prompt_sample.sort_values(by="LSAT", ascending=False)

def serialize(df, row_index):
    """
    Takes a row of the data and serializes it as a string
    returns: string
    """
    result = 'The student ID is ' + str(df.iloc[row_index]['unique_id']) + '. The gender is ' + str(
        df.iloc[row_index]['Gender']) + ". The UGPA score is " + str(
        df.iloc[row_index]['UGPA']) + ". The LSAT score is " + str(df.iloc[row_index]['LSAT']) + "."
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
        listed_gt_data.loc[index, 0] = serialize(df, index)
    return listed_gt_data.values.tolist()


def get_llama2_response(prompt: str) -> None:
    sequences = llama2_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    print("Chatbot:", "Llama response:", sequences[0]['generated_text'])


def SampleData():
    prompts = []
    # Loop until the parent DataFrame is exhausted
    number_of_samples = 200

    while number_of_samples > 0:
        current_sample = data_to_sample.sample(n=sample_size, replace=True, random_state=seed)

        # serialize sample
        current_sample = serialize_to_list(current_sample)

        prompts.append(current_sample)

        number_of_samples = number_of_samples - 1


SampleData()
hf_login.login_to_hugging_face()

prompt_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of applicants "
                   "based on their undergrad GPA and some first LSAT scores. Rank the following applicants:")

prompt = prompt_template + str(serialize_to_list(prompt_sample))

model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model, token=True)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available.")
else:
    device = torch.device("cpu")
    print("CUDA is not available, exiting")
    exit()

llama2_pipeline = pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device=device,
)

get_llama2_response(prompt)
