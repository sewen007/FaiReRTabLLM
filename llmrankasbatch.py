import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import pipeline

from hf_login import CheckLogin
from llmrank import serialize_to_list, llama2_pipeline, tokenizer, SampleData

start = time.time()
# read data
test_data = pd.read_csv('LAW_test_data.csv')

# Random selection of data
seed = 123

# data_to_sample = gt_data

sample_size = 4  # min(10, len(trained_data))  # Adjust the sample size as needed

model = "meta-llama/Llama-2-7b-chat-hf"

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


def get_llama2_response(prompts: list) -> list:
    sequences = llama2_pipeline(
        prompts,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )
    return [seq[0]['generated_text'] for seq in sequences]


def RankWithLLM(shot=0):
    CheckLogin()

    sample_list = SampleData()

    if shot == 0:
        prompt_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                           "applicants to predict their success in the program. The school wants to rank the applicants"
                           "using their UGPA score and LSAT scores. Rank these applicants:")
    elif shot == 1:
        prompt_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                           "applicants to predict their success in the program. The school wants to rank the applicants"
                           "using their UGPA score and LSAT scores. An example of ranked applicants is: 1. Student "
                           "ID: 18642 (female, UGPA: 3.4, LSAT: 48)\n 2. Student ID: 4939 (male, UGPA: 2.8, LSAT: 33)\n "
                           "3. Student ID: 9105 (male, UGPA: 3.1, LSAT: 41)\n 4. Student ID: 9046 (Male, UGPA: 4, "
                           "LSAT: 34) .Rank these applicants:")
    elif shot == 2:
        prompt_template = ("You are the admissions ranking system for a Law school; their goal is to rank a list of "
                           "applicants to predict their success in the program. The school wants to rank the applicants"
                           "using their UGPA score and LSAT scores. An example of ranked applicants is: 1. Student "
                           "ID: 18642 (female, UGPA: 3.4, LSAT: 48)\n 2. Student ID: 4939 (male, UGPA: 2.8, LSAT: 33)\n "
                           "3. Student ID: 9105 (male, UGPA: 3.1, LSAT: 41)\n 4. Student ID: 9046 (Male, UGPA: 4, "
                           "LSAT: 34) . Another example of ranked applicants is: 1. Student "
                           "ID: 3119 (male, UGPA: 3.4, LSAT: 37)\n 2. Student ID: 2778 (male, UGPA: 3.3, LSAT: 43)\n "
                           "3. Student ID: 11151 (male, UGPA: 3.6, LSAT: 39)\n 4. Student ID: 10395 (male, UGPA: 3.9, "
                           "LSAT: 42) .Rank these applicants:")

    """ DIRECTORY MANAGEMENT """
    graph_path = Path("../LLMFairRank/Llama2OutputBatched/LAW/shot_" + str(shot) + "/")

    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    batch_size = 8
    count = 0
    batched_prompts = []

    for sample in sample_list:
        serialized_sample = serialize_to_list(sample)
        prompt = prompt_template + str(serialized_sample)
        batched_prompts.append(prompt)

        if len(batched_prompts) == batch_size:
            responses = get_llama2_response(batched_prompts)
            for i, response in enumerate(responses):
                count += 1
                save_path = str(graph_path) + '/output_' + str(count) + '.txt'
                sample = sample_list[count - 1]
                sample = sample.sort_values(by='ZFYA', ascending=False)
                with open(save_path, 'w') as file:
                    file.write(response)
                    file.write("\nGroundtruth: \n")
                    file.write(str(sample))
            batched_prompts = []

    if batched_prompts:
        responses = get_llama2_response(batched_prompts)
        for i, response in enumerate(responses):
            count += 1
            save_path = str(graph_path) + '/output_' + str(count) + '.txt'
            sample = sample_list[count - 1]
            sample = sample.sort_values(by='ZFYA', ascending=False)
            with open(save_path, 'w') as file:
                file.write(response)
                file.write("\nGroundtruth: \n")
                file.write(str(sample))


RankWithLLM()
RankWithLLM(1)
RankWithLLM(2)
