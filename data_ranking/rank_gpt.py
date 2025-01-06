# functions contained in the file are adapted from sunnweiwei's code. https://github.com/sunnweiwei/RankGPT
import copy
import time
import json
import os

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


class OpenaiClient:
    def __init__(self, keys=None, start_id=None, proxy=None):
        from openai import OpenAI
        import openai
        if isinstance(keys, str):
            keys = [keys]
        if keys is None:
            raise "Please provide OpenAI Key."

        self.key = keys
        self.key_id = start_id or 0
        self.key_id = self.key_id % len(self.key)
        self.api_key = self.key[self.key_id % len(self.key)]
        self.client = OpenAI(api_key=self.api_key)

    def chat(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.chat.completions.create(*args, **kwargs, timeout=30)
                break
            except Exception as e:
                print(str(e))
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].message.content
        print('completion', completion)
        return completion

    def text(self, *args, return_text=False, reduce_length=False, **kwargs):
        while True:
            try:
                completion = self.client.completions.create(
                    *args, **kwargs
                )
                break
            except Exception as e:
                print(e)
                if "This model's maximum context length is" in str(e):
                    print('reduce_length')
                    return 'ERROR::reduce_length'
                time.sleep(0.1)
        if return_text:
            completion = completion.choices[0].text
        return completion


def get_prefix_prompt(query, hits, rank_end, num, Item, model_type, prompt_id):
    # post_prompt is now included in this function
    from llmrank import fair_instruction as fi
    rank = 0
    rank_start = 0
    combined_content = ""  # Initialize an empty string to store all content
    # Process hits
    for hit in hits[rank_start: rank_end]:
        rank += 1
        content = hit.get('content', '').strip()  # Ensure content is a valid string
        if content:
            combined_content += f"[{rank}] {content} "  # Append each hit's content to the combined_content string
    if 'llama' in model_type:
        combined_content += "\n\nOUTPUT FORMAT\n" + (
            f"The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
            f"results, do not say any word or explain.")
    else:
        combined_content += "\n\nOUTPUT FORMAT\n" + (f"The output format should be [] > [], e.g., [1] > [2]. Only return the ranking "
                             f"results, do not say any word or explain.")
    instruct_prompt = f"{fi(prompt_id)}{query} {combined_content.strip()}"
    if experiment_name == 'LAW':
        return [{'role': 'system',
                 'content': f"You are the admissions officer for a Law school and your goal is to rank {Item}s by "
                            f"predicted success in the program, while ensuring fairness."},
                {'role': 'user',
                 'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
                            f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
                            f" success scores, and you are to re-rank to incorporate fairness{instruct_prompt}"}]
    elif experiment_name == 'bostonmarathon':
        return [{'role': 'system',
                 'content': f"You are the ranking officer for a marathon race and your goal is to rank {Item}s by "
                            f"predicted success in the race, while ensuring fairness."},
                {'role': 'user',
                 'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
                            f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
                            f" official time scores, and you are to re-rank to incorporate fairness{instruct_prompt}"}]
    elif experiment_name == 'COMPASSEX':
        return [{'role': 'system',
                 'content': f"You are the parole officer for criminal offenders and your goal is to rank {Item}s by "
                            f"predicted likelihood of re-offending, while ensuring fairness."},
                {'role': 'user',
                 'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
                            f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
                            f" recidivism scores, and you are to re-rank to incorporate fairness{instruct_prompt}"}]
    elif experiment_name == 'NBAWNBA':
        return [{'role': 'system',
                 'content': f"You are the ranking officer for an athlete ranking site and your goal is to rank basketball {Item}s by "
                            f"overall career success, while ensuring fairness."},
                {'role': 'user',
                 'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
                            f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
                            f" career points, and you are to re-rank to incorporate fairness{instruct_prompt}"}]


# def get_prefix_prompt(query, hits, rank_end, num, Item, model_type, prompt_id):
#     # post_prompt is now included in this function
#     from llmrank import fair_instruction as fi
#     rank = 0
#     rank_start = 0
#     combined_content = ""  # Initialize an empty string to store all content
#     # Process hits
#     for hit in hits[rank_start: rank_end]:
#         rank += 1
#         content = hit.get('content', '').strip()  # Ensure content is a valid string
#         if content:
#             combined_content += f"[{rank}] {content} "  # Append each hit's content to the combined_content string
#     if 'llama' in model_type:
#         combined_content += (f". The output format should be [] > [] > [] > ... , e.g., [1] > [2] > [20] > ... Only return the ranking "
#                              f"results, do not say any word or explain.")
#     else:
#         combined_content += (f". The output format should be [] > [], e.g., [1] > [2]. Only return the ranking "
#                          f"results, do not say any word or explain.")
#     if experiment_name == 'LAW':
#         return [{'role': 'system',
#                  'content': f"You are the admissions officer for a Law school and your goal is to rank {Item}s by "
#                             f"predicted success in the program, while ensuring fairness."},
#                 {'role': 'user',
#                  'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
#                             f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
#                             f" success scores, and you are to re-rank to incorporate fairness{fi(prompt_id)}{query} {combined_content.strip()}"}]
#     elif experiment_name == 'bostonmarathon':
#         return [{'role': 'system',
#                  'content': f"You are the ranking officer for a marathon race and your goal is to rank {Item}s by "
#                             f"predicted success in the race, while ensuring fairness."},
#                 {'role': 'user',
#                  'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
#                             f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
#                             f" official time scores, and you are to re-rank to incorporate fairness{fi(prompt_id)}{query} {combined_content.strip()}"}]
#     elif experiment_name == 'COMPASSEX':
#         return [{'role': 'system',
#                  'content': f"You are the parole officer for criminal offenders and your goal is to rank {Item}s by "
#                             f"predicted likelihood of re-offending, while ensuring fairness."},
#                 {'role': 'user',
#                  'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
#                             f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
#                             f" recidivism scores, and you are to re-rank to incorporate fairness{fi(prompt_id)}{query} {combined_content.strip()}"}]
#     elif experiment_name == 'NBAWNBA':
#         return [{'role': 'system',
#                  'content': f"You are the ranking officer for an athlete ranking site and your goal is to rank basketball {Item}s by "
#                             f"overall career success, while ensuring fairness."},
#                 {'role': 'user',
#                  'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
#                             f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
#                             f" career points, and you are to re-rank to incorporate fairness{fi(prompt_id)}{query} {combined_content.strip()}"}]
#     # return [{'role': 'system',
#     # return [{'role': 'system',
#     #          'content': f"You are the admissions officer for a Law school and your goal is to rank {Item}s by "
#     #                     f"predicted success in the program, while ensuring fairness."},
#     #         {'role': 'user',
#     #          'content': f"I will provide a list of {num} {Item}s, each described by a sequential"
#     #                     f" index (e.g., [1]), an ID, sex and a score. The list is already ranked by"
#     #                     f" success scores, and you are to rank to incorporate fairness {fi(prompt_id)}{query} {combined_content.strip()}"}]
#     # ,{'role': 'assistant', 'content': 'Okay, please provide the list.'}]


def sliding_windows(item=None, rank_start=0, rank_end=100, window_size=20, step=10, model_name='gpt-3.5-turbo',
                    api_key=None):
    item = copy.deepcopy(item)
    end_pos = rank_end
    start_pos = rank_end - window_size
    while start_pos >= rank_start:
        start_pos = max(start_pos, rank_start)
        item = permutation_pipeline(item, start_pos, end_pos, model_name=model_name, api_key=api_key)
        end_pos = end_pos - step
        start_pos = start_pos - step
    return item


def permutation_pipeline(item=None, rank_start=0, rank_end=100, model_name='gpt-3.5-turbo', api_key=None):
    messages = create_permutation_instruction(item=item, rank_start=rank_start, rank_end=rank_end)  # chan
    permutation = run_llm(messages, api_key=api_key, model_name=model_name)
    item = receive_permutation(item, permutation, rank_start=rank_start, rank_end=rank_end)
    return item


def create_permutation_instruction(df, item=None, rank_start=0, rank_end=50, item_type='applicant', prompt_id=1,
                                   model_type=None):
    # print('item is ', item)
    # check if item is a valid dictionary
    if not isinstance(item, dict):
        print('Invalid item')
        exit()
    query = item.get('query', '')  # Ensure query is a valid string
    hits = item.get('hits', [])  # Ensure hits is a valid list
    num = len(hits[rank_start: rank_end])

    messages = get_prefix_prompt(query, hits, rank_end, num, item_type, model_type,
                                 prompt_id)  # Get prefix prompt messages
    if model_type == 'gemini':
        messages = [message for message in messages]

    # Ensure prefix prompt messages are valid
    for message in messages:
        if 'content' not in message or not isinstance(message['content'], str):
            print('No content available')

    # Final check to ensure all messages have valid content
    valid_messages = [message for message in messages if
                      'content' in message and isinstance(message['content'], str) and message['content'].strip()]
    # Extract the content values and concatenate them into a single string
    if model_type == 'gemini':
        concatenated_content = ' '.join([message['content'].replace("'", '"')
                                         for i, message in enumerate(valid_messages)])

        return concatenated_content
    else:
        return valid_messages


def run_llm(messages, api_key=None, model_name="gpt-3.5-turbo"):
    Client = OpenaiClient
    agent = Client(api_key)
    response = agent.chat(model=model_name, messages=messages, temperature=0.1, return_text=True)
    return response


def get_post_prompt(query, num, Item):
    # return (f"Query: {query}. \nRank the {num} {Item}s based on their relevance to the "
    #         f"query. The {Item}s should be listed in descending order using identifiers. The most relevant "
    #         f"should be listed first. The output format should be [] > [], e.g., [1] > [2]. Only return the ranking "
    #         f"results, do not say any word or explain.")
    return (f". The output format should be [] > [], e.g., [1] > [2]. Only return "
            f"the ranking "
            f"results, do not say any word or explain.")


def clean_response(response: str):
    new_response = ''
    for c in response:
        if not c.isdigit():
            new_response += ' '
        else:
            new_response += c
    new_response = new_response.strip()
    return new_response


def remove_duplicate(response):
    new_response = []
    for c in response:
        if c not in new_response:
            new_response.append(c)
    return new_response


def receive_permutation(item, permutation, rank_start=0, rank_end=100):
    response = clean_response(permutation)
    response = [int(x) - 1 for x in response.split()]
    response = remove_duplicate(response)
    cut_range = copy.deepcopy(item['hits'][rank_start: rank_end])
    original_rank = [tt for tt in range(len(cut_range))]
    response = [ss for ss in response if ss in original_rank]
    response = response + [tt for tt in original_rank if tt not in response]
    for j, x in enumerate(response):
        item['hits'][j + rank_start] = copy.deepcopy(cut_range[x])
        if 'rank' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['rank'] = cut_range[j]['rank']
        if 'score' in item['hits'][j + rank_start]:
            item['hits'][j + rank_start]['score'] = cut_range[j]['score']
    return item
