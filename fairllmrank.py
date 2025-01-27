import time

from LLMFairRank import *

start_time = time.time()


with open('./settings.json', 'r') as f:
    settings = json.load(f)

experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


test_set = f"./Datasets/{experiment_name}/{experiment_name}_test_data.csv"
full_size = len(pd.read_csv(test_set))


# 1. Clean the data
# Clean()
#
# 2. Prepare data
# Prep()

# 3. Rank the data with LLMs
ltr_rank_options = [False]
for option in ltr_rank_options:
    print('option = ', option)
    for prmpt_id in range(2, 19, 2):
        # RankWithLLM("gemini-1.5-pro", 0, full_size, prompt_id=prmpt_id, ltr_ranked=option, post_process=True, full_test=True)
        # RankWithLLM("gemini-1.5-flash", 0, full_size, prompt_id=prmpt_id, ltr_ranked=option, post_process=True, full_test=True)
        RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", 0, full_size, prompt_id=prmpt_id, ltr_ranked=option, post_process=True, full_test=True)
        # for shot in shots:
        #     RankWithLLM("gemini-1.5-flash", shot, 20, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)
        #     RankWithLLM("gemini-1.5-pro", shot, 20, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)
        #     RankWithLLM("meta-llama/Meta-Llama-3-8B-Instruct", shot, 20, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)

# 4. Calculate metric results
# CalculateResultMetrics('meta-llama/Meta-Llama-3-8B-Instruct')
#
# # 5. Collate different results
# Collate('meta-llama/Meta-Llama-3-8B-Instruct', prompt_remove=['prompt_1'])

# 6. Plot the results
# Plot()

end = time.time()

print("time taken = ", end - start)
