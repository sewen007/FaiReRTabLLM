import time


from LLMFairRank import *

start_time = time.time()

with open('./settings.json', 'r') as f:
    settings = json.load(f)

experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
sizes = settings["GENERAL_SETTINGS"]["rank_sizes"]
# 1. Clean the data
#Clean()
# exit()
#
#
# # 2. Prepare data
test_set = f"./Datasets/{experiment_name}/{experiment_name}_test_data.csv"
full_size = len(pd.read_csv(test_set))
full = False
meta_exp = 'meta-llama/Meta-Llama-3-8B-Instruct'
meta_app = 'Meta-Llama-3-8B-Instruct'

#3. Rank the data with LLMs
for size in sizes:
    # RankWithLLM_Llama("meta-llama/Meta-Llama-3-8B-Instruct", size=size, post_process=False)
    # Prep(size=size)
    #
    #
    # ltr_rank_options = [False]
    # for option in ltr_rank_options:
    #     print('option = ', option)
    #     for prmpt_id in range(2, 19, 2):
    #         for shot in shots:
    #             RankWithLLM("gemini-1.5-pro", shot, size, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)
    #             RankWithLLM("gemini-1.5-flash", shot, size, prompt_id=prmpt_id, ltr_ranked=option, post_process=True)

#4. Calculate metric results and plot results

    # CalculateResultMetrics(meta_exp, size=size)
    # # # # # # # # # # #
    # # # # # # # # # # # 5. Collate different results
    # Collate(meta_exp, prompt_remove=['prompt_1'])
    # # #
    # # # 6. Plot the results
    Plot(size=size, meta_app=meta_app)

end = time.time()

print("time taken = ", end - start)
