import json
import os
import time

import pandas as pd

start = time.time()

with open('settings.json', 'r') as f:
    settings = json.load(f)

sample_size = settings["GENERAL_SETTINGS"]["sample_size"]

experiment_name = "LAW"

# read data
test_data = pd.read_csv('./Datasets/' + experiment_name + '/LAW_test_data_for_LLM.csv')

seed = 123


sample_folder = './Datasets/' + experiment_name + '/Splits/size_' + str(sample_size) + '/'
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder)


def SampleData():
    samples = []
    # Loop until the parent DataFrame is exhausted
    number_of_samples = 50

    while number_of_samples > 0:
        current_sample = test_data.sample(n=int(sample_size), replace=True, random_state=number_of_samples, axis=0)
        current_sample.reset_index(drop=True, inplace=True)
        samples.append(current_sample)
        current_sample.to_csv(sample_folder + '/test_' + str(number_of_samples) + '.csv', index=False)
        number_of_samples = number_of_samples - 1
    return samples


SampleData()

end = time.time()

print("time taken = ", end - start)
