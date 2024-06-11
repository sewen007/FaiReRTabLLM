import time

import pandas as pd

start = time.time()

experiment_name = "LAW"

# read data
test_data = pd.read_csv('./Datasets/' + experiment_name + '/LAW_test_data.csv')

seed = 123

sample_size = 4


def SampleData():
    samples = []
    # Loop until the parent DataFrame is exhausted
    number_of_samples = 50

    while number_of_samples > 0:
        current_sample = test_data.sample(n=sample_size, replace=True, random_state=number_of_samples)
        current_sample.reset_index(drop=True, inplace=True)
        samples.append(current_sample)
        current_sample.to_csv('./Datasets/' + experiment_name + '/Splits/test_' + str(number_of_samples) + '.csv')
        number_of_samples = number_of_samples - 1
    return samples


SampleData()

end = time.time()

print("time taken = ", end - start)
