# https://github.com/shiba24/learning2rank
import os
import pandas as pd
from learning2rank.rank import ListNet

sample_number = 50
Model = ListNet.ListNet()

experiment_name = 'LAW'
training_data = pd.read_csv('./Datasets/' + experiment_name + '/LAW_train_data.csv')

results_folder = './Datasets/' + experiment_name + '/Results/'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

sample_folder = './Datasets/' + experiment_name + '/Splits/'
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder)

X = training_data[:, :-1]
y = training_data[-1]

# train the model
Model.fit(X, y)

# test each sample and save the results
for i in range(sample_number):  # the number of samples
    sample = pd.read_csv(sample_folder + 'test_' + str(i) + '.csv')
    # remove the unique_id column and change gender to 0 and 1
    sample = sample.drop(['unique_id'], axis=1)
    dict_3 = {"female": 0, "male": 1}
    sample['Gender'] = sample['Gender'].replace(dict_3)
    X_test = sample[:, :-1]
    y_test = sample[-1]
    results = Model.predict(X_test)
    sample['predicted'] = results
    sample.to_csv(sample_folder + 'test_' + str(i) + '_results.csv', index=False)
