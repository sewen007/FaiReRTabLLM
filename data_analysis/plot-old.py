import json
import os

import pandas as pd
from matplotlib import pyplot as plt

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


def Plot_Result():
    # get result folder
    # print working directory
    print(os.getcwd())
    result_folder = '../Results/' + experiment_name + '/'
    # iterate through size folders in result folder
    for size_folder in os.listdir(result_folder):
        # get path of size folder
        size_folder_path = os.path.join(result_folder, size_folder)
        # iterate through shot folders in size folder
        for shot_folder in os.listdir(size_folder_path):
            # get path of shot folder
            shot_folder_path = os.path.join(size_folder_path, shot_folder)
            # get metrics file path
            metrics_file_path = os.path.join(shot_folder_path, 'metrics.csv')
            # read metrics file
            metrics_df = pd.read_csv(metrics_file_path)
            # plot metrics file for all shots in size folder. order sample accprding to sample number
            metrics_df = metrics_df.sort_values(by='Sample')
            plt.scatter(metrics_df['Sample'], metrics_df['Kendall Tau'], label=shot_folder, marker='*')
        plt.xlabel('Sample')
        plt.ylabel('Kendall Tau')
        plt.title('Kendall Tau vs Sample. rank: ' + size_folder)
        plt.legend()
        """ DIRECTORY MANAGEMENT """
        # create a directory for the plots based on the size folder
        plot_path = 'Plots/' + experiment_name + '/' + size_folder + '/'
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # save the plot in the plot folder
        plt.savefig(os.path.join(plot_path, 'Kendall_Tau_vs_Sample.png'))
        plt.close()

        # # iterate through sample folders in shot folder
        # for sample_folder in os.listdir(shot_folder_path):
        #     # get path of sample folder
        #     sample_folder_path = os.path.join(shot_folder_path, sample_folder)
        #     # iterate through result files in sample folder
        #     for result_file in os.listdir(sample_folder_path):
        #         # get path of result file
        #         result_file_path = os.path.join(sample_folder_path, result_file)
        #         # read result file
        #         result_df = pd.read_csv(result_file_path)
        #         # plot result
        #         plt.plot(result_df['Sample'], result_df['Kendall Tau'], label=result_file)
        #         plt.xlabel('Sample')
        #         plt.ylabel('Kendall Tau')
        #         plt.title('Kendall Tau vs Sample')
        #         plt.legend()
        #         plt.show()


Plot_Result()
