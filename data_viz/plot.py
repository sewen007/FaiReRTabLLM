import json
import os
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

with open('../settings.json', 'r') as f:
    settings = json.load(f)

start = time.time()

experiment_name = "LAW"


def get_files(directory, word):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            match = re.search(word, file)
            if match:
                # temp.append(directory + '/' + file)
                temp.append(os.path.join(dirpath, file))
    return temp


def PlotNCollate():
    print("Plotting data...")
    # result folder
    folder = Path(f"../Results/{experiment_name}/Ranked")
    print(folder)
    # select all files in the folder with 'ndcg' in the name
    experiments = [f for f in os.listdir(folder) if os.path.isdir(folder / f)]
    for experiment in experiments:
        if 'llama' in experiment:
            experiment = 'meta-llama/Meta-Llama-3-8B-Instruct'
        experiment_path = folder / experiment
        # Get the list of sizes
        sizes = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        for size in sizes:
            size_path = experiment_path / size
            # Get the list of shots
            shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
            for shot in shots:
                shot_path = size_path / shot
                print("Plotting graphs for shot", shot, "in experiment", experiment, "and size", size)

                # get ndcg files
                ndcg_files = get_files(shot_path, 'ndcg.csv')

                for ndcg_file in ndcg_files:

                    ndcg_data = pd.read_csv(ndcg_file)

                    if 'Position' in ndcg_data.columns:
                        ndcg_data = ndcg_data.drop(columns=['Position'])

                    if 'ListNet' in experiment:
                        # don't get average for ListNet
                        for col in ndcg_data.columns:
                            # if 'NDCG' in col:
                            col_name = col + '\n' + size + '\n' + experiment
                            ndcg_data[col_name] = ndcg_data[col]
                            avg_or_same_ndcg = ndcg_data[[col_name]]
                            # Save the results to a new CSV file
                            output_file = f'../Results/{experiment_name}/collated_ndcg.csv'
                            if not os.path.exists(output_file):
                                avg_or_same_ndcg.to_csv(output_file, index=False)
                            else:  # read the file and append the new data
                                collated_data = pd.read_csv(output_file)
                                collated_data[col_name] = avg_or_same_ndcg
                                collated_data.to_csv(output_file, index=False)
                    else:

                        col_name = 'AverageNDCG_' + shot + '\n' + size + '\n' + experiment

                        # Calculate the average NDCG for each position
                        ndcg_data[col_name] = ndcg_data.mean(axis=1)

                        # get the aggrregate mean and std - keep for special cases ndcg_data[col_name] =
                        # ndcg_data.apply(lambda row: f"{row.mean():.2f} ± {row.std():.2f}", axis=1)

                        avg_or_same_ndcg = ndcg_data[[col_name]]

                        # Save the results to a new CSV file
                        output_file = f'../Results/{experiment_name}/collated_ndcg.csv'
                        if not os.path.exists(output_file):
                            avg_or_same_ndcg.to_csv(output_file, index=False)
                        else:  # read the file and append the new data
                            collated_data = pd.read_csv(output_file)
                            collated_data[col_name] = avg_or_same_ndcg
                            collated_data.to_csv(output_file, index=False)

    # files = get_files(read_folder, 'ndcg')


def plot_ndcgs_bu():
    write_folder = Path(f"../Plots/{experiment_name}")
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    collated_data = pd.read_csv(f'../Results/{experiment_name}/collated_ndcg.csv')

    for index, row in collated_data.iterrows():
        # get the name of the model
        # model_name = index.name
        # Create bar chart for the current row
        fig, ax = plt.subplots(figsize=(12, 8))
        row.plot(kind='bar')

        # Set the title and labels
        plt.title(f'NDCG at {index + 1} for each model')
        plt.xlabel('Models')
        plt.ylabel('NDCG')
        plt.tight_layout()

        # Rotate the x-labels to ensure they are readable
        plt.xticks(rotation=45, ha='right')  # Rotate labels by 45 degrees and align them to the right

        # Save the plot as a PDF file with the specified naming convention
        plt.savefig(write_folder / f'NDCG_at_{index + 1}_Bar_Chart.pdf', bbox_inches='tight')

        # Close the plot to free memory
        plt.close()


def plot_ndcgs():
    write_folder = Path(f"../Plots/{experiment_name}")
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    collated_data = pd.read_csv(f'../Results/{experiment_name}/collated_ndcg.csv')

    # Determine which columns contain 'gpt' and 'llama'
    gpt_columns = [col for col in collated_data.columns if 'gpt' in col.lower()]
    llama_columns = [col for col in collated_data.columns if 'llama' in col.lower()]

    for idx, row in collated_data.iterrows():
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot each column individually with specified colors
        for col in collated_data.columns:
            if col in gpt_columns:
                color = '#F00000'
            elif col in llama_columns:
                color = '#3D6D9E'
            else:
                color = '#FFC725'

            ax.bar(col, row[col], color=color)

        # Set the title and labels
        plt.title(f'NDCG at {idx + 1} for each model')
        plt.xlabel('Models')
        plt.ylabel('NDCG')
        plt.tight_layout()
        plt.xticks(rotation=90, ha='center')
        # Save the plot as a PDF file with the specified naming convention
        plt.savefig(write_folder / f'NDCG_at_{idx + 1}_Bar_Chart.pdf', bbox_inches='tight')

        # Close the plot to free memory
        plt.close()


# Function to extract mean from "mean ± std" format
def extract_mean(value):
    if isinstance(value, float):
        return value
    return float(value.split(' ± ')[0])


# PlotNCollate()
plot_ndcgs()

end = time.time()
