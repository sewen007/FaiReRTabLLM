import json
import os
import re
import time
import math
from pathlib import Path

import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
import matplotlib.patches as patches

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()

start = time.time()

prompt_dict = {'prompt_1': 'Neutral', 'prompt_2': 'FC0', 'prompt_3': 'Tabular', 'prompt_4': 'FC1',
               'prompt_5': 'Neutral 2', 'prompt_6': 'CC1', 'prompt_8': 'CC2', 'prompt_10': 'FC2',
               'prompt_12': 'WC1', 'prompt_14': 'WC2', 'prompt_16': 'FD0', 'prompt_18': 'FD1', 'prompt_20': 'FD2'}


def get_files(directory, word):
    temp = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for file in filenames:
            match = re.search(word, file)
            if match:
                # temp.append(directory + '/' + file)
                temp.append(os.path.join(dirpath, file))
    return temp


def Collate(meta_exp='meta-llama/Meta-Llama-3.1-8B-Instruct', prompt_remove=None):
    print("Collating data...")
    # result folder
    folder = Path(f"../Results/{experiment_name}/Ranked")
    print(folder)
    # select all files in the folder with 'ndcg' in the name
    experiments = [f for f in os.listdir(folder) if os.path.isdir(folder / f)]
    for experiment in experiments:
        if 'llama' in experiment:
            experiment = meta_exp
        experiment_path = folder / experiment
        prompts = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        # Define the order for reorganization
        # prompts = ['prompt_16', 'prompt_18', 'prompt_20', 'prompt_2', 'prompt_4', 'prompt_10', 'prompt_6', 'prompt_8', 'prompt_12', 'prompt_14']

        for prompt in prompts:
            if prompt in prompt_remove:
                continue
            prompt_path = experiment_path / prompt
            # Get the list of sizes
            sizes = [f for f in os.listdir(prompt_path) if os.path.isdir(prompt_path / f)]
            for size in sizes:
                size_path = prompt_path / size
                # get the list of shots
                shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
                # select only shot_0, shot_1_, shot_2
                shots = [shot for shot in shots]
                # shots = [shot for shot in shots if
                #          (
                #                  'shot_0' in shot or 'shot_1' in shot or 'shot_2' in shot or 'shot_NA' in shot) and 'shot_10' not in shot]

                sorted_shots = sorted(shots, key=lambda x: float('inf') if 'NA' in x else int(x.split('_')[1]))
                for shot in sorted_shots:
                    shot_path = size_path / shot
                    print("Collating metrics for shot", shot, "in experiment", experiment, "and size", size)

                    # get ndcg file
                    ndcg_file_path = shot_path / 'ndcg.csv'
                    collate_ndcgs(ndcg_file_path, prompt, shot, size, experiment)

                    # get metric file
                    metric_file_path = shot_path / 'metrics.csv'
                    collate_metrics(metric_file_path, prompt, shot, size, experiment)


def collate_ndcgs(ndcg_file, prompt, shot, size, experiment):
    # check if the file exists
    if not os.path.exists(ndcg_file):
        return
    ndcg_data = pd.read_csv(ndcg_file)

    if 'Position' in ndcg_data.columns:
        ndcg_data = ndcg_data.drop(columns=['Position'])

    if 'GroundTruth' in experiment:
        sorted_columns = sorted(ndcg_data.columns, key=lambda x: int(x.split('_')[-1]))

        # don't get average for ListNet
        for col in sorted_columns:
            # if 'NDCG' in col:
            col_name = col + '\n' + size + '\n' + prompt + '\n' + experiment
            ndcg_data[col_name] = ndcg_data[col]
            avg_or_same_ndcg = ndcg_data[[col_name]]
            # approximate each value to 2 decimal places
            avg_or_same_ndcg = avg_or_same_ndcg.round(2)

            # Save the results to a new CSV file
            output_file = f'../Results/{experiment_name}/{experiment_name}_collated_ndcg.csv'
            if not os.path.exists(output_file):
                avg_or_same_ndcg.to_csv(output_file, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file)
                collated_data[col_name] = avg_or_same_ndcg
                collated_data.to_csv(output_file, index=False)

            # save the results to a new CSV file
            output_file_with_std = f'../Results/{experiment_name}/{experiment_name}_collated_ndcg_with_std.csv'
            if not os.path.exists(output_file_with_std):
                avg_or_same_ndcg.to_csv(output_file_with_std, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file_with_std)
                collated_data[col_name] = avg_or_same_ndcg
                collated_data.to_csv(output_file_with_std, index=False)
    else:

        col_name = 'AverageNDCG_' + shot + '\n' + size + '\n' + prompt + '\n' + experiment

        # Calculate the average NDCG for each position
        ndcg_data[col_name] = ndcg_data.mean(axis=1)

        # get the aggregate mean and std
        ndcg_mean_n_std = ndcg_data.apply(lambda row: f"{row.mean():.2f} ± {row.std():.2f}", axis=1)

        # Convert the Series to a DataFrame and add a column name
        ndcg_mean_n_std_df = pd.DataFrame(ndcg_mean_n_std, columns=[col_name])

        avg_or_same_ndcg = ndcg_data[[col_name]]

        # Save the results to a new CSV file
        output_file = f'../Results/{experiment_name}/{experiment_name}_collated_ndcg.csv'
        if not os.path.exists(output_file):
            avg_or_same_ndcg.to_csv(output_file, index=False)
        else:  # read the file and append the new data
            collated_data = pd.read_csv(output_file)
            collated_data[col_name] = avg_or_same_ndcg
            collated_data.to_csv(output_file, index=False)

        # Save the results to a new CSV file
        output_file_with_std = f'../Results/{experiment_name}/{experiment_name}_collated_ndcg_with_std.csv'
        if not os.path.exists(output_file_with_std):
            ndcg_mean_n_std_df.to_csv(output_file_with_std, index=False)
        else:  # read the file and append the new data
            collated_data = pd.read_csv(output_file_with_std)
            collated_data[col_name] = ndcg_mean_n_std_df
            collated_data.to_csv(output_file_with_std, index=False)


def collate_metrics(metric_file, prompt, shot, size, experiment):
    metric_data = pd.read_csv(metric_file)

    output_file = f'../Results/{experiment_name}/{experiment_name}_collated_metrics.csv'
    output_file_with_std = f'../Results/{experiment_name}/{experiment_name}_collated_metrics_with_std.csv'
    row_names = ['Kendall Tau', 'NDKL', 'Average Exposure', 'AveExpR CI-95']
    # if output file does not exist, create it
    if not os.path.exists(output_file):
        # Initialize the DataFrame with the row names
        output_df = pd.DataFrame(index=row_names)
    else:
        output_df = pd.read_csv(output_file, index_col=0)
    if not os.path.exists(output_file_with_std):
        output_df_with_std = pd.DataFrame(index=row_names)
    else:
        output_df_with_std = pd.read_csv(output_file_with_std, index_col=0)
    # DO NOT DELETE YET UNTIL END OF PAPER WRITING
    # if 'ListNet' in experiment:
    #
    #     # sort the rows in ascending order
    #     metric_data = metric_data.sort_values(by='Run', ascending=True)
    #
    #     # iterate through rows
    #     for i in range(len(metric_data)):
    #         # get the row
    #         row = metric_data.iloc[i]
    #
    #         # get the run number
    #         train_size = int(row['Run'])
    #         col_name = f'{train_size}' + '\n' + size + '\n' + prompt + '\n' + experiment
    #         kT_value = row['Kendall Tau']
    #         NDKL_value = row['NDKL']
    #         avg_Exp = row['Average Exposure']
    #
    #         # add new columns to the output dataframes
    #         output_df[col_name] = np.nan
    #
    #         # write column values to the output dataframes
    #         output_df.loc['Kendall Tau', col_name] = kT_value
    #         output_df.loc['NDKL', col_name] = NDKL_value
    #         output_df.loc['Average Exposure', col_name] = avg_Exp
    #
    #         # Save the results to a new CSV file
    #         output_file = f'../Results/{experiment_name}/collated_metrics.csv'
    #         output_df.to_csv(output_file, index=True)
    # else:  # not ListNet
    col_name = f'{shot}' + '\n' + size + '\n' + prompt + '\n' + experiment
    kT_mean = metric_data['Kendall Tau'].mean()
    kT_std = metric_data['Kendall Tau'].std()
    NDKL_mean = metric_data['NDKL'].mean()
    NDKL_std = metric_data['NDKL'].std()
    avg_Exp_mean = metric_data['Average Exposure'].mean()
    avg_Exp_std = metric_data['Average Exposure'].std()
    ci_95 = 1.96 * (avg_Exp_std / np.sqrt(len(metric_data)))  # 95% CI

    # add new columns to the output dataframes
    output_df[col_name] = np.nan
    output_df_with_std[col_name] = np.nan

    # write column values to the output dataframes
    output_df.loc['Kendall Tau', col_name] = kT_mean
    output_df.loc['NDKL', col_name] = NDKL_mean
    output_df.loc['Average Exposure', col_name] = avg_Exp_mean
    output_df.loc['AveExpR CI-95', col_name] = ci_95
    output_df_with_std[col_name] = output_df_with_std[col_name].astype('object')

    output_df_with_std.loc['Kendall Tau', col_name] = f"{kT_mean:.2f} ± {kT_std:.2f}"
    output_df_with_std.loc['NDKL', col_name] = f"{NDKL_mean:.2f} ± {NDKL_std:.2f}"
    output_df_with_std.loc['Average Exposure', col_name] = f"{avg_Exp_mean:.2f} ± {avg_Exp_std:.2f}"
    output_df_with_std.loc['AveExpR CI-95', col_name] = f"{avg_Exp_mean:.2f} ± {ci_95:.2f}"
    # Save the results to a new CSV file
    output_file_with_std = f'../Results/{experiment_name}/{experiment_name}_collated_metrics_with_std.csv'
    output_df_with_std.to_csv(output_file_with_std, index=True)

    # Save the results to a new CSV file
    output_file = f'../Results/{experiment_name}/{experiment_name}_collated_metrics.csv'
    output_df.to_csv(output_file, index=True)


# Function to extract numerical part from column names
def extract_number(col_name):
    parts = re.split(r'[_\n]', col_name)
    if 'GroundTruth' in parts:
        return 0
    return int(parts[2])


def plot_ndcgs(zero_only=False, non_zero_only=False, prompt='prompt_1'):
    write_folder = Path(f"../Plots/{experiment_name}/{prompt}")

    data = pd.read_csv(f'../Results/{experiment_name}/{experiment_name}_collated_ndcg.csv')

    # select only the columns with the specified prompt and ListNet
    collated_data = data.loc[:, data.columns.str.contains(prompt) | data.columns.str.contains('GroundTruth')]
    if zero_only:
        # add Listnet columns and columns with shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('GroundTruth') | collated_data.columns.str.contains('shot_0')]
        write_folder = write_folder / 'zero_only'

    if non_zero_only:
        # add Listnet columns and columns without shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('GroundTruth') | ~collated_data.columns.str.contains('shot_0')]
        write_folder = write_folder / 'non_zero_only'

    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    for idx, row in collated_data.iterrows():
        fig, ax = plt.subplots(figsize=(12, 8))

        # store handles and labels for the legend
        handles = []
        labels = []
        added_labels = set()  # tracking labels to avoid duplicates

        # for col in collated_data.columns:
        for col in collated_data.columns:
            color = get_color_and_label(col)[0]
            label = get_color_and_label(col)[1]

            bar = ax.bar(col, row[col], color=color, label=label)

            if label not in added_labels:
                handles.append(bar)
                labels.append(label)
                added_labels.add(label)

            # plt.legend()

            # add the integer value on top of the bar and make it vertical
            ax.text(col, row[col], f'{row[col]:.2f}', ha='center', va='bottom', rotation='vertical')

        # Set the title and labels
        plt.title(f'NDCG at {idx + 1} for each model, {experiment_name}, {prompt_dict[prompt]}')
        plt.xlabel('LLM shots')
        plt.ylabel('NDCG')
        plt.tight_layout()
        # new labels
        ax.set_xticks(range(len(collated_data.columns)))
        ax.set_xticklabels([get_label(col) for col in collated_data.columns])
        plt.xticks(rotation=90, ha='center')
        # Save the plot as a PDF file with the specified naming convention
        plt.savefig(write_folder / f'NDCG_at_{idx + 1}_Bar_Chart.png', bbox_inches='tight')
        # Close
        plt.close()
    # Create a new figure to contain just the legend
    legend_fig, legend_ax = plt.subplots(figsize=(2, 1))
    legend = legend_ax.legend(handles=handles, labels=labels, ncol=1, loc='center')
    legend_ax.axis('off')

    # Step 4: Save the legend as an image file
    legend_fig.tight_layout()
    legend_fig.savefig(write_folder / f'legend_ndcg.png', bbox_inches='tight')

    plt.close(legend_fig)


def get_color_and_label(col):
    # Define color and label mappings for each group
    if 'gpt-4' in col.lower() and 'pre-ranked' in col.lower():
        return '#F00000', 'GPT-4o-mini-pre-ranked'  # red
    elif 'llama-3-' in col.lower() and 'pre-ranked' in col.lower():
        return '#3D6D9E', 'Llama-3-8B-Instruct-pre-ranked'  # blue
    elif 'gemini-1.5-flash' in col.lower() and 'pre-ranked' in col.lower():
        return '#A6192E', 'Gemini 1.5 Flash-pre-ranked'  # red/orange
    elif 'gemini-1.5-pro' in col.lower() and 'pre-ranked' in col.lower():
        return '#008080', 'Gemini 1.5 Pro-pre-ranked'  # teal
    elif 'gpt-4' in col.lower() and 'pre-ranked' not in col.lower():
        return '#800080', 'GPT-4o-mini'  # purple
    elif 'llama-3-' in col.lower() and 'pre-ranked' not in col.lower():
        return '#00CED1', 'Llama-3-8B-Instruct'  # dark turquoise
    elif 'gemini-1.5-flash' in col.lower() and 'pre-ranked' not in col.lower():
        return '#808000', 'Gemini 1.5 Flash'  # olive
    elif 'gemini-1.5-pro' in col.lower() and 'pre-ranked' not in col.lower():
        return '#FF00FF', 'Gemini 1.5 Pro'  # magenta
    elif 'prompt_NA' in col:
        return '#FFC725', 'GroundTruth'  # yellow
    else:
        return '#000000', 'Unknown'  # default color and label if no match


def plot_metrics(zero_only=False, non_zero_only=False, prompt='prompt_1'):
    """Plots Kendall Tau, NDKL and Average Exposure"""
    write_folder = Path(f"../Plots/{experiment_name}/{prompt}")

    data = pd.read_csv(f'../Results/{experiment_name}/{experiment_name}_collated_metrics_with_std.csv')
    # make first column the index
    data = data.set_index(data.columns[0])

    # select only the columns with the specified prompt and ListNet
    collated_data = data.loc[:, data.columns.str.contains(prompt) | data.columns.str.contains('GroundTruth')]

    if zero_only:
        # add Listnet columns and columns with shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('GroundTruth') | collated_data.columns.str.contains('shot_0')]
        write_folder = write_folder / 'zero_only'

    if non_zero_only:
        # add Listnet columns and columns without shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('GroundTruth') | ~collated_data.columns.str.contains('shot_0')]
        write_folder = write_folder / 'non_zero_only'
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    # plot each row of metrics: Kendall Tau, NDKL, Average Exposure
    # first plot the kendall tau row
    fig, ax = plt.subplots(figsize=(12, 8))

    for col in collated_data.columns:
        value = float(collated_data.loc['Kendall Tau', col].split(' ± ')[0])
        std = float(collated_data.loc['Kendall Tau', col].split(' ± ')[1])
        ax.bar(col, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], label=col.split('\n')[0])

        # plt.legend()

    plt.title(f'Kendall Tau for each model, {experiment_name}')
    plt.xlabel('LLM shots')
    plt.ylabel('Kendall Tau')
    plt.tight_layout()
    ax.set_xticks(range(len(collated_data.columns)))
    ax.set_xticklabels([get_label(col) for col in collated_data.columns])

    # add the integer value on top of the bar and make it vertical
    # for i, col in enumerate(collated_data.columns):
    #     y_value = collated_data[col].values[0]  # Adjust if the values aren't in the first row
    #     plt.text(i, y_value, f'{y_value:.2f}', ha='center', va='bottom', rotation='vertical')

    plt.xticks(rotation=90, ha='center')

    plt.savefig(write_folder / f'Kendall_Tau_Bar_Chart.png', bbox_inches='tight')
    # k_plt = plt
    plt.close()

    # plot the NDKL row
    fig, ax = plt.subplots(figsize=(12, 8))
    for col in collated_data.columns:
        value = float(collated_data.loc['NDKL', col].split(' ± ')[0])
        std = float(collated_data.loc['NDKL', col].split(' ± ')[1])
        ax.bar(col, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], label=col.split('\n')[0])
        ax.text(col, value, f'{value:.2f}', ha='center', va='bottom',
                rotation='vertical')

    plt.title(f'NDKL for each model, {experiment_name}, {prompt_dict[prompt]}')
    plt.xlabel('LLM shots')
    plt.ylabel('NDKL')
    plt.tight_layout()
    ax.set_xticks(range(len(collated_data.columns)))
    ax.set_xticklabels([get_label(col) for col in collated_data.columns])

    plt.xticks(rotation=90, ha='center')
    plt.savefig(write_folder / f'NDKL_Bar_Chart.png', bbox_inches='tight')
    #n_plt = plt
    plt.close()

    # plot the Average Exposure row
    fig, ax = plt.subplots(figsize=(12, 8))
    for col in collated_data.columns:
        value = float(collated_data.loc['Average Exposure', col].split(' ± ')[0])
        std = float(collated_data.loc['Average Exposure', col].split(' ± ')[1])
        ax.bar(col, value, yerr=std, capsize=5, color=get_color_and_label(col)[0],
               label=col.split('\n')[0])
        # add the integer value on top of the bar and make it vertical
        ax.text(col, value, f'{value:.2f}',
                ha='center', va='bottom', rotation='vertical')
    # Draw a red line at y=1
    plt.axhline(y=1, color='red', linestyle='--')

    plt.text(x=len(collated_data.columns), y=1, s='oracle', color='black', va='bottom',
             ha='left')  # Label near the right edge

    plt.title(f'DAdv/Adv Average Exposure Ratio for each model, {experiment_name}, {prompt_dict[prompt]}')
    plt.xlabel('LLM shots')
    plt.ylabel('Average Exposure')
    #ax.legend()
    plt.tight_layout()
    ax.set_xticks(range(len(collated_data.columns)))
    ax.set_xticklabels([get_label(col) for col in collated_data.columns])
    plt.xticks(rotation=90, ha='center')
    plt.savefig(write_folder / f'Average_Exposure_Bar_Chart.png', bbox_inches='tight')
    plt.close()
    # a_plt = plt
    return  # k_plt, n_plt, a_plt


# Function to extract mean from "mean ± std" format
def extract_mean(value):
    if isinstance(value, float):
        return value
    return float(value.split(' ± ')[0])


def plot_ndcg_across():
    plot_folder = Path(f"../Plots/{experiment_name}/NDCG_Across")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    collated_data = pd.read_csv(f'../Results/{experiment_name}/{experiment_name}_collated_ndcg.csv')

    # # plot each column as a line plot with different colors on the same plot.
    # # Use red for GPT, blue for LLAMA, and yellow for the rest
    # Add markers to each line. iterate through the following markers '*,o,^,s,P,X,D'
    gpt_columns_0 = [col for col in collated_data.columns if 'gpt' in col.lower() and 'prompt_0' in col.lower()]
    llama_columns_0 = [col for col in collated_data.columns if 'llama' in col.lower() and 'prompt_0' in col.lower()]
    gpt_columns_1 = [col for col in collated_data.columns if 'gpt' in col.lower() and 'prompt_1' in col.lower()]
    llama_columns_1 = [col for col in collated_data.columns if 'llama' in col.lower() and 'prompt_1' in col.lower()]
    markers = ['*', 'o', '^', 's', 'P', 'X', 'D', 'H', 'v', '<', '>']
    fig, ax = plt.subplots(figsize=(12, 8))
    for col in collated_data.columns:
        if col in gpt_columns_0:
            color = '#F00000'
        elif col in llama_columns_0:
            color = '#3D6D9E'
        elif col in gpt_columns_1:
            color = 'pink'
        elif col in llama_columns_1:
            color = 'skyblue'
        else:
            color = '#FFC725'
        # iterate through the markers as they are used up
        ax.plot(collated_data[col], label=col, color=color, marker=markers[0], markerfacecolor='none')
        markers.append(markers.pop(0))

    plt.title(f'NDCG across different positions,{experiment_name}')
    plt.xlabel('Positions')
    plt.ylabel('NDCG')
    # plt.legend()
    plt.tight_layout()
    plt.xticks(rotation=90, ha='center')
    # save only legend

    plt.savefig(plot_folder / f'NDCG_Across_Line_Chart_4.png', bbox_inches='tight')
    plt.close()

    # Extract the handles and labels from the original plot
    handles, labels = ax.get_legend_handles_labels()

    # Create a new figure to contain just the legend
    legend_fig, legend_ax = plt.subplots()
    legend = legend_ax.legend(handles=handles, labels=labels, ncol=4, loc='center')
    legend_ax.axis('off')

    # Step 4: Save the legend as an image file
    legend_fig.savefig(plot_folder / f'legend.pdf', bbox_inches='tight')
    plt.close(legend_fig)


# Function to determine new labels based on substrings
def get_label(column_name):
    # look for text "shot in the column name and extract the number"
    if 'shot' in column_name:
        if 'NDCG' in column_name:
            return (column_name.split('\n')[0]).split('NDCG_')[1]
        else:
            return column_name.split('\n')[0]
    else:
        if 'NDCG' in column_name:
            return (column_name.split('\n')[0]).split('NDCG_')[1]
        else:
            return 'train_size_' + str(column_name.split('\n')[0])


def PlotLoss():
    loss_file_folder = Path(f"../LNLoss/{experiment_name}")
    loss_files = get_files(loss_file_folder, 'csv')

    for file in loss_files:
        print(file)
        loss_df = pd.read_csv(file)
        sns.set(style="white")
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        chart = sns.lineplot(data=loss_df[["loss"]], legend=False)

        plt.subplots_adjust(left=.25, bottom=0.15)

        delimiters = "(", ")"
        regex_pattern = '|'.join(map(re.escape, delimiters))
        graph_name = re.split(regex_pattern, file)
        train_size = (graph_name[2].split('_')[3]).split('.')[0]
        # print('graph_name', graph_name[0].split('/')[4].replace("\\", ""))

        chart.set_title(graph_name[1].split(',')[1] + ", " + train_size + ", " + experiment_name, fontdict={'size': 15})

        plt.xlabel("Iterations")
        plt.ylabel("DELTR Loss")
        # plt.legend(fontsize='large')

        """DIRECTORY MANAGEMENT"""
        graph_path = Path(
            "../LNLoss/" + experiment_name + '/Graphs/Loss/'
        )

        if not os.path.exists(graph_path):
            os.makedirs(graph_path)
        plt.savefig(os.path.join(graph_path, os.path.basename(file) + '.png'))
        plt.close()


def plot_gt_graphs():
    read_path = Path(f"../Results/{experiment_name}/Tests/50")
    write_folder = Path(f"../Plots/{experiment_name}/GroundTruth")
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)

    metric_file = read_path / "metrics.csv"
    ndcg_file = read_path / "ndcg.csv"
    metric_data = pd.read_csv(metric_file)
    ndcg_data = pd.read_csv(ndcg_file)

    # plot ndcg line graph
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.plot(ndcg_data['NDCG'])
    plt.title(f'GT NDCG for each position, {experiment_name}')
    plt.xlabel('Positions')
    plt.ylabel('NDCG')
    plt.tight_layout()
    # save plot
    plt.savefig(write_folder / f'NDCG_Line_Chart.png', bbox_inches='tight')

    return


def plot_skew(metrics_file_path):
    skew_file = pd.read_csv(metrics_file_path)

    skew_file['Ideal'] = 1

    # get dataset from file path
    dataset = experiment_name
    g_dis = protected_group

    if g_dis == 'female':
        g_adv = 'Males'
        g_dis = 'Females'
    else:
        g_adv = 'Females'
        g_dis = 'Males'

    skew_file.rename(columns={'Group_0': str(g_adv), 'Group_1': str(g_dis)}, inplace=True)
    sns.set(style="white")
    fig, ax = plt.subplots(figsize=(4, 4))

    sns.set(font_scale=1.2,  # Font scale factor (adjust to your preference)
            rc={"font.style": "normal",  # Set to "normal", "italic", or "oblique"
                "font.family": "serif",  # Choose your preferred font family
                # Font size
                "font.weight": "normal"  # Set to "normal", "bold", or a numeric value
                })

    sns.lineplot(data=skew_file[[g_dis, g_adv, "Ideal"]], dashes=False, ax=ax)

    # pipe, graph_title = get_graph_name(metrics_file_path)

    ax.set_title(dataset, fontsize='x-large', fontfamily='serif')

    # ax.title.set_position([0.5, -0.1])

    # ax.set_xticks(range(0, len(metrics_file), 200))

    # Set the x and y limits to start from 0
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.xlabel("Ranking position", fontfamily='serif')
    plt.ylabel("Skew", fontfamily='serif')
    plt.tight_layout()
    plt.legend(frameon=False, fontsize='xx-small', loc='upper right')
    # if dataset=='LAW':
    #     plt.legend(frameon=False, fontsize='xx-small',loc='upper right')
    # else:
    #     plt.legend("", frameon=False)
    # check = metrics_file_path.split(os.sep)[-2].split("/")[-1]
    # check2 = metrics_file_path.split(os.sep)
    """ DIRECTORY MANAGEMENT """
    graph_path = Path(
        "../../LLMFairRank/Plots/" +
        dataset + "/" + str(os.path.dirname(metrics_file_path).split(experiment_name)[1]))
    if not os.path.exists(graph_path):
        os.makedirs(graph_path)

    plt.savefig(os.path.join(graph_path, str(dataset) + '_skews.png'))
    plt.close()

    return


#
# meta_exs = ['meta-llama/Meta-Llama-3-8B-Instruct', 'meta-llama/Meta-Llama-3-8B-Instruct_pre-ranked']
# for meta_ex in meta_exs:
#     Collate(meta_ex)
#Collate('meta-llama/Meta-Llama-3-8B-Instruct', prompt_remove=['prompt_1'])

#Collate('meta-llama/Meta-Llama-3-8B-Instruct_pre-ranked')
#test_graphing()

#plot_legend()
#plot_ndcg_across()
# plot_metrics()
#plot_gt_graphs()

#PlotLoss()
# prompts = ['prompt_2', 'prompt_4', 'prompt_6', 'prompt_8', 'prompt_10', 'prompt_12', 'prompt_14', 'prompt_16',
#            'prompt_18', 'prompt_20']
# # prompts = ['prompt_16', 'prompt_18', 'prompt_20']
# # prompts = ['Tests']
# for prompt in prompts:
#     # plot_ndcgs(prompt=prompt)
#     # plot_ndcgs(zero_only=False, non_zero_only=True, prompt=prompt)
#     plot_ndcgs(zero_only=True, non_zero_only=False, prompt=prompt)
#     plot_metrics(prompt=prompt)
#     # plot_metrics(zero_only=False, non_zero_only=True, prompt=prompt)
#     # plot_metrics(zero_only=True, non_zero_only=False, prompt=prompt)

# end = time.time()


