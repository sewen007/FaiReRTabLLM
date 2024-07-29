import json
import os
import re
import time
from pathlib import Path
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]

start = time.time()


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
    print("Collating data...")
    # result folder
    folder = Path(f"../Results/{experiment_name}/Ranked")
    print(folder)
    # select all files in the folder with 'ndcg' in the name
    experiments = [f for f in os.listdir(folder) if os.path.isdir(folder / f)]
    for experiment in experiments:
        if 'llama' in experiment:
            experiment = 'meta-llama/Meta-Llama-3-8B-Instruct'
        experiment_path = folder / experiment
        prompts = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        for prompt in prompts:
            prompt_path = experiment_path / prompt
            # Get the list of sizes
            sizes = [f for f in os.listdir(prompt_path) if os.path.isdir(prompt_path / f)]
            for size in sizes:
                size_path = prompt_path / size
                # Get the list of shots
                shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
                for shot in shots:
                    shot_path = size_path / shot
                    print("Collating metrics for shot", shot, "in experiment", experiment, "and size", size)

                    # get ndcg files
                    ndcg_files = get_files(shot_path, 'ndcg.csv')
                    collate_ndcgs(ndcg_files, prompt, shot, size, experiment)

                    # get metric files
                    metric_files = get_files(shot_path, 'metrics.csv')
                    collate_metrics(metric_files, prompt, shot, size, experiment)
        # # Get the list of sizes
        # sizes = [f for f in os.listdir(experiment_path) if os.path.isdir(experiment_path / f)]
        # for size in sizes:
        #     size_path = experiment_path / size
        #     # Get the list of shots
        #     shots = [f for f in os.listdir(size_path) if os.path.isdir(size_path / f)]
        #     if 'ListNet' in experiment:
        #         sorted_shots = shots
        #     else:
        #         sorted_shots = sorted(shots, key=lambda x: int(x.split('_')[1]))
        #     print(shots)
        #     for shot in sorted_shots:
        #         shot_path = size_path / shot
        #         print("Collating metrics for shot", shot, "in experiment", experiment, "and size", size)
        #
        #         # # get ndcg files
        #         ndcg_files = get_files(shot_path, 'ndcg.csv')
        #         collate_ndcgs(ndcg_files, shot, size, experiment)
        #
        #         # get metric files
        #         metric_files = get_files(shot_path, 'metrics.csv')
        #         collate_metrics(metric_files, shot, size, experiment)


def collate_ndcgs(ndcg_files, prompt, shot, size, experiment):
    for ndcg_file in ndcg_files:

        ndcg_data = pd.read_csv(ndcg_file)

        if 'Position' in ndcg_data.columns:
            ndcg_data = ndcg_data.drop(columns=['Position'])

        if 'ListNet' in experiment:
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
                output_file = f'../Results/{experiment_name}/collated_ndcg.csv'
                if not os.path.exists(output_file):
                    avg_or_same_ndcg.to_csv(output_file, index=False)
                else:  # read the file and append the new data
                    collated_data = pd.read_csv(output_file)
                    collated_data[col_name] = avg_or_same_ndcg
                    collated_data.to_csv(output_file, index=False)

                # Save the results to a new CSV file
                output_file_with_std = f'../Results/{experiment_name}/collated_ndcg_with_std.csv'
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
            output_file = f'../Results/{experiment_name}/collated_ndcg.csv'
            if not os.path.exists(output_file):
                avg_or_same_ndcg.to_csv(output_file, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file)
                collated_data[col_name] = avg_or_same_ndcg
                collated_data.to_csv(output_file, index=False)

            # Save the results to a new CSV file
            output_file_with_std = f'../Results/{experiment_name}/collated_ndcg_with_std.csv'
            if not os.path.exists(output_file_with_std):
                ndcg_mean_n_std_df.to_csv(output_file_with_std, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file_with_std)
                collated_data[col_name] = ndcg_mean_n_std_df
                collated_data.to_csv(output_file_with_std, index=False)


def collate_metrics(metric_files, prompt, shot, size, experiment):
    for metric_file in metric_files:
        metric_data = pd.read_csv(metric_file)

        if 'ListNet' in experiment:
            # sort rows in ascending order
            metric_data = metric_data.sort_values(by='Run', ascending=True)

            # iterate through rows
            for i in range(len(metric_data)):
                # get the row
                row = metric_data.iloc[i]

                # get the run number
                train_size = int(row['Run'])
                col_name = f'KT_train_size_{train_size}' + '\n' + size + '\n' + prompt + '\n' + experiment
                col_NDKL = f'NDKL_train_size_{train_size}' + '\n' + size + '\n' + prompt + '\n' + experiment
                col_avgExp = f'AvgExp_train_size_{train_size}' + '\n' + size + '\n' + prompt + '\n' + experiment
                kT_value = round(row['Kendall Tau'], 2)
                NDKL_value = row['NDKL']
                avg_Exp = row['Average Exposure']

                # round to 2 decimal places
                # kT_value = round(kT_value, 2)

                # create new data frame with the mean value
                output_df = pd.DataFrame(columns=[col_name])
                output_df[col_name] = row['Kendall Tau']
                output_df[col_NDKL] = NDKL_value
                output_df[col_avgExp] = avg_Exp

                # Save the results to a new CSV file
                output_file = f'../Results/{experiment_name}/collated_metrics.csv'
                if not os.path.exists(output_file):
                    output_df[col_name].to_csv(output_file, index=False)
                else:  # read the file and append the new data
                    collated_data = pd.read_csv(output_file)
                    collated_data[col_name] = [kT_value]
                    collated_data[col_NDKL] = [NDKL_value]
                    collated_data[col_avgExp] = [avg_Exp]
                    collated_data.to_csv(output_file, index=False)

                # Save the results to a new CSV file
                output_file_with_std = f'../Results/{experiment_name}/collated_metrics_with_std.csv'
                if not os.path.exists(output_file_with_std):
                    output_df[col_name].to_csv(output_file_with_std, index=False)
                else:  # read the file and append the new data
                    collated_data = pd.read_csv(output_file_with_std)
                    collated_data[col_name] = [kT_value]
                    collated_data[col_NDKL] = [NDKL_value]
                    collated_data[col_avgExp] = [avg_Exp]
                    collated_data.to_csv(output_file_with_std, index=False)

        else:

            col_name = 'KT_' + shot + '\n' + size + '\n' + prompt + '\n' + experiment
            col_NDKL = 'NDKL_' + shot + '\n' + size + '\n' + prompt + '\n' + experiment
            col_avgExp = 'AvgExp_' + shot + '\n' + size + '\n' + prompt + '\n' + experiment

            kT_mean = metric_data['Kendall Tau'].mean()
            kT_std = metric_data['Kendall Tau'].std()
            NDKL_mean = metric_data['NDKL'].mean()
            NDKL_std = metric_data['NDKL'].std()
            avg_Exp_mean = metric_data['Average Exposure'].mean()
            avg_Exp_std = metric_data['Average Exposure'].std()

            # create new data frame with the mean value
            output_df = pd.DataFrame(columns=[col_name, col_NDKL])
            output_df[col_name] = [kT_mean]
            output_df[col_NDKL] = [NDKL_mean]
            output_df[col_avgExp] = [avg_Exp_mean]

            # create new data frame with the mean and std value
            output_df_with_std = pd.DataFrame(columns=[col_name, col_NDKL])
            output_df_with_std[col_name] = [f"{kT_mean:.2f} ± {kT_std:.2f}"]
            output_df_with_std[col_NDKL] = [f"{NDKL_mean:.2f} ± {NDKL_std:.2f}"]
            output_df_with_std[col_avgExp] = [f"{avg_Exp_mean:.2f} ± {avg_Exp_std:.2f}"]

            # Save the results to a new CSV file
            output_file = f'../Results/{experiment_name}/collated_metrics.csv'
            if not os.path.exists(output_file):
                output_df.to_csv(output_file, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file)
                collated_data[col_name] = [kT_mean]
                collated_data[col_NDKL] = [NDKL_mean]
                collated_data[col_avgExp] = [avg_Exp_mean]
                collated_data.to_csv(output_file, index=False)

            # Save the results to a new CSV file
            output_file_with_std = f'../Results/{experiment_name}/collated_metrics_with_std.csv'
            if not os.path.exists(output_file_with_std):
                output_df_with_std.to_csv(output_file_with_std, index=False)
            else:  # read the file and append the new data
                collated_data = pd.read_csv(output_file_with_std)
                collated_data[col_name] = [f"{kT_mean:.2f} ± {kT_std:.2f}"]
                collated_data[col_NDKL] = [f"{NDKL_mean:.2f} ± {NDKL_std:.2f}"]
                collated_data[col_avgExp] = [f"{avg_Exp_mean:.2f} ± {avg_Exp_std:.2f}"]
                collated_data.to_csv(output_file_with_std, index=False)


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
        plt.title(f'NDCG at {idx + 1} for each model, {experiment_name}')
        plt.xlabel('Models')
        plt.ylabel('NDCG')
        plt.tight_layout()
        plt.xticks(rotation=90, ha='center')
        # Save the plot as a PDF file with the specified naming convention
        plt.savefig(write_folder / f'NDCG_at_{idx + 1}_Bar_Chart.png', bbox_inches='tight')

        # Close the plot to free memory
        plt.close()


# Function to extract mean from "mean ± std" format
def extract_mean(value):
    if isinstance(value, float):
        return value
    return float(value.split(' ± ')[0])


def plot_ndcg_across():
    plot_folder = Path(f"../Plots/{experiment_name}/NDCG_Across")
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)
    collated_data = pd.read_csv(f'../Results/{experiment_name}/collated_ndcg.csv')

    # # plot each column as a line plot with different colors on the same plot.
    # # Use red for GPT, blue for LLAMA, and yellow for the rest
    # Add markers to each line. iterate through the following markers '*,o,^,s,P,X,D'
    markers = ['*', 'o', '^', 's', 'P', 'X', 'D', 'H', 'v', '<', '>']
    fig, ax = plt.subplots(figsize=(12, 8))
    for col in collated_data.columns:
        if 'gpt' in col.lower():
            color = '#F00000'
        elif 'llama' in col.lower():
            color = '#3D6D9E'
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
    legend_fig.savefig(plot_folder / f'legend.png', bbox_inches='tight')
    plt.close(legend_fig)

    # # plot each column as a line plot with different colors on the same plot.
    # # Use red for GPT, blue for LLAMA, and yellow for the rest
    # fig, ax = plt.subplots(figsize=(12, 8))
    # for col in collated_data.columns:
    #     if 'gpt' in col.lower():
    #         color = '#F00000'
    #     elif 'llama' in col.lower():
    #         color = '#3D6D9E'
    #     else:
    #         color = '#FFC725'
    #     ax.plot(collated_data[col], label=col, color=color)
    # plt.title(f'NDCG across different positions')
    # plt.xlabel('Positions')
    # plt.ylabel('NDCG')
    # plt.legend()
    # plt.tight_layout()
    # plt.xticks(rotation=90, ha='center')
    # plt.savefig(plot_folder / f'NDCG_Across_Line_Chart_2.png', bbox_inches='tight')

    # # plot each column as a line plot with different colors on the same plot
    # fig, ax = plt.subplots(figsize=(12, 8))
    # for col in collated_data.columns:
    #     ax.plot(collated_data[col], label=col)
    # plt.title(f'NDCG across different positions')
    # plt.xlabel('Positions')
    # plt.ylabel('NDCG')
    # plt.legend()
    # plt.tight_layout()
    # plt.xticks(rotation=90, ha='center')
    # plt.savefig(plot_folder / f'NDCG_Across_Line_Chart.png', bbox_inches='tight')
    # plt.close()

    # plot each column as a line plot with different colors
    # for col in collated_data.columns:
    #     fig, ax = plt.subplots(figsize=(12, 8))
    #     ax.plot(collated_data[col], label=col)
    #     plt.title(f'NDCG across different positions for {col}')
    #     plt.xlabel('Positions')
    #     plt.ylabel('NDCG')
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.xticks(rotation=90, ha='center')
    #     plt.savefig(plot_folder / f'_Line_Chart.png', bbox_inches='tight')
    #     #plt.close()


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


PlotNCollate()
# plot_ndcgs()
# plot_ndcg_across()

#PlotLoss()

end = time.time()
