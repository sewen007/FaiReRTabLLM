from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import os
import json

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

with open('./settings.json', 'r') as f:
    settings = json.load(f)
dataset = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]
protected_group = settings["READ_FILE_SETTINGS"]["DADV_GROUP"].lower()

datasets = ["NBAWNBA", "LOAN", "BostonMarathon", "LAW"]

prompt_dict = {'prompt_2': 'BASE', 'prompt_4': 'REP', 'prompt_6': 'EXP', 'prompt_8': 'FEA', 'prompt_10': 'FEA+REP',
               'prompt_12': 'FEA+EXP', 'prompt_14': 'FEA+PRO', 'prompt_16': 'FEA+PRO+REP', 'prompt_18': 'FEA+PRO+EXP',
               'prompt_NA': 'Initial', 'prompt_NAD': 'DetConstSort'}
experiment_name = dataset
prompts = ['prompt_2', 'prompt_4', 'prompt_6', 'prompt_8', 'prompt_10', 'prompt_12', 'prompt_14', 'prompt_16',
           'prompt_18']


def get_color_and_label(col):
    # Define color and label mappings for each group
    if 'gpt-4' in col.lower():
        return '#800080', 'GPT-4o-mini'  # purple
    elif 'llama-3-' in col.lower():
        return '#00CED1', 'Llama-3-8B-Instruct'  # dark turquoise
    elif 'gemini-1.5-flash' in col.lower():
        return '#DFF2DF', 'Gemini 1.5 Flash'  # light olive
    elif 'gemini-1.5-pro' in col.lower():
        return '#FFCCFF', 'Gemini 1.5 Pro'  # lighter magenta
    elif 'prompt_NA\n' in col:
        return '#FFC725', 'Initial'  # yellow
    elif 'prompt_NAD' in col:
        return '#4682B4', 'DetConstSort'
    else:
        return '#000000', 'Unknown'  # default color and label if no match


def get_prompt(col):
    # in col, search for 'prompt' and return string after prompt but before /
    if 'prompt' in col:
        return prompt_dict[col.split('\n')[2]]
        # return col.split('prompt')[1].split('/')[0]
    else:
        return 'ListNet'


def get_label(column_name):
    # look for text "shot in the column name and extract the number"
    if 'shot_NAD' in column_name:
        return 'shot_NA'
    elif 'shot' in column_name:
        if 'NDCG' in column_name:
            return (column_name.split('\n')[0]).split('NDCG_')[1]
        else:
            return column_name.split('\n')[0]
    else:
        if 'NDCG' in column_name:
            return (column_name.split('\n')[0]).split('NDCG_')[1]
        else:

            return 'train_size_' + str(column_name.split('\n')[0])


# Define hatching patterns based on custom logic
def get_hatch_pattern(label):
    if 'REP' in label:
        return '//'
    elif 'EXP' in label:
        return '.'
    else:
        return ''  # No hatch


def get_hatch_color(label):
    if 'REP' in label:
        return 'black'
    elif 'EXP' in label:
        return 'black'
    else:
        return 'black'  # Default hatch color


def plot_metrics(zero_only=False, non_zero_only=False, llm='gemini-1.5-flash', metric='Kendall Tau',
                 specific_prompts=None):
    data = pd.read_csv(f'./Results/{experiment_name}/{experiment_name}_collated_metrics_with_std.csv')

    # make first column the index
    data = data.set_index(data.columns[0])

    # select only the columns with the specified llm, Initial and DetConstSort
    collated_data = data.loc[:,
                    data.columns.str.contains(llm) | data.columns.str.contains('Initial') | data.columns.str.contains(
                        'DetConstSort')]
    order = ['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_8\n', 'prompt_10\n', 'prompt_12\n', 'prompt_14\n',
             'prompt_16\n', 'prompt_18\n', 'prompt_NA\n', 'prompt_NAD\n']

    custom_size = 'large'
    if zero_only:
        custom_size = 'xx-large'

    if zero_only:
        # add Listnet columns and columns with shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains(
                            'shot_0') | collated_data.columns.str.contains('DetConstSort')]
        #  = write_folder / 'zero_only'

    if non_zero_only:
        # add Listnet columns and columns without shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | data.columns.str.contains(
                            'DetConstSort') | ~collated_data.columns.str.contains('shot_0')]

    # Reorder the columns
    ordered_columns = sorted(collated_data.columns,
                             key=lambda col: next((i for i, word in enumerate(order) if word in col), len(order)))
    collated_data = collated_data[ordered_columns]

    if specific_prompts is not None:
        collated_data = collated_data.loc[:, collated_data.columns.str.contains('|'.join(specific_prompts))]

    fig, ax = plt.subplots(figsize=(15, 6))

    # store handles and labels for the legend
    handles = []
    labels = []
    added_labels = set()  # tracking labels to avoid duplicates

    bar_count = 1

    for col in collated_data.columns:
        value = float(collated_data.loc[metric, col].split(' ± ')[0])
        std = float(collated_data.loc[metric, col].split(' ± ')[1])
        if zero_only:
            custom_label = get_prompt(col)
        else:
            custom_label = get_prompt(col) + ', ' + get_label(col)
            # if (bar_count-1) % 3 == 0:
            #   custom_label = get_prompt(col)+ ', ' + get_label(col)
            # else:
            #   custom_label = get_label(col)
        if metric == 'Average Exposure' or metric == 'AveExpR CI-95':
            if 'Initial' in col:
                plt.axhline(y=value, color='blue', linestyle='--')
                plt.text(x=len(collated_data.columns), y=1, s='', color='black', va='bottom',
                         ha='left')  # Label near the right edge

        # Use the custom label (wetin) for bar placement
        x_pos = custom_label
        hatch = get_hatch_pattern(x_pos)
        # bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,
        #             label=col.split('\n')[2])
        if metric == 'NDKL':
            tinny_bitty = 0.005
        else:
            tinny_bitty = 0.01

        written_number_size = 25
        #
        if experiment_name == 'BostonMarathon':
            if 'Ave' in metric:
                if 'prompt_16' in col:
                    if 'shot_0' in col:
                        if llm == 'gemini-1.5-flash':
                            bar = ax.bar(x_pos, value, color=get_color_and_label(col)[0], hatch=hatch,
                                         label=col.split('\n')[2])
                            if not zero_only:
                                if not non_zero_only:
                                    written_number_size = 'large'
                            ax.text(x_pos, value + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                    fontsize=written_number_size, fontweight='bold')
                        elif llm == 'gemini-1.5-pro':
                            # do not add std for prompt_16
                            bar = ax.bar(x_pos, value, color=get_color_and_label(col)[0], hatch=hatch,
                                         label=col.split('\n')[2])
                            if not zero_only:
                                if not non_zero_only:
                                    written_number_size = 'large'
                            ax.text(x_pos, value + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                    fontsize=written_number_size, fontweight='bold')
                        else:
                            bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0],
                                         hatch=hatch,
                                         label=col.split('\n')[2])
                            if not zero_only:
                                if not non_zero_only:
                                    written_number_size = 'large'
                            ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                    fontsize=written_number_size, rotation=90, fontweight='bold')
                    else:
                        bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,
                                     label=col.split('\n')[2])
                        if not zero_only:
                            if not non_zero_only:
                                written_number_size = 'large'
                        ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                fontsize=written_number_size, rotation=90, fontweight='bold')
                #else:
                #bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,label=col.split('\n')[2])
                elif 'prompt_18' in col:
                    if 'shot_0' in col:
                        if llm == 'gemini-1.5-flash':
                            bar = ax.bar(x_pos, value, color=get_color_and_label(col)[0], hatch=hatch,
                                         label=col.split('\n')[2])
                            if not zero_only:
                                if not non_zero_only:
                                    written_number_size = 'large'
                            ax.text(x_pos, value + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                    fontsize=written_number_size, fontweight='bold')
                        elif llm == 'gemini-1.5-pro':
                            # do not add std for prompt_16
                            bar = ax.bar(x_pos, value, color=get_color_and_label(col)[0], hatch=hatch,
                                         label=col.split('\n')[2])
                            if not zero_only:
                                if not non_zero_only:
                                    written_number_size = 'large'
                            ax.text(x_pos, value + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                    fontsize=written_number_size, fontweight='bold')
                        else:
                            bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0],
                                         hatch=hatch,
                                         label=col.split('\n')[2])
                            if not zero_only:
                                if not non_zero_only:
                                    written_number_size = 'large'
                            ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                    fontsize=written_number_size, rotation=90, fontweight='bold')
                    else:
                        bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,
                                     label=col.split('\n')[2])
                        if not zero_only:
                            if not non_zero_only:
                                written_number_size = 'large'
                        ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                                fontsize=written_number_size, rotation=90, fontweight='bold')
                else:
                    bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,
                                 label=col.split('\n')[2])
                    if not zero_only:
                        if not non_zero_only:
                            written_number_size = 'large'
                    ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                            fontsize=written_number_size, rotation=90, fontweight='bold')
            else:
                bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,
                             label=col.split('\n')[2])
                if not zero_only:
                    if not non_zero_only:
                        written_number_size = 'large'
                ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                        fontsize=written_number_size, rotation=90, fontweight='bold')


        else:
            bar = ax.bar(x_pos, value, yerr=std, capsize=5, color=get_color_and_label(col)[0], hatch=hatch,
                         label=col.split('\n')[2])
            if not zero_only:
                if not non_zero_only:
                    written_number_size = 'large'
            ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
                    fontsize=written_number_size, rotation=90, fontweight='bold')

        for patch in bar:
            patch.set_edgecolor(get_hatch_color(x_pos))

        # Increment the group index for each new bar
        bar_count += 1

        # add value on each bar
        # add the integer value on top of the bar

        # if metric == 'NDKL':
        #     tinny_bitty = 0.005
        # else:
        #     tinny_bitty = 0.01
        #
        # written_number_size = 'xx-large'
        # if not zero_only:
        #     if not non_zero_only:
        #         written_number_size = 'large'
        # ax.text(x_pos, value + std + tinny_bitty, f'{value:.2f}', ha='center', va='bottom',
        #         fontsize=written_number_size, rotation=90, fontweight='bold')
        color = get_color_and_label(col)[0]
        label = get_color_and_label(col)[1]

        if label not in added_labels:
            handles.append(bar)
            labels.append(label)
            added_labels.add(label)

    plt.title(f'{llm}, {dataset}', fontsize=40)
    # Set the y-axis maximum limit
    y = 1.9

    if metric == 'NDKL':
        plt.ylim(0.05, 0.7)
    elif 'Ave' in metric:
        plt.ylim(0.6, 1.85)
    else:
        plt.ylim(0, y)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=25)
    if zero_only:
        plt.xlabel('prompts', fontsize=30)
    else:
        plt.xlabel('prompts\n' + 'LLM shots', fontsize=30)

    plt.ylabel(f'{metric}', fontsize=30)
    plt.tight_layout()
    ax.set_xticks(range(len(collated_data.columns)))
    #print('checking index', collated_data.columns)
    # Get the current x-tick positions and rotation
    current_ticks = ax.get_xticks()
    if specific_prompts is not None:
        if specific_prompts == ['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_NA', 'prompt_NAD']:
            notion = 'base'
            new_labels = {'0': 'BASE, 0', '1': '1', '2': '2', '3': '3', '4': '4', '5': 'REP, 0', '6': '1', '7': '2',
                          '8': '3', '9': '4',
                          '10': 'EXP, 0', '11': '1', '12': '2', '13': '3', '14': '4', '15': 'Initial',
                          '16': 'DetConstSort'}
            if zero_only:
                new_labels = {'0': 'BASE', '1': 'REP', '2': 'EXP', '3': 'Initial', '4': 'DetConstSort'}
            new_labels_list = [new_labels.get(str(tick), str(tick)) for tick in current_ticks]
            ax.set_xticklabels(new_labels_list, fontsize=30)
        elif specific_prompts == ['prompt_8\n', 'prompt_10\n', 'prompt_12\n', 'prompt_NA']:
            notion = 'feature'
            new_labels = {'0': 'FEA, 0', '1': '1', '2': '2', '3': '3', '4': '4', '5': 'FEA+REP, 0', '6': '1', '7': '2',
                          '8': '3', '9': '4',
                          '10': 'FEA+EXP, 0', '11': '1', '12': '2', '13': '3', '14': '4', '15': 'Initial', '16': 'DetConstSort'}
            if zero_only:
                new_labels = {'0': 'FEA', '1': 'FEA+REP', '2': 'FEA+EXP', '3': 'Initial', '4': 'DetConstSort'}
            new_labels_list = [new_labels.get(str(tick), str(tick)) for tick in current_ticks]
            ax.set_xticklabels(new_labels_list, fontsize=30)
        elif specific_prompts == ['prompt_14\n', 'prompt_16\n', 'prompt_18\n', 'prompt_NA']:
            notion = 'feature_protected'
            new_labels = {'0': 'FEA+PRO, 0', '1': '1', '2': '2', '3': '3', '4': '4', '5': 'FEA+PRO+REP, 0', '6': '1',
                          '7': '2', '8': '3', '9': '4',
                          '10': 'FEA+PRO+EXP, 0', '11': '1', '12': '2', '13': '3', '14': '4', '15': 'Initial', '16': 'DetConstSort'}
            if zero_only:
                new_labels = {'0': 'FEA+PRO', '1': 'FEA+PRO+REP', '2': 'FEA+PRO+EXP', '3': 'Initial', '4': 'DetConstSort'}
            new_labels_list = [new_labels.get(str(tick), str(tick)) for tick in current_ticks]
            ax.set_xticklabels(new_labels_list, fontsize=30)

    else:  # specific prompt
        if not zero_only:
            new_labels = {'0': 'BASE, 0', '1': '1', '2': '2', '3': '3', '4': '4',
                          '5': 'REP, 0', '6': '1', '7': '2', '8': '3', '9': '4',
                          '10': 'EXP, 0', '11': '1', '12': '2', '13': '3', '14': '4',
                          '15': 'FEA, 0', '16': '1', '17': '2', '18': '3', '19': '4',
                          '20': 'FEA+REP, 0', '21': '1', '22': '2', '23': '3', '24': '4',
                          '25': 'FEA+EXP, 0', '26': '1', '27': '2', '28': '3', '29': '4',
                          '30': 'FEA+PRO, 0', '31': '1', '32': '2', '33': '3', '34': '4',
                          '35': 'FEA+PRO+REP, 0', '36': '1', '37': '2', '38': '3', '39': '4',
                          '40': 'FEA+PRO+EXP, 0', '41': '1', '42': '2', '43': '3', '44': '4', '45': "Initial",
                          '46': "DetConstSort"}
            new_labels_list = [new_labels.get(str(tick), str(tick)) for tick in current_ticks]
            ax.set_xticklabels(new_labels_list, fontsize=20)
        else:  # zero_only
            new_labels = {'0': 'BASE', '1': 'REP', '2': 'EXP', '3': 'FEA', '4': 'FEA+REP', '5': 'FEA+EXP',
                          '6': 'FEA+PRO',
                          '7': 'FEA+PRO+REP', '8': 'FEA+PRO+EXP', '9': "Initial", '10': "DetConstSort"}
            # Use the dictionary to update x-tick labels, safely handling missing keys
            # Get the labels from the positions (you can use the new_labels mapping here)
            new_labels_list = [new_labels.get(str(tick), str(tick)) for tick in current_ticks]
            ax.set_xticklabels(new_labels_list, fontsize=30)
        # Apply the new labels and rotate by 90 degrees
        ax.set_xticks(current_ticks)
    oracle = 'oracle'
    if specific_prompts is not None:
        oracle = ' '
    # Draw a red line at y=1
    if metric == 'Average Exposure' or metric == 'AveExpR CI-95':
        plt.axhline(y=1, color='red', linestyle='--')
        plt.text(x=len(collated_data.columns), y=1, s=oracle, color='black', va='bottom',
                 ha='left', fontsize=25)  # Label near the right edge

    plt.xticks(rotation=45, ha='right')

    if non_zero_only:
        # do nothing
        pass
    else:
        if zero_only:
            for i in range(2, len(collated_data.columns), 3):  # Start from index 2 and step by 3
                plt.axvline(x=i + 0.5, color='grey', linestyle='--')  # Use x=i+0.5 to place it after the bar
        else:
            # Add vertical lines after every third bar
            for i in range(-1, len(collated_data.columns), 5):  # Start from index 2 and step by 3
                plt.axvline(x=i + 0.5, color='grey', linestyle='--',
                            linewidth=1)  # Use x=i+0.5 to place it after the bar
            # Add vertical lines after every third bar

    # plt.legend()
    if specific_prompts is not None:
        save_folder = f'./Plots/{dataset}/specific_prompts/{notion}/'
        if zero_only:
            save_folder = f'./Plots/{dataset}/specific_prompts/{notion}/zero_only/'

    else:
        save_folder = f'./Plots/{dataset}/'
        if zero_only:
            save_folder = f'./Plots/{dataset}/zero_only/'
        if non_zero_only:
            save_folder = f'./Plots/{dataset}/non_zero_only/'
        # create save folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # plt.close(legend_fig)
    plt.savefig(f'{save_folder}{llm}_{metric}.pdf', bbox_inches='tight')

    # Create a new figure to contain just the legend
    legend_fig, legend_ax = plt.subplots(figsize=(2, 1))
    legend = legend_ax.legend(handles=handles, labels=labels, ncol=1, loc='center')
    legend_ax.axis('off')

    # Step 4: Save the legend as an image file
    legend_fig.tight_layout()
    legend_fig.savefig(f'legend.png', bbox_inches='tight')


def plot_ndcgs(zero_only=False, non_zero_only=False, prompt='prompt_1'):
    write_folder = f"./Plots/{dataset}/NDCG"

    data = pd.read_csv(f'./Results/{experiment_name}/{experiment_name}_collated_ndcg.csv')

    # select only the columns with the specified prompt and ListNet
    collated_data = data.loc[:, data.columns.str.contains(prompt) | data.columns.str.contains(
        'Initial') | data.columns.str.contains('DetConstSort')]
    order = ['gemini-1.5-flash', 'gemini-1.5-pro', 'Meta-Llama-3-8B-Instruct', 'Initial', 'DetConstSort']
    # Reorder the columns
    ordered_columns = sorted(collated_data.columns,
                             key=lambda col: next((i for i, word in enumerate(order) if word in col), len(order)))
    collated_data = collated_data[ordered_columns]
    if zero_only:
        # add Listnet columns and columns with shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains(
                            'shot_0') | collated_data.columns.str.contains('DetConstSort')]
        write_folder = write_folder / 'zero_only'

    if non_zero_only:
        # add Listnet columns and columns without shot_0
        collated_data = collated_data.loc[:,
                        collated_data.columns.str.contains('Initial') | collated_data.columns.str.contains(
                            'DetConstSort') | ~collated_data.columns.str.contains('shot_0')]
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
            hatch = get_hatch_pattern(prompt)

            bar = ax.bar(col, row[col], color=color, label=label, hatch=hatch)
            # manually change name beneath the bar withouth changing col name

            if label not in added_labels:
                handles.append(bar)
                labels.append(label)
                added_labels.add(label)

            # plt.legend()

            # add the integer value on top of the bar and make it vertical
            ax.text(col, row[col], f'{row[col]:.2f}', ha='center', va='bottom', rotation='vertical', fontsize=25,
                    fontweight='bold')
        # Change the names under the bars manually
        # new_labels = ['shot_0', 'shot_1', 'shot_2', 'shot_3', 'shot_4', '']  # Custom names
        # ax.set_xticks(range(len(collated_data)))  # Ensure ticks align with bars
        # ax.set_xticklabels(new_labels)
        # Set the title and labels
        plt.title(f'prompt:{prompt_dict[prompt]}, {dataset}', fontsize=35)
        # plt.xlabel('LLM shots', fontsize='xx-large')
        # plt.ylabel(f'NDCG@{idx + 1}', fontsize=40)
        plt.ylim(0.3, 0.9)
        plt.xlabel('LLM shots', fontsize=30)
        if prompt == 'prompt_2' or prompt == 'prompt_10':
            plt.ylabel(f'{dataset} \n NDCG@{idx + 1}', fontsize=30)
        else:
            plt.ylabel(f'NDCG@{idx + 1}', fontsize=30)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.tight_layout()

        # new labels
        ax.set_xticks(range(len(collated_data.columns)))
        ax.set_xticklabels([get_label(col) for col in collated_data.columns])
        plt.xticks(rotation=90, ha='center')
        # Save the plot as a PDF file with the specified naming convention
        plt.savefig(f'{write_folder}/{prompt}_NDCG_at_{idx + 1}_Bar_Chart.pdf', bbox_inches='tight')
        # Close
        plt.close()
    # Create a new figure to contain just the legend
    legend_fig, legend_ax = plt.subplots(figsize=(3, 2))
    legend = legend_ax.legend(handles=handles, labels=labels, ncol=1, loc='center', fontsize=10)
    legend_ax.axis('off')

    # Step 4: Save the legend as an image file
    legend_fig.tight_layout()
    legend_fig.savefig(f'{write_folder}/legend_ndcg.pdf', bbox_inches='tight')

    plt.close(legend_fig)


def plot_legend(option='case'):
    # Create a separate figure and axis for the legend
    figsize = (15, 1)
    if option == 'synth':
        figsize = (1, 1)
    else:
        if option == 'pareto':
            figsize = (12.5, 2)
    legend_fig, legend_ax = plt.subplots(figsize=figsize)
    lw = 1
    markersize = 8

    if option == 'llm':
        legend_items = [

            Patch(edgecolor='black', label='fairness in representation', hatch='///', facecolor='none'),
            Patch(edgecolor='black', label='fairness in exposure', hatch='.', facecolor='none'),

        ]
        # plt.text(-0.05, 0.47, 'Legend', fontsize=10, weight='bold')
        ncol = 2


    else:
        if option == 'llm-all':
            ncol = 7
            # Create custom legend items using matplotlib.patches.Patch
            legend_items = [

                Patch(edgecolor='black', label='fairness in representation', hatch='///', facecolor='none'),
                Patch(edgecolor='black', label='fairness in exposure', hatch='.', facecolor='none'),
                Patch(edgecolor='black', label='Gemini 1.5 Flash', hatch='', facecolor='#DFF2DF'),
                Patch(edgecolor='black', label='Gemini 1.5 Pro', hatch='', facecolor='#FFCCFF'),
                Patch(edgecolor='black', label='Meta-Llama-3-8B-Instruct', hatch='', facecolor='#00CED1'),
                Patch(edgecolor='black', label='Initial', hatch='', facecolor='#FFC725'),
                Patch(edgecolor='black', label='DetConstSort', hatch='', facecolor='#4682B4')

            ]

        else:

            # option == 'pareto'
            legend_items = [Line2D([0], [0], color='#F00000', lw=lw, marker='*', markersize=markersize,
                                   label='LTR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#F00000', lw=lw, marker='o', markersize=markersize,
                                   label='LTR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#F00000', lw=lw, marker='^', markersize=markersize,
                                   label='LTR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3D6D9E', lw=lw, marker='*', markersize=markersize,
                                   label='FairLTR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3D6D9E', lw=lw, marker='o', markersize=markersize,
                                   label='FairLTR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3D6D9E', lw=lw, marker='^', markersize=markersize,
                                   label='FairLTR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#FFC725', lw=lw, marker='*', markersize=markersize,
                                   label='LTR-FairRR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#FFC725', lw=lw, marker='o', markersize=markersize,
                                   label='LTR-FairRR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#FFC725', lw=lw, marker='^', markersize=markersize,
                                   label='LTR-FairRR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3EC372', lw=lw, marker='*', markersize=markersize,
                                   label='Hidden-FairRR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3EC372', lw=lw, marker='o', markersize=markersize,
                                   label='Hidden-FairRR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='#3EC372', lw=lw, marker='^', markersize=markersize,
                                   label='Hidden-FairRR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='k', lw=lw, marker='*', markersize=markersize,
                                   label='Oblivious-FairRR $g_{dis}\\leftrightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='k', lw=lw, marker='o', markersize=markersize,
                                   label='Oblivious-FairRR $g_{dis}\\rightarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='k', lw=lw, marker='^', markersize=markersize,
                                   label='Oblivious-FairRR $g_{dis}\\leftarrow g_{adv}$',
                                   markerfacecolor='none', linestyle=' '),
                            Line2D([0], [0], color='darkorange', lw=lw, marker='+', markersize=markersize,
                                   label='Oblivious',
                                   markerfacecolor='darkorange', linestyle=' '),
                            Line2D([0], [0], color='#6600CC', lw=lw, marker='+', markersize=markersize,
                                   label='Hidden',
                                   markerfacecolor='#6600CC', linestyle=' ')

                            ]
        ncol = 7

    plt.axis('off')

    # Add the legend to the separate legend axis
    legend_ax.legend(handles=legend_items, loc='center', ncol=ncol, edgecolor='k')
    plt.tight_layout()

    plt.savefig('legend_' + str(option) + '.pdf')

    return


llms = ['gemini-1.5-flash', 'gemini-1.5-pro', 'Meta-Llama-3-8B-Instruct']


def Plot():
    # for prompt in prompts:
    #     plot_ndcgs(prompt=prompt)
    plot_legend(option='llm-all')
    # plot_legend(option='llm')
    # plot_metrics(zero_only=False, non_zero_only=False, llm='gemini-1.5-flash', metric='AveExpR CI-95')
    # plot_metrics(zero_only=False, non_zero_only=False, llm='gemini-1.5-pro', metric='AveExpR CI-95')
    # plot_metrics(zero_only=False, non_zero_only=False, llm='Meta-Llama-3-8B-Instruct', metric='AveExpR CI-95')
    # plot_metrics(zero_only=False, non_zero_only=False, llm='gemini-1.5-flash', metric='NDKL')
    # plot_metrics(zero_only=False, non_zero_only=False, llm='gemini-1.5-pro', metric='NDKL')
    # plot_metrics(zero_only=False, non_zero_only=False, llm='Meta-Llama-3-8B-Instruct', metric='NDKL')
    # plot_metrics(zero_only=True, non_zero_only=False, llm='gemini-1.5-flash', metric='AveExpR CI-95')
    # plot_metrics(zero_only=True, non_zero_only=False, llm='gemini-1.5-flash', metric='NDKL')
    # plot_metrics(zero_only=True, non_zero_only=False, llm='gemini-1.5-pro', metric='AveExpR CI-95')
    # plot_metrics(zero_only=True, non_zero_only=False, llm='gemini-1.5-pro', metric='NDKL')
    # plot_metrics(zero_only=True, non_zero_only=False, llm='Meta-Llama-3-8B-Instruct', metric='AveExpR CI-95')
    # plot_metrics(zero_only=True, non_zero_only=False, llm='Meta-Llama-3-8B-Instruct', metric='NDKL')
    # # SIMPLE PROMPTS

    # for llm in llms:
    #
    #     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
    #                  specific_prompts=['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_NA', 'prompt_NAD'])
    #     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='NDKL',
    #                  specific_prompts=['prompt_2\n', 'prompt_4\n', 'prompt_6\n', 'prompt_NA', 'prompt_NAD'])
    #     # # FEATURE PROMPTS
    #
    #     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
    #                  specific_prompts=['prompt_8\n', 'prompt_10\n', 'prompt_12\n', 'prompt_NA'])
    #     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='NDKL',
    #                  specific_prompts=['prompt_8\n', 'prompt_10\n', 'prompt_12\n', 'prompt_NA'])
    #
    #     # # PROTECTED PROMPTS
    #     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='AveExpR CI-95',
    #                  specific_prompts=['prompt_14\n', 'prompt_16\n', 'prompt_18\n', 'prompt_NA'])
    #     plot_metrics(zero_only=True, non_zero_only=False, llm=llm, metric='NDKL',
    #                  specific_prompts=['prompt_14\n', 'prompt_16\n', 'prompt_18\n', 'prompt_NA'])
