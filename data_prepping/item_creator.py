import json
import os

import pandas as pd

with open('../settings.json', 'r') as f:
    settings = json.load(f)
experiment_name = os.path.basename(settings["READ_FILE_SETTINGS"]["PATH"]).split('.')[0]


def create_items():
    """creates items with format item = {
    'query': 'You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to '
             'predict their success in the program. The school wants to rank the applicants using their UGPA score '
             'and LSAT scores.  Without including explanations, rank the following applicants in a predicted order of '
             'success in the program. Return your ranked results in the following json only {student_id: the student\'s'
             'ID,gender: the student gender}:',
    'hits': [
        {'content': 'Student ID: 18739 (Gender: female, UGPA: 3.7, LSAT: 44.0)'},
        {'content': 'Student ID: 8742 (Gender: male, UGPA: 3.2, LSAT: 20.0)'},
        {'content': 'Student ID: 17967 (Gender: male, UGPA: 3.2, LSAT: 34.5)'},
        {'content': 'Student ID: 13964 (Gender: male, UGPA: 3.7, LSAT: 32.0)'}
    ]
}"""
    df = pd.read_csv('../Datasets/' + experiment_name + '/' + experiment_name + '_test_data_for_LLM.csv')[0:100]

    # Hardcoded query
    query = ('You are the admissions ranking system for a Law school; their goal is to rank a list of applicants to '
             'predict their success in the program. The school wants to rank the applicants using their UGPA score '
             'and LSAT scores. Without including explanations, rank the following applicants in a predicted order of '
             'success in the program.')
    if experiment_name == 'dummy':
        query = 'You are a ranking system for a list of numbers. Rank the following numbers in descending order:'

    # Creating the hits list from the DataFrame
    hits = []
    if experiment_name == 'LAW':
        for index, row in df.iterrows():
            content = f"Student ID: {row['unique_id']} (Gender: {row['Gender']}, UGPA: {row['UGPA']}, LSAT: {row['LSAT']})"
            hits.append({'content': content})
    else:
        for index, row in df.iterrows():
            content_parts = [f"Unique ID: {row['doc_id']}"]
            for column in df.columns:
                if column != 'doc_id':
                    content_parts.append(f"{column}: {row[column]}")
            content = ", ".join(content_parts)
            hits.append({'content': content})

    # Creating the final JSON object
    result = {
        'query': query,
        'hits': hits
    }
    return result


print(create_items())
