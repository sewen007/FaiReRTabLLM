import json
import os
from pathlib import Path

experiment_name = 'LAW'
model_name = 'ListNet'

with open('settings.json', 'r') as f:
    settings = json.load(f)

sample_sizes = settings["GENERAL_SETTINGS"]["sample_sizes"]
shots = settings["GENERAL_SETTINGS"]["shots"]


def rank_ltr(size=50, shot=1):
    write_path = Path(f'Datasets/{experiment_name}/Ranked/{model_name}/rank_size_{size}/shot_{shot}')
    if not os.path.exists(write_path):
        os.makedirs(write_path)
