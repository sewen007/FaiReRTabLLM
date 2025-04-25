# FairLLMRank

## Running Experiments with `fairllmrank.py`

Follow the steps below to run experiments using the `fairllmrank.py` script.

### Step-by-Step Instructions

1. Open the `fairllmrank.py` file.
2. Uncomment and execute the code blocks in sequential order (e.g., Step 1, Step 2, Step 3, etc.).
3. If you are using the provided datasets in this repository, **you may skip Steps 1 and 2**.

### Dataset Configuration

- To run experiments for a specific dataset, rename the corresponding settings file:
settings-<dataset>.json → settings.json
- For example, to run experiments on the NBA dataset:
settings-nba.json → settings.json

> **Note:** The code is currently configured to use the **Boston Marathon** dataset. If you're running this dataset, no renaming is necessary.

### ⚠️ Important Warning

When switching between datasets:
- Always revert the previous settings file to its original name before renaming a new one.
- Example:
- Rename `settings.json` back to `settings-nba.json` **before** renaming `settings-bostonmarathon.json` to `settings.json`.

Failure to do so may result in incorrect configuration or overwritten files.
