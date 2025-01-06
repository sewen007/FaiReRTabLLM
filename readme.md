The test sets for the LLMs are the same

order of running the code:
# clean - clean.py if needed
# prep
# ltr.py - needed to rank the data with LTR 
# llmrank
# train
# rank
# calculate_metrics
# plot in data_viz
# visualize_tables


prompt_0 is for the neutral prompt
prompt_1 has no fairness 
prompt_2 incorporates fairness with this text: "while incorporating fairness with respect to sex"
prompt_4 incorporates fairness with this text: "while incorporating fairness (equal opportunity) with respect to sex, where female is the disadvantaged group"
prompt_5 is the same as prompt_1


Data 
- The data is in the Datasets Folder        

Tests
- The tests are in the Tests Folder in the Datasets Folder
- We have 5 test sets each of equal percentage in each demographic group (e.g. 50% male, 50% female)


Question 1: Will FairLLM rank better if given previously ranked data than if given unranked data? 
- We will compare the performance of FairLLM on the test sets when given previously ranked data and when given unranked data.
- Pipelines for each case: 
    - FairLLM with previously ranked data: data is taken from Tests/LTR_ranked 
    - FairLLM with unranked data: data is taken from Tests

# important- we are currently using protected attributes default calculated from the entire test set instead of the smaller ranking sets