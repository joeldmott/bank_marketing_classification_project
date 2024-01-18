# Bank Marketing Campaign Classification Project

This project optimizes three different kinds of classification models (Decision Tree, StatsModels Logit, and Scikit-Learn Logit) to find the best one for predicting which customers to target in a marketing campaign for a bank.

## business scenario
The data comes from a bank's marketing campaign where existing customers were contacted by phone in an attempt to have them open a **term deposit**. Unlike a checking account, term deposit funds cannot be withdrawn without penalties for a predetermined amount of time. In return, the client recieves higher interest rates. In this case, the bank has found that the most effective contact method is phone calls. They pay a call center to do the actual calling.

During the last campaign, only 11.7% of customers wound up opening a term deposit while 88.3% did not. 

![image](https://github.com/joeldmott/bank_marketing_classification_project/assets/51928528/690246a7-8dc6-488c-8f3e-f7371b9f8f1d)

My goal is to help **increase the efficiency of the next marketing campaign by providing a model which selects the most likely-to-deposit customers, thereby reducing wasted call center costs.** 

## project overview

For each model, I establish a baseline instance, then experiment with over & under-sampling techniques as well as hyperparamater tuning. 

These models are evaluated with confusion matrices, recall scores and area-under-the-curve (AUC) scores. **I priveledge recall the most because new term deposits will net more than a wasted phone call (so false negatives are more of a concern than false positives). However, I still want to keep call center costs in check, so AUC and confusion matrices are also important.**

Keeping the stakeholder's specific interests in focus while also understanding the nature and distribution of the data, I find that a decision tree model with a specific undersampling strategy (50%) and max depth parameter (max_depth=6) makes for the best predictions that decrease wasted calls. 

## dataset

This project makes use of a [Kaggle dataset from a bank's term deposit marketing campaign](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets/data) that contains just under 50,000 records, each representing an existing customer who was contacted as a part of this marketing effort. 

It also contains fifteen attributes and one target column, which indicates whether the customer opened a term account. The attributes include age, education level, current checking account balance, whether they've opened such an account before or have another loan with the bank, whether they've defaulted on a loan before, and data on how previous marketing efforts went for the client. 

This data is pretty clean and comprehensive (no NaN's, although there are some features with "unknown" categories). However, I found that some predictors weren't helpful and added noise to the models, so columns pertaining to dates and durations of calls were eliminated.

## modeling process

I begin with a decision tree modeling process because I thought entropy would be a helpful first insight into how these disparate features relate to whether someone opened a term deposit. 

The basic process here is the same as with every model: establish a baseline model (no over or under-sampling techniques, no hyperparameter specification) and evaluate it via a confusion matrix, recall, and AUC scores. Next, gradual improvments are made via the following methods:

- oversampling the minority class (the "yes"/"1" class in the target column as opposed to the majority "no"/"0" class) with Scikit-Learn's RandomOverSampler from 110% up to 150% oversampling
- undersampling the majority class (the "no"/"0" class) with Scikit-Learn's RandomUnderSampler from 90% down to 50% undersampling
- combining undersampling & oversampling
- oversampling with SMOTE
- hyperparameter tuning

These models differ in some ways, so the process isn't always identical. After optimizing all three, the optimized decision tree model wound up with the best predictions.

## model evaluations & stakeholder recommendations

If we really want to balance out term deposits and wasted call center costs (but favor term deposits to an extent), the best model is the aforementioned **decision tree** (undersampled by 50% and with a max_depth param of 6). 

![image](https://github.com/joeldmott/bank_marketing_classification_project/assets/51928528/85423fe3-2070-46af-8e60-76f05fd37fc1)

It results in 7.3% true positives and 4.4% false negatives (keeping in mind that the data had 11.7% positives in the first place, this 7.3-to-4.4 ratio means it's predicting positives 62% correctly). It also shows 81.2% true negatives and 7.1% false positives (predicting negatives 92% correctly). The lower positive rate is due to our attempts to keep call center costs under control as well.

The **StatsModels Logit** predictions aren't as good as the other two, but it does prove helpful with its coefficients. Specifically, we learn that repeated contact attempts during a campaign strongly correlate with a "no" result, as well as confirming that previous depositors and high checking account balances are by far the best predictors for a new term deposit. 

Below, I list each statistically significant coefficient's value, with the vertical line showing a coefficient value of 1 (coefficients near the line have little to no real impact on term deposits). The "campaign_normalized" coefficient is the aforementioned negative (below 1) impact in terms of the amount of times a client was contacted over the previous campaign. It's the only non-positive one that seems significant. The other two strong positive coefficients ("balance" and "previous-outcome-as-a-success") definitely stand out.

![image](https://github.com/joeldmott/bank_marketing_classification_project/assets/51928528/0e608945-d8ff-4897-bbad-b1a9710ebcfb)

Finally, if things change someday and call center costs really aren't too bad or the stakeholder isn't concerned with them nearly as much as term deposits, there's a Scikit-Learn Logit model that results in 9.6% true positives and 2.1% false negatives (predicting positives 82% correctly) while showing 73.7% true negatives and 14.6% false positives (predicting negatives 84% correctly). In other words, it winds up with a *lot* more wasted calls, but the most new term deposits overall.

## conclusions 

Decision tree models tend to make better predictions on large datasets with unrelated features and this data is no exception; an optimized decision tree model worked better than a StatsModels & Scikit-Learn Logit model did. However, the StatsModels Logit relayed the following about contacting clients in future marketing campaigns for this stakeholder:

- the odds of a successful outcome for a client **decrease by 0.007%** for repeated contact attempts 
- a previous depositor has **869% higher odds** of making another deposit compared to a client who has not done so before
- higher checking account balances **greatly** increase the odds of a successful term deposit outcome (this was a normalized feature, so exact odds are less interpretable here)

As I continue learning, it looks like the next steps involve Pipelines and Random Forests, which streamline the operations put into place here. Nonetheless, manually performing these procedures helped reinforce how they help to improve data predictions based on a stakeholder's specific needs.

## repository structure:
├── notebook checkpoints  

├── dataset_files

──── bank_marketing_train.csv

──── bank_marketing_test.csv 

├── README.md

├── presentation.pdf

├── project_notebook.ipynb

├── project_notebook.pdf

## contact

Please contact me with any questions or comments at joel.mott8@gmail.com
