# Bank Marketing Campaign Classification Project

## business scenario
The data is from a bank marketing campaign where existing customers were contacted by phone in an attempt to have them open a **term deposit**. Unlike a checking account, term deposit funds cannot be withdrawn without penalties for a predetermined amount of time. In return, the customer is offered higher interest rates. The bank knows the most effective contact method is via phone calls; they pay a call center to do the actual calling.

Only 12% of customers wound up opening a term deposit while 88% did not. My goal is to help **increase the efficiency of the next marketing campaign by providing a model which selects the most likely-to-deposit customers, thereby also reducing wasted call center costs.** 

## project overview

Specifically, this project optimizes three different kinds of classification models (Decision Tree, StatsModels Logit, and Scikit-Learn Logit) to find the best one for predicting which customers to target in a bank's marketing campaign. For each model, I establish a baseline instance, then experiment with over & under-sampling techniques as well as hyperparamater tuning. 

These models are evaluated with confusion matrices, recall scores and area-under-the-curve (AUC) scores. **I priveledge recall the most because new term deposits will net more than a wasted phone call (so false negatives are a bigger deal than false positives). However, I still want to keep call center costs in check, so AUC and confusion matrices are also important.**

Keeping the stakeholder's specific interests in focus while also understanding the nature and distribution of the data, I find that a decision tree model with a specific undersampling strategy (50%) and max_depth parameter (6) makes for the best predictions that increase term deposits while decreasing wasted calls. 

## dataset

This project makes use of a [Kaggle dataset from a bank's term deposit marketing campaign](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets/data) that originally comes in two files: one is the training set and the other is the test set. Altogether, they contain just under 50,000 records, each representing an existing customer who was contacted as a part of this marketing effort. 

It also contains fifteen attributes and one target column, which indicates whether the customer opened a term account. The attributes include age, education level, current checking account balance, whether they've opened such an account before or have another loan with the bank, whether they've defaulted on a loan before, and data on how previous marketing efforts went for the client. 

This data is pretty clean and comprehensive (no NaN's, although there are some features with "unknown" categories). However, I found that some predictors weren't helpful and added noise to the models, so columns pertaining to dates and durations of calls were eliminated.

## modeling process

I begin with a decision tree modeling process because I think it's the best way to untangle these attributes since they're so different. Entropy will be a helpful way to gain first insights into how these features relate to whether someone opened a term deposit. 

The basic process here is the same as with every model: establish a baseline model (no over or under-sampling techniques, no hyperparameter specification) and evaluate it via a confusion matrix, recall, and AUC scores. Next, gradual improvments are made via the following methods:

- oversampling the minority class (the "yes"/"1" class in the target column as opposed to the majority "no"/"0" class) with Scikit-Learn's RandomOverSampler from 110% up to 150% oversampling
- undersampling the majority class (the "no"/"0" class in the target column as opposed to the minority yes"/"1" class) with Scikit-Learn's RandomUnderSampler from 90% down to 50% undersampling
- oversampling with SMOTE
- combining undersampling & oversampling
- hyperparameter tuning

These models differ in some ways, so the process isn't always identical, but some patterns emerge among all three:

## evaluation

- Undersampling tends to make for better predictions than oversampling. SMOTE and a combination of over/undersampling never works as well.
- The StatsModels Logit predictions aren't as good as the other two, but it does prove helpful with its coefficients. Specifically, we learn that repeated contact attempts during a campaign strongly correlate with a "no" result, as well as confirming that previous depositors and high checking account balances are by far the best predictors for a new term deposit.
- If we really want to balance out term deposits and wasted call center costs (but favor term deposits to an extent), the best model is the aforementioned decision tree (undersampled by 50% and with a max_depth param of 6). It results in 7.3% true positives and 4.4% false negatives (so it's predicting positives 62% correctly). It also shows 81.2% true negatives and 7.1% false positives (predicting negatives 92% correctly). The lower positive rate is due to inherent class imbalance and the fact that we're trying to keep call center costs under control.
- If call center costs really aren't too bad and the stakeholder isn't concerned with them nearly as much as term deposits, there's a Scikit-Learn Logit model that results in 9.6% true positives and 2.1% false negatives (predicting positives 82% correctly) while showing 73.7% true negatives and 14.6% false positives (so it predicts negatives 84% correctly). In other words, it winds up with a *lot* more wasted calls, but the most new term deposits overall.

## summary 

As I continue learning, it looks like the next steps involve Pipelines and Random Forests, which streamline the operations put into place here. Nonetheless, manually performing these procedures helped reinforce how they help to improve data predictions based on a stakeholder's specific needs.

## Repository Structure:
├── notebook checkpoints  

├── dataset_files

──── bank_marketing_train.csv

──── bank_marketing_test.csv 

├── README.md

├── presentation.pdf

├── project_notebook.ipynb

├── project_notebook.pdf
