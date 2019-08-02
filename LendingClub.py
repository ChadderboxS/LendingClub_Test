# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 10:16:30 2019

@author: stilw
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
loans = pd.read_csv('loan.csv', low_memory=False)
df = loans.iloc[:, [2,3,5,6,8,13,15,24,32,38,16]]

# Getting relevant data
df['loan_status'].value_counts(dropna=False)
df['term'].value_counts(dropna=False)

df = df.loc[df['loan_status'].isin(['Fully Paid', 'Charged Off','Default'])]
df = df.loc[df['term'].isin([' 36 months'])]

# remove term now that we have single value
df = df.iloc[:, [0,1,3,4,5,6,7,8,9,10]]

#Create sample for quick reference of data
sample = df.sample(10)

# Check for missing values
def missing_zero_values_table(df):
        zero_val = (df == 0.00).astype(int).sum(axis=0)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mz_table = pd.concat([zero_val, mis_val, mis_val_percent], axis=1)
        mz_table = mz_table.rename(
        columns = {0 : 'Zero Values', 1 : 'Missing Values', 2 : '% of Total Values'})
        mz_table['Total Zero Missing Values'] = mz_table['Zero Values'] + mz_table['Missing Values']
        mz_table['% Total Zero Missing Values'] = 100 * mz_table['Total Zero Missing Values'] / len(df)
        mz_table['Data Type'] = df.dtypes
        mz_table = mz_table[
            mz_table.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns and " + str(df.shape[0]) + " Rows.\n"      
            "There are " + str(mz_table.shape[0]) +
              " columns that have missing values.")
#         mz_table.to_excel('D:/sampledata/missing_and_zero_values.xlsx', freeze_panes=(1,0), index = False)
        return mz_table

missing_zero_values_table(df)

# Missing values are < 1% so we will drop the rows with missing data
df = df.dropna(subset=['dti'])
null_check = df.isnull().sum()


# converting Date
df[["issue_d"]] = pd.to_datetime(df["issue_d"], format = "%b-%Y")


#
#
#Review Data

#
#1 What percentage of loans have been fully paid?
df['loan_status'].value_counts(normalize=True, dropna=False)
#83.9%

#2 Highest rate of default bucketed by grade and year
#add year
df['year'] = df['issue_d'].dt.year
#Create y variable loan_is_bad
df['loan_is_bad'] = (df['loan_status'] != 'Fully Paid').apply(np.uint8)
df.groupby('loan_status')['loan_is_bad'].describe()

default = df.groupby(['year','grade'], as_index=False)['loan_is_bad'].mean()
default.sort_values(by=['loan_is_bad'], ascending=False)
#Loans originating in 2016 with grade 'G' have the highest rate of default at 58%

#3 Average rate of return bucketed by grade and year
rtrn = df.groupby(['year','grade'], as_index=False).agg({
        'funded_amnt':sum,
        'total_pymnt':sum
        })

rtrn['rate'] = (rtrn['total_pymnt']/rtrn['funded_amnt'])**(1/3)-1

# Because we're only using variables available to investors before the loan was funded, issue_d will not be included in the final model. We're keeping it for now just to perform the train/test split later, then we'll drop it.
#Delete variables not available before the loan was funded.
df.drop('total_pymnt', axis=1, inplace=True)
df.drop('funded_amnt', axis=1, inplace=True)


#loan_amnt 
df['loan_amnt'].describe()
#loans range from $500 to $40,000 with a median of $10,000. 
# Plot 
plt.xlabel('loan_amnt')
plt.ylabel('Count')
plt.title('loan_amnt')
plt.hist(df['loan_amnt'])

ax = sns.boxplot(x=df['loan_amnt'], y=df['loan_status'], data=df)
ax.set_title('Loan Amnt by Status')


#int_rate
df['int_rate'].describe()
#Interest rates range from 5.31% to 30.99% (!) with a median of 11.99%.
# Plot 
plt.xlabel('int_rate')
plt.ylabel('Count')
plt.title('int_rate')
plt.hist(df['int_rate'])

ax = sns.boxplot(x=df['int_rate'], y=df['loan_status'], data=df)
ax.set_title('Int Rate by Status')
#Charged off loans have a higher rate than paid ones and Defaults are even higher than charged off.
df.groupby('loan_status')['int_rate'].describe()


#grade
print(sorted(df['grade'].unique()))
# Plot 
plt.xlabel('grade')
plt.ylabel('Count')
plt.title('grade')
plt.hist(df['grade'])

ax = sns.countplot(df['grade'], order=sorted(df['grade'].unique()), color='#5975A4', ax=ax)
ax.set_title('grade')

sns.barplot(x=df['grade'], y=df['loan_status'], data=df, color='#5975A4', saturation=1, ax=ax)
    ax.set_title('Grade by Status')



# annual_inc
df['annual_inc'].describe()
# Annual income ranges from  $16 𝑡𝑜 $10,999,200, with a median of $62,000. Because of the large range of incomes, we should take a log transform of the annual income variable.
df['log_annual_inc'] = df['annual_inc'].apply(lambda x: np.log10(x+1))
df.drop('annual_inc', axis=1, inplace=True)
df['log_annual_inc'].describe()
# Plot 
plt.xlabel('log_annual_inc')
plt.ylabel('Count')
plt.title('log_annual_inc')
plt.hist(df['log_annual_inc'])

ax = sns.boxplot(x=df['log_annual_inc'], y=df['loan_status'], data=df)
ax.set_title('Inc by Status')




#dti
df3['dti'].describe()
#not sure min of -1 and max of 999 make sense
#plotting histogram
plt.figure(figsize=(8,3), dpi=90)
sns.distplot(df3.loc[df3['dti'].notnull() & (df3['dti']<60), 'dti'], kde=False)
plt.xlabel('Debt-to-income Ratio')
plt.ylabel('Count')
plt.title('Debt-to-income Ratio')
#How many of the dti values are "outliers" (above 60)?
(df3['dti']>=60).sum()
#947 doesn't seem like a lot
(df3['dti']>=60).sum()/df3['dti'].sum()
#and is less than a fraction of a percent
df3.groupby('loan_status')['dti'].describe()
#paid off loans tend to have less dti on average

#revol_bal
df3['revol_bal'].describe()
#large range, doing a log transform
df3['log_revol_bal'] = df3['revol_bal'].apply(lambda x: np.log10(x+1))
df3.drop('revol_bal', axis=1, inplace=True)

# Plot 
plt.xlabel('log_revol_bal')
plt.ylabel('Count')
plt.title('log_revol_bal')
plt.hist(df3['log_revol_bal'])

ax = sns.boxplot(x=df3['log_revol_bal'], y=df3['loan_status'], data=df3)
ax.set_title('Revoling Ballance by Status')

df3.groupby('loan_status')['log_revol_bal'].describe()
#Revol Bal is about the same for all statuses 


#Create y variable loan_is_bad
df3['loan_is_bad'] = (df3['loan_status'] != 'Fully Paid').apply(np.uint8)
df3.groupby('loan_status')['loan_is_bad'].describe()
df3.drop('loan_status', axis=1, inplace=True)

#look at issue_d
df3['issue_d'].describe()
#Plot by year 
plt.figure(figsize=(6,3), dpi=90)
df3['issue_d'].dt.year.value_counts().sort_index().plot.bar(color='darkblue')
plt.xlabel('Year')
plt.ylabel('Number of Loans Funded')
plt.title('Loans Funded per Year')
# to make the most realistick test and train data sets We'll form the test set from the most recent 10% of the loans.
loans_train = df3.loc[df3['issue_d'] <  df3['issue_d'].quantile(0.9)]
loans_test =  df3.loc[df3['issue_d'] >= df3['issue_d'].quantile(0.9)]
#Check that we properly partitioned the loans:
print('Number of loans in the partition:   ', loans_train.shape[0] + loans_test.shape[0])
print('Number of loans in the full dataset:', df3.shape[0])
#confirm test size of ~10%
loans_test.shape[0] / df3.shape[0]
#Look at distribution of dates in test and train
loans_train['issue_d'].describe()
loans_test['issue_d'].describe()
#Delete the issue_d variable, because it was not available before the loan was funded.
loans_train.drop('issue_d', axis=1, inplace=True)
loans_test.drop('issue_d', axis=1, inplace=True)

#Separate the predictor variables from the response variable:
y_train = loans_train['loan_is_bad']
y_test = loans_test['loan_is_bad']

X_train = loans_train.drop('loan_is_bad', axis=1)
X_test = loans_test.drop('loan_is_bad', axis=1)

#Test predictive value of variables
#On the training set, compute the Pearson correlation, 𝐹 -statistic, and 𝑝 value of each predictor with the response variable charged_off.

linear_dep = pd.DataFrame()

#Pearson correlations:

for col in X_train.columns:
    linear_dep.loc[col, 'pearson_corr'] = X_train[col].corr(y_train)
linear_dep['abs_pearson_corr'] = abs(linear_dep['pearson_corr'])

#𝐹-statistics:

from sklearn.feature_selection import f_classif
for col in X_train.columns:
    mask = X_train[col].notnull()
    (linear_dep.loc[col, 'F'], linear_dep.loc[col, 'p_value']) = f_classif(pd.DataFrame(X_train.loc[mask, col]), y_train.loc[mask])

#Sort the results by the absolute value of the Pearson correlation:

linear_dep.sort_values('abs_pearson_corr', ascending=False, inplace=True)
linear_dep.drop('abs_pearson_corr', axis=1, inplace=True)

#Reset the index:

linear_dep.reset_index(inplace=True)
linear_dep.rename(columns={'index':'variable'}, inplace=True)

#View the results for the top 20 predictors most correlated with charged_off:

linear_dep.head(20)




## Model Training and Testing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV

#Logistic regression with SGD training
from sklearn.linear_model import SGDClassifier

#The machine learning pipeline:
pipeline_sgdlogreg = Pipeline([
    ('imputer', Imputer(copy=False)), # Mean imputation by default
    ('scaler', StandardScaler(copy=False)),
    ('model', SGDClassifier(loss='log', max_iter=1000, tol=1e-3, random_state=1, warm_start=True))
])

#A small grid of hyperparameters to search over:
param_grid_sgdlogreg = {
    'model__alpha': [10**-5, 10**-2, 10**1],
    'model__penalty': ['l1', 'l2']
}

#Create the search grid object:
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

#Conduct the grid search and train the final model on the whole dataset:
grid_sgdlogreg.fit(X_train, y_train)

#Mean cross-validated AUROC score of the best model:
grid_sgdlogreg.best_score_
#0.6710893359343707

#Best hyperparameters:
grid_sgdlogreg.best_params_


#
#
#
#Random forest classifier
from sklearn.ensemble import RandomForestClassifier

#The machine learning pipeline:
pipeline_rfc = Pipeline([
    ('imputer', Imputer(copy=False)),
    ('model', RandomForestClassifier(n_jobs=-1, random_state=1))
])

#Set hyperparameter
param_grid_rfc = {
    'model__n_estimators': [50] # The number of randomized trees to build
}

#Create the search grid object:
grid_rfc = GridSearchCV(estimator=pipeline_rfc, param_grid=param_grid_rfc, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

#Conduct the grid search and train the final model on the whole dataset:
grid_rfc.fit(X_train, y_train)

#Mean cross-validated AUROC score of the random forest:
grid_rfc.best_score_
#0.5887085159483046
#Not quite as good as logistic regression


#
#
#
#k-nearest neighbors
from sklearn.neighbors import KNeighborsClassifier

#The machine learning pipeline:
pipeline_knn = Pipeline([
    ('imputer', Imputer(copy=False)),
    ('scaler', StandardScaler(copy=False)),
    ('lda', LinearDiscriminantAnalysis()),
    ('model', KNeighborsClassifier(n_jobs=-1))
])

#Set hyperparameter
param_grid_knn = {
    'lda__n_components': [3, 9], # Number of LDA components to keep
    'model__n_neighbors': [5, 25, 125] # The 'k' in k-nearest neighbors
}

#Create the search grid object:
grid_knn = GridSearchCV(estimator=pipeline_knn, param_grid=param_grid_knn, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)

#Conduct the grid search and train the final model on the whole dataset:
grid_knn.fit(X_train, y_train)

#Mean cross-validated AUROC score of the random forest:
grid_knn.best_score_
#0.6592043010694129
#Almost as good as logistic regression

#Best hyperparameters:
grid_knn.best_params_


#
#
#
#Tune hyperparameters on the chosen model
#The three models performed quite similarly according to the AUROC:

print('Cross-validated AUROC scores')
print(grid_sgdlogreg.best_score_, '- Logistic regression')
print(grid_rfc.best_score_, '- Random forest')
print(grid_knn.best_score_, '- k-nearest neighbors')

#Model Selected: Logistic regression
#was the most accurate and the fastest

#Hypertuning
param_grid_sgdlogreg = {
    'model__alpha': np.logspace(-4.5, 0.5, 11), # Fills in the gaps between 10^-5 and 10^1
    'model__penalty': ['l1', 'l2']
}

print(param_grid_sgdlogreg)
#Create the search grid object:
grid_sgdlogreg = GridSearchCV(estimator=pipeline_sgdlogreg, param_grid=param_grid_sgdlogreg, scoring='roc_auc', n_jobs=-1, pre_dispatch='2*n_jobs', cv=5, verbose=1, return_train_score=False)
#Conduct the grid search and train the final model on the whole dataset:
grid_sgdlogreg.fit(X_train, y_train)
#Mean cross-validated AUROC score of the best model:
grid_sgdlogreg.best_score_
#0.6711479758192964
#a little better
#Best hyperparameters:
grid_sgdlogreg.best_params_


#
#
#
#Test set evaluation
from sklearn.metrics import roc_auc_score

y_score = grid_sgdlogreg.predict_proba(X_test)[:,1]

roc_auc_score(y_test, y_score)
#0.6831697082273466
#The test set AUROC score is higher than the cross-validated score!! :)

#Set Confidence interval
y_score2 = np.digitize(y_score,np.arange(0,1,0.5)) -1

#Confusion Matrix
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 

results = confusion_matrix(y_test, y_score2) 
print ('Confusion Matrix :')
print(results) 
print ('Accuracy Score :',accuracy_score(y_test, y_score2) )
print ('Report : ')
print (classification_report(y_test, y_score2) )










