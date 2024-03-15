#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:37:43 2024

@author: 
    yichenhsu
    Kwok Wing, Tang
    Atefeh, Arabi
    Kymecia Alana, Rodrigues
    Tessa, Mathew
"""

'''
pip install scikit-learn
pip install pandas
pip install matplotlib
pip install spyder-kernels==2.5.*
pip install imblearn
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''
Data Exploration
'''
#Loading data
df = pd.read_csv('KSI.csv')
#Knowing the data
print(df.head(3))
print(df.info())
print("Target column unique value: ", df['ACCLASS'].unique())
print(df.describe().T)
columns = list(df.columns)
def count_value(dataframe, columns):
    total_rows = dataframe.shape[0]
    return pd.DataFrame({
        "Missing": dataframe[columns].isna().sum(),
        "MPercentage": round(dataframe[columns].isna().mean()*100,2),
        "Unique": dataframe[columns].nunique(),
        "UPercentage": round((dataframe[columns].nunique()/total_rows)*100,2)
    })
data = count_value(df, columns)

'''
Data Preprocessing
'''
#columns list with greater than 80% missing values just drop the columns
to_drop_cols = data[(data.MPercentage > 80) & (data.Unique > 1)].index.tolist()
#columns list with less than 3% missing values (don't impute)-just drop the rows
to_drop_rows = data[(data['MPercentage'] < 3) & (data['MPercentage'] > 0)].index.tolist()
to_drop_rows.remove('ACCLASS')
df_g6 = df.dropna(subset = to_drop_rows)
#columns list with one unique value which is yes
to_fill_no_cols = data[(data.Unique == 1)].index.tolist()
for col in to_fill_no_cols:
    #print(df_g6[col].unique())
    df_g6[col] = np.where(df_g6[col] == 'Yes', 1, 0)
#columns list with unique values
for col in data[(data.UPercentage == 100)].index.tolist():
    to_drop_cols.append(col)
#columns list for non-binary columns with missing values - replace with nan
fill_nan_cols = data[(data.Missing > 0)].index.tolist()
for col in fill_nan_cols:
    df_g6[col].fillna(value=np.nan)

df_g6 = df_g6.drop(columns=to_drop_cols)
df_g6['ACCLASS'] = np.where(df_g6['ACCLASS'] != 'Fatal', 0, 1)
df_g6.INVAGE.replace('unknown', np.nan, inplace=True)

def age_transform(text):
    if isinstance(text, str):
        age_range = text.split(' ')
        if len(age_range) == 3:
            return (int(age_range[0]) + int(age_range[2])) / 2
    return np.nan

df_g6['INVAGE'] = df_g6['INVAGE'].apply(age_transform)

#Add day of week
dates_column = pd.to_datetime(df_g6['DATE'])
df_g6['WEEKDAY'] = dates_column.dt.day_name()
df_g6['MONTH'] = pd.to_datetime(df_g6['DATE']).dt.month
df_g6['DAY'] = pd.to_datetime(df_g6['DATE']).dt.day

#handle time
bins = [0, 500, 1200, 1700, 2100, 2400]
names = ['Night', 'Morning', 'Afternoon', 'Evening', 'Night2']
df_g6['TIMERANGE'] = pd.cut(df_g6['TIME'], bins, labels=names)
df_g6.TIMERANGE.replace(['Night2'], ['Night'], inplace=True)

'''
Data Visualization
'''
def split_time(text):
    return text.split(' ')[0]

#Correlation Matrix
#if it is 1, they are duplicate columns
df['TARGET'] = np.where(df['ACCLASS'] != 'Fatal', 0, 1)
corr_matrix = df.corr(numeric_only=True)
corr_matrix['TARGET'].sort_values(ascending=False)
sns.heatmap(corr_matrix)

#Drop duplicate columns
to_drop_dup_cols = ['X','Y']
df_g6 = df_g6.drop(columns=to_drop_dup_cols)

#Target columns vs INVTYPE
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="ACCLASS", hue="INVTYPE", palette="Set2")
plt.xlabel('Fatal Level')
plt.ylabel('Count')
plt.title('Involvement Type')
plt.show()
#Target columns vs TIME
plt.figure(figsize=(10, 6))
sns.countplot(data=df_g6, x="TIMERANGE", hue="ACCLASS")
plt.xlabel('TIMERANGE')
plt.ylabel('Count')
plt.title('TIMERANGE Type')
plt.show()
############ Chi-square
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# Assuming 'df' is your DataFrame and it has been loaded properly.

# The column to test against others
column_to_test = "ACCLASS"

# List of other columns to test
cols = df_g6.columns #All columns
num_cols = df_g6._get_numeric_data().columns #Numeric columns
other_columns = list(set(cols) - set(num_cols)) #Categorical columns
if "ACCLASS" in other_columns:
    other_columns.remove("ACCLASS")
# Initialize a list to store Chi-square statistics
chi_square_stats = {}

# Perform Chi-square test for 'ACCLASS' against each of the other columns
for col in other_columns:
    # Create a contingency table
    contingency_table = pd.crosstab(df_g6[column_to_test], df_g6[col])
    
    # Perform the Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    
    # Store the Chi-square statistic
    chi_square_stats[col] = chi2
    
    # Output the result
    print(f"Chi-square test between {column_to_test} and {col}:")
    print(f"Chi-square Statistic: {chi2}, p-value: {p}\n")

#sort
chi_square_stats = dict(sorted(chi_square_stats.items(), key=lambda item: item[1]))

# Plotting the Chi-square statistics
plt.figure(figsize=(10, 8))
variables_names =list(chi_square_stats.keys())
plt.barh(variables_names, chi_square_stats.values(), color='skyblue')
plt.ylabel('Features')
plt.xlabel('Chi-square Statistic')
plt.title('Chi-square test')
plt.show()
#################

'''
Prepared Data
'''
#Checking
columns = list(df_g6.columns)
data_g6 = count_value(df_g6, columns)
print(data_g6)

df_g6 = df_g6.drop(['HOOD_158','NEIGHBOURHOOD_158','HOOD_140','NEIGHBOURHOOD_140'], axis = 1 )
corr = df_g6.corr(numeric_only=True)['ACCLASS'].sort_values(ascending=False)
corr.abs().head(50)

target = 'ACCLASS'
features = ['INVTYPE', 'INVAGE', 'AUTOMOBILE', 'WEEKDAY', 'MONTH', 'DAY', 'TIMERANGE', 'LIGHT', 
            'LATITUDE', 'LONGITUDE', 'DISTRICT', 'VISIBILITY', 'RDSFCOND', 'VEHTYPE']

final_df = df_g6[[target] + features].copy()

print(final_df.info())


'''
Data modelling
'''
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=53)

for train_idx, test_idx in sss.split(final_df, final_df[target]):
    train_set = final_df.iloc[train_idx]
    test_set = final_df.iloc[test_idx]

X_train = train_set.drop(target, axis=1)
y_train = train_set[target].copy()
X_test = test_set.drop(target, axis=1)
y_test = test_set[target].copy()

'''
Recursive Feature Elimination
'''
'''
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Create a base classifier
logreg = LogisticRegression()

# Create the RFE object
rfe = RFE(estimator=logreg, n_features_to_select=10, step=1)

# Fit the RFE object to the training data
rfe.fit(X_train, y_train)

# Get the column indices selected by RFE
selected_features = X_train.columns[rfe.support_]

# Print the selected features
print("Selected features:", selected_features)

# Update the training and testing data with the selected features
X_train = X_train[:, rfe.support_]
X_test = X_test[:, rfe.support_]
'''

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_features = ['INVAGE']
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())

    ])

cat_features = ['INVTYPE','WEEKDAY','TIMERANGE','LIGHT','DISTRICT','VISIBILITY','RDSFCOND','VEHTYPE']
cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas"))
    
    ])

#combine numeric & cat pipeline using column transformer
preprocessor = ColumnTransformer([
    ('num', numeric_pipe, numeric_features),
    ('cat', cat_pipe, cat_features)
    
    ])

#create a full pipeline
full_pipeline = Pipeline([
    ('preprocessor', preprocessor)
    
    ])
#print(X_train['ACCLASS'].value_counts())
X_train_prepared = full_pipeline.fit_transform(X_train)
#x_train_prepared
X_test_prepared = full_pipeline.transform(X_test)
#print(X_train_prepared['ACCLASS'].value_counts())

##################### Balancing

from sklearn.utils import resample

###################


###################### SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_SMOTE, y_train_SMOTE = sm.fit_resample(X_train_prepared, y_train.ravel()) 

###################


import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
kernels = ['linear', 'rbf', 'poly', 'sigmoid']

for kernel in kernels:
    clf = SVC(kernel=kernel, C=0.1 if kernel == 'linear' else 1.0).fit(X_train_prepared, y_train)
    y_pred_train = clf.predict(X_train_prepared)
    y_pred_test = clf.predict(X_test_prepared)
    print(f"Kernel: {kernel} ")
    print("Accuracy for train data: ", accuracy_score(y_train, y_pred_train))
    print("Accuracy for test data: ", accuracy_score(y_test, y_pred_test))
    print("Confusion matrix: \n", confusion_matrix(y_test, y_pred_test))
    print('\n')


#10. Print accuracy score for testing set
'''
y_pred = clf_linear_KwokWing.predict(X_test_prepared)
acc = accuracy_score(X_test_prepared, y_pred)
print("Accuracy Score(Testing data): ", acc)


#11. Generate the accuracy matrix.
import seaborn as sns
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{cm}")
plt.title('Confusion Matrix - Kernel: Linear', fontsize = 15)
sns.heatmap(cm, annot = True)
plt.show()
'''
#############################

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
# Separate input features (X) and target variable (y)
#y = df.balance
#X = df.drop('balance', axis=1)
 
# Train model
clf_4 = RandomForestClassifier(random_state=123)
clf_4.fit(X_train_SMOTE, y_train_SMOTE)
 
# Predict on training set
pred_y_4 = clf_4.predict(X_test_prepared)
 
# Is our model still predicting just one class?
print( np.unique( pred_y_4 ) )
# [0 1]
 
# How's our accuracy?
print( accuracy_score(y_test, pred_y_4) )
# 1.0
 
# What about AUROC?
#prob_y_4 = clf_4.predict_proba(X_test)
#prob_y_4 = [p[1] for p in prob_y_4]
#print( roc_auc_score(y_test, prob_y_4) )

