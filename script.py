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

t_num_cols = ['X', 'Y', 'INDEX_', 'ACCNUM', 'YEAR', 'TIME', 'WARDNUM', 'LATITUDE', 'LONGITUDE', 'FATAL_NO', 'ObjectId']
df[t_num_cols].hist(bins=30, figsize=(15,10))
plt.show()

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

#VEHTYPE vs occurence of accidents
#Counts occurrences of each unique value in the 'VEHTYPE' column
vehtype_counts = df_g6['VEHTYPE'].value_counts()
#Plotting
plt.figure(figsize=(10, 5))
sns.barplot(x=vehtype_counts.index, y=vehtype_counts.values, palette="viridis")
plt.title('Accidents by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=90)
plt.show()

#WEEKDAY vs occurence of accidents
#Counts accidents per day of the week, ensuring days are in the correct order
weekday_counts = df_g6['WEEKDAY'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
])
#Plotting
plt.figure(figsize=(10, 5))
sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette="Blues")
plt.title('Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=40)
plt.show()

#LOCCOORD vs occurence of accidents
#Counts occurrences of each unique value in the LOCCOORD column
loccoord_counts = df_g6['LOCCOORD'].value_counts()
#Plotting
plt.figure(figsize=(9, 5))
sns.barplot(x=loccoord_counts.index, y=loccoord_counts.values, palette="Spectral")
plt.title('Accidents by Location Coordinates')
plt.xlabel('Location')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=90)
plt.show()

# Visualize the relationship between "RDSFCOND" (road surface condition) and the target variable "ACCLASS"
plt.figure(figsize=(10, 6))
sns.countplot(data=df_g6, x="RDSFCOND", hue="ACCLASS")
plt.xlabel('Road Surface Condition')
plt.ylabel('Count')
plt.title('Road Surface Condition vs. Fatal Level')
plt.xticks(rotation=55)
plt.show()


# Visualize the relationship between "LIGHT" (light condition) and the target variable "ACCLASS"
plt.figure(figsize=(10, 6))
sns.countplot(data=df_g6, x="LIGHT", hue="ACCLASS")

plt.xlabel('Light Condition')
plt.ylabel('Count')
plt.title('Light Condition vs. Fatal Level')
plt.xticks(rotation=55)
plt.show()

# Visualize the relationship between "INVAGE" (age of involved party) and the target variable "ACCLASS"
plt.figure(figsize=(10, 6))
sns.histplot(data=df_g6, x="INVAGE", hue="ACCLASS", bins=20, kde=True)
plt.xlabel('Age of Involved Party')
plt.ylabel('Count')
plt.title('Age of Involved Party vs. Fatal Level')

############################################
#Kym

# Visualization for Relationship between Numerical feature LATITUDE AND TARGET
'''
plt.figure(figsize=(10, 6))
sns.countplot(data=df_g6, x="LATITUDE", hue="ACCLASS")
plt.xlabel('Latitude')
plt.ylabel('ACCLASS')
plt.title('LATITUDE vs. Fatal Level')
plt.xticks(rotation=55)
plt.show()
'''

#Year - Accidents in different years

# Visualize the relationship between YEAR and the target variable "ACCLASS"
plt.figure(figsize=(10, 6))
plt.title('Accidents that occurred by year')
sns.countplot(data=df_g6, x="YEAR", hue="ACCLASS")



#Months - Accidents in different months
the_months = df_g6['MONTH'].value_counts()
the_months = the_months.sort_index(axis=0)
the_months.index = [
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
]
#Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=the_months.index, y=the_months.values, palette="Blues")
plt.title('Accidents that occurred by month')
plt.xlabel('Month of the year')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=40)
plt.show()

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

############ Chi-square
import pandas as pd
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# The column to test against others
column_to_test = "ACCLASS"

# List of other columns to test
cols = df_g6.columns #All columns
num_cols = df_g6._get_numeric_data().columns #Numeric columns
other_columns = list(set(cols) - set(num_cols)) #Categorical columns
other_columns.extend(['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY', 'MONTH', 'DAY'])
if "STREET1" in other_columns: other_columns.remove("STREET1")
if "STREET2" in other_columns: other_columns.remove("STREET2")
if "INJURY" in other_columns: other_columns.remove("INJURY")
if "DATE" in other_columns: other_columns.remove("DATE")
if "ACCLASS" in other_columns: other_columns.remove("ACCLASS")

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
    #print(f"Chi-square test between {column_to_test} and {col}:")
    #print(f"Chi-square Statistic: {chi2}, p-value: {p}\n")

#sort
chi_square_stats = dict(sorted(chi_square_stats.items(), key=lambda item: item[1]))
# Plotting the Chi-square statistics
plt.figure(figsize=(10, 8))
variables_names =list(chi_square_stats.keys())
plt.barh(variables_names, chi_square_stats.values(), color='skyblue')
plt.ylabel('Features')
plt.xlabel('Chi-square statistic')
plt.title('Chi-square test')
plt.show()

highest_n_items = 12
chi_square_selected_features = {k: chi_square_stats[k] for k in list(chi_square_stats)[0-highest_n_items:]}
chi_square_selected_features = list(chi_square_selected_features.keys())
print(chi_square_selected_features)

target = 'ACCLASS'
#features = ['INVTYPE', 'INVAGE', 'AUTOMOBILE', 'WEEKDAY', 'MONTH', 'DAY', 'TIMERANGE', 'LIGHT', 
#            'LATITUDE', 'LONGITUDE', 'DISTRICT', 'VISIBILITY', 'RDSFCOND', 'VEHTYPE']

final_df = df_g6[[target] + chi_square_selected_features + ['LATITUDE', 'LONGITUDE', 'INVAGE']].copy()

print("Selected features: ")
print(final_df.info())
#################

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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_features = ['INVAGE']
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())

    ])

#cat_features = ['INVTYPE','WEEKDAY','TIMERANGE','LIGHT','DISTRICT','VISIBILITY','RDSFCOND','VEHTYPE']
cat_features = chi_square_selected_features
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

X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)
#print(X_train_prepared['ACCLASS'].value_counts())

###################### SMOTE
smote_df = pd.Series(y_train)
print("Before SMOTE: ")
print("Fatal: ", smote_df.value_counts()[1])
print("Non-Fatal: ", smote_df.value_counts()[0])
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_SMOTE, y_train_SMOTE = sm.fit_resample(X_train_prepared, y_train.ravel()) 
smote_df = pd.Series(y_train_SMOTE)
print("After SMOTE: ")
print("Fatal: ", smote_df.value_counts()[1])
print("Non-Fatal: ", smote_df.value_counts()[0])
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

