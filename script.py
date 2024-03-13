#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 16:37:43 2024

@author: yichenhsu
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Loading data
df = pd.read_csv('KSI.csv')
#Knowing the data
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

drop_col_df = data[(data.MPercentage > 80)]
drop_row_df = data[(data.MPercentage < 3)]

df.describe().T
df.info()

#Visualization

def split_time(text):
    return text.split(' ')[0]

#Correlation Matrix
#if it is 1, they are duplicate columns
df['TARGET'] = np.where(df['ACCLASS'] != 'Fatal', 0, 1)
corr_matrix = df.corr(numeric_only=True)
corr_matrix['TARGET'].sort_values(ascending=False)
sns.heatmap(corr_matrix)

#Target columns vs INVTYPE
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="ACCLASS", hue="INVTYPE", palette="Set2")
plt.xlabel('Fatal Level')
plt.ylabel('Count')
plt.title('Involvement Type')
plt.show()
#Target columns vs TIME - PENDING
plt.figure(figsize=(100, 6))
sns.countplot(data=df, x="TIME", hue="ACCLASS", palette="Set2")
plt.xlabel('Fatal Level')
plt.ylabel('Count')
plt.title('Involvement Type')
plt.show()

#Preprocessing
drop_cols = ['ObjectId']
dummy_cols = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK',
'TRSN_CITY_VEH', 'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV',
'REDLIGHT', 'ALCOHOL', 'DISABILITY']

def replace_null_dummy(dataframe, cols_name):
    dataframe_copy = dataframe.copy()
    for col in cols_name:
        dataframe_copy[col] = np.where(dataframe_copy[col] == 'Yes', 1, 0)
    return dataframe_copy

df_new = replace_null_dummy(df, dummy_cols)



from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

numeric_features = []
numeric_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())

    ])
cat_features = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
cat_features.extend(['VISIBILITY', 'LIGHT', 'RDSFCOND'])
cat_features.extend(['day_of_week', 'TIMERANGE'])
cat_features.extend(['INVAGE'])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
    
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

