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

#Checking
columns = list(df_g6.columns)
data_g6 = count_value(df_g6, columns)
print(data_g6)

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
#Target columns vs TIME - PENDING
plt.figure(figsize=(100, 6))
sns.countplot(data=df, x="TIME", hue="ACCLASS", palette="Set2")
plt.xlabel('Fatal Level')
plt.ylabel('Count')
plt.title('Involvement Type')
plt.show()


'''
Data modelling
'''
final_df = df_g6.drop('ACCLASS')
target = 'ACCLASS'

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
##################### Balancing

from sklearn.utils import resample

###################


###################### SMOTE
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2) 
X_train_SMOTE, y_train_SMOTE = sm.fit_resample(X_train_prepared, y_train.ravel()) 

#tmp = pd.DataFrame ({'values': y_train_res})
#print(tmp['values'].value_counts())
###################




#9.	
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
#9. set the kernel to linear and set the regularization parameter to C= 0.1
clf_linear_KwokWing = SVC(kernel='poly', C= 0.1)
#9. Train an SVM classifier using the training data
clf_linear_KwokWing.fit(X_train_SMOTE, y_train_SMOTE)

#10. Print accuracy score for the model on the training set
score = clf_linear_KwokWing.score(X_train_SMOTE, y_train_SMOTE)
print("SVC poly:")
print("Accuracy Score(Training data): ", score)

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

