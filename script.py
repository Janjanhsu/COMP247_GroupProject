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
pip install geopandas shapely
pip install seaborn
pip install tabulate
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
print(df_g6.iloc[:, -3:])

'''
Data Visualization
'''
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
df['FATAL'] = np.where(df['ACCLASS'] != 'Fatal', 0, 1)
df['TIMEHOUR'] = df['TIME'] // 100

# Grouping by TIMEHOUR and calculating number of accidents and number of persons died in accidents
n_accident_per_hour = df.groupby('TIMEHOUR')['ACCNUM'].nunique()
n_fatal_case_per_hour = df.groupby('TIMEHOUR')['FATAL'].sum()

result = pd.DataFrame({
        'No. of accidents occurred': n_accident_per_hour, 
        'No. of persons died in accidents': n_fatal_case_per_hour
    }) 
result.plot.bar(xlabel='Time (hour)')
plt.show()

#VEHTYPE vs occurence of accidents
plt.figure(figsize=(10, 5))
sns.countplot(data = df, x="VEHTYPE", hue="ACCLASS", palette="viridis")
plt.title('Accidents by Vehicle Type')
plt.xlabel('Vehicle Type')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=90)
plt.show()

#WEEKDAY vs occurence of accidents
weekday_ordered = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
]

plt.figure(figsize=(10, 5))
sns.countplot(data = df_g6, x='WEEKDAY', hue="ACCLASS", palette="Blues", order= weekday_ordered)
plt.title('Accidents by Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=40)
plt.show()

#LOCCOORD vs occurence of accidents
plt.figure(figsize=(9, 5))
sns.countplot(data = df, x="LOCCOORD", hue="ACCLASS", palette="Set1")
plt.title('Accidents by Location Coordinates')
plt.xlabel('Location')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=90)
plt.show()

# Visualize the relationship between "RDSFCOND" (road surface condition) and the target variable "ACCLASS"
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="RDSFCOND", hue="ACCLASS")
plt.xlabel('Road Surface Condition')
plt.ylabel('Count')
plt.title('Road Surface Condition vs. Fatal Level')
plt.xticks(rotation=55)
plt.show()


# Visualize the relationship between "LIGHT" (light condition) and the target variable "ACCLASS"
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x="LIGHT", hue="ACCLASS")

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

myadd = df_g6[['LATITUDE', 'LONGITUDE', 'ACCLASS']]
myadd2 = myadd[myadd.ACCLASS == 1]
#fig = px.scatter_geo(df,lat='LATITUDE',lon='LONGITUDE')
#fig.update_layout(title = 'World map', title_x=0.5)
#fig.show()
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame

geometry = [Point(xy) for xy in zip(myadd['LONGITUDE'], myadd['LATITUDE'])]
gdf = GeoDataFrame(myadd, geometry=geometry)
geometry2 = [Point(xy) for xy in zip(myadd2['LONGITUDE'], myadd2['LATITUDE'])]
gdf2 = GeoDataFrame(myadd2, geometry=geometry2)

# Load a simple map that goes with geopandas
map1 = gpd.read_file("NEIGHBORHOODS_WGS84_2.shp")

# Plot the points on the map
#gdf.plot(ax=map1.plot(figsize=(10, 6)), marker='o', color='orange', markersize=2)
gdf2.plot(ax=map1.plot(figsize=(10, 6)), marker='o', color='red', markersize=2)
plt.title('Fatal accidents')
plt.show()

#Year - Accidents in different years
# Visualize the relationship between YEAR and the target variable "ACCLASS"
plt.figure(figsize=(10, 6))
plt.title('Accidents that occurred by year')
sns.countplot(data=df, x="YEAR", hue="ACCLASS")

#Months - Accidents in different months
the_months = df_g6['MONTH'].value_counts()
the_months = the_months.sort_index(axis=0)
the_months.index = [
    'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'
]
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

numeric_features = ['INVAGE','LATITUDE', 'LONGITUDE']
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

#############################
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000)

# Fit the model on the training data
log_reg.fit(X_train_SMOTE, y_train_SMOTE)

# Make predictions on the testing set
y_pred_log_reg = log_reg.predict(X_test_prepared)

# Evaluate the model
print("Model: Logistic Regression ")
print("Accuracy on test data: ", accuracy_score(y_test, y_pred_log_reg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Classification Report:\n", classification_report(y_test, y_pred_log_reg))


###############################
#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_Clf = DecisionTreeClassifier(criterion="gini", random_state=53, max_depth=3, min_samples_leaf=5)


# Performing training
dt_Clf.fit(X_train_SMOTE, y_train_SMOTE)

y_pred_dt = dt_Clf.predict(X_test_prepared)

print("Model: Decision Tree ")
print("Accuracy on test data: ", accuracy_score(y_test, y_pred_dt))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Classification Report:\n", classification_report(y_test, y_pred_dt))


###############################
from sklearn.ensemble import RandomForestClassifier

# Train model
forestClf = RandomForestClassifier(random_state=53)
forestClf.fit(X_train_SMOTE, y_train_SMOTE)

y_forest_pred_test = forestClf.predict(X_test_prepared)
 
print("Model: Random Forest ")
print("Accuracy for test data: ", accuracy_score(y_test, y_forest_pred_test))

cm = confusion_matrix(y_test, y_forest_pred_test)
print(f"Confusion Matrix: \n{cm}")

########################
#Neural network
from sklearn.neural_network import MLPClassifier
nnClf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=(5, 2), random_state=53)
nnClf.fit(X_train_SMOTE, y_train_SMOTE)
y_nn_pred_test = nnClf.predict(X_test_prepared)
 
print("Model: Neural Network ")
print("Accuracy for test data: ", accuracy_score(y_test, y_nn_pred_test))

cm = confusion_matrix(y_test, y_nn_pred_test)
print(f"Confusion Matrix: \n{cm}")

###Randomized Search
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve

parameters=[
    {
        'clf': SVC(),
        'name':'SVM',
        'C': [0.001, 0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma':[1,0.1,0.001]
    },
    {
        'clf': LogisticRegression(),
        'name':'Logistic Regression',
        'max_iter' : range(100, 500, 1000),
        'warm_start' : [True, False],
        'solver' : ['lbfgs', 'newton-cg', 'liblinear'],
        'C' : np.arange(0, 1, 0.01)
    },
    {
        'clf': DecisionTreeClassifier(),
        'name':'Decision Tree',
        'min_samples_split' : range(10,300,20),
        'max_depth': range(1,30,2),
        'min_samples_leaf':range(1,15,3)
    },
    {
        'clf': RandomForestClassifier(),
        'name':'Random Forest',
        'max_depth': [None, 5, 10, 20],
        'max_features': [0.5, 0.7, 0.9],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10],
        'n_estimators': [100, 500, 1000],
        'bootstrap': [True,False],
        'criterion': ['gini','entropy']
    },
    {
        'clf': MLPClassifier(max_iter=100),
        'name':'Neural Network',
        'hidden_layer_sizes': [(10,30,10),(20,)],
        'activation': ['tanh', 'relu'],
        'solver': ['sgd', 'adam'],
        'alpha': [0.0001, 0.05],
        'learning_rate': ['constant','adaptive']
     }
]

from sklearn.model_selection import RandomizedSearchCV
def plot_roc_curve(true_y, y_prob, name):
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr, label=name)
    plt.title("ROC Curves")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')

def runRandomizedSearch(clf, parameters):
    randSearch = RandomizedSearchCV(estimator=clf,
                        scoring='accuracy', param_distributions=parameters, cv=4,
                        n_iter = 7, refit = True, verbose = 3)
    
    randSearch.fit(X_train_SMOTE, y_train_SMOTE)
    #Print out the best parameters
    print(randSearch.best_params_)
    
    #Print out the score of the model 
    rand_score = randSearch.best_score_
    print('The score of the model: ', rand_score)
    
    best_model = randSearch.best_estimator_
    #Printout the best estimator 
    print(best_model)
    
    y_pred = best_model.predict(X_test_prepared)
    
    m_accuracy = accuracy_score(y_test, y_pred)
    m_precision = precision_score(y_test, y_pred)
    m_recall = recall_score(y_test, y_pred)
    m_f1 = f1_score(y_test, y_pred)
    m_cm = confusion_matrix(y_test, y_pred)
    y_proba = best_model.predict_proba(X_test_prepared)
    
    return [m_accuracy, m_precision, m_recall, m_f1], m_cm, y_proba

from sklearn.model_selection import GridSearchCV
def runGridSearchCV(clf, parameters):
    gridSearch = GridSearchCV(estimator=clf,
                        scoring='accuracy', param_grid=parameters, cv=4,
                        refit = True, verbose = 3)
    
    gridSearch.fit(X_train_SMOTE, y_train_SMOTE)
    #Print out the best parameters
    print(gridSearch.best_params_)
    
    #Print out the score of the model 
    grid_score = gridSearch.best_score_
    print('The score of the model: ', grid_score)
    
    best_model = gridSearch.best_estimator_
    #Printout the best estimator 
    print(best_model)
    
    y_pred = best_model.predict(X_test_prepared)
    
    m_accuracy = accuracy_score(y_test, y_pred)
    m_precision = precision_score(y_test, y_pred)
    m_recall = recall_score(y_test, y_pred)
    m_f1 = f1_score(y_test, y_pred)
    m_cm = confusion_matrix(y_test, y_pred)
    y_proba = best_model.predict_proba(X_test_prepared)
    
    return [m_accuracy, m_precision, m_recall, m_f1], m_cm, y_proba

model_results = []
cm_results={}
for params in parameters:
    classifier = params.pop('clf')
    name = params.pop('name')
    #result, cm, y_proba_bm = runRandomizedSearch(classifier, params)
    result, cm, y_proba_bm = runGridSearchCV(classifier, params)
    result.insert(0, name)
    model_results.append(result)
    plot_roc_curve(y_test, y_proba_bm[:, 1], name)
    cm_results[name] = cm

from tabulate import tabulate
headers = ['Model','Accuracy', 'Precision', 'Recall', 'F1']
print(tabulate(model_results, headers=headers, tablefmt="grid", numalign="center"))

for e in cm_results.keys():
    print(e)