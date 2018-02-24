import sys
import string
import math
import itertools
import re
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score
from sklearn.grid_search import GridSearchCV
from scipy.stats import mode



sexClassifier = lambda sex: 0 if sex == "male" else 1
embarkClassifier = lambda emb: 0 if emb == "S" else 1 if emb == "C" else 2
familyDiscretizer = lambda familySize: 0 if familySize == 1 else 1 if familySize < 4 else 2 if familySize == 4 else 3
hasCabin = lambda cabin: 0 if pd.isnull(cabin) else 1
#fareNormalizer = lambda fare: 0 if fare < 1.0 else math.log(fare, 2)
ageDiscretizer = lambda age: 0 if age < 18.0 else 1 if age <= 30 else 2 if age <= 40 else 3
fareDiscretizer = lambda fare: 0 if fare < 8.0 else 1 if fare < 14.5 else 2 if fare <= 31 else 3

def visual_hr(label = '') :
    print('')
    print('-------------------- ', label, ' --------------------')
    print('')

def print_df_info(df, label = '') :
    visual_hr('INFO >> ' + label)
    print(df.info())
    visual_hr('DESCRIBE >> ' + label)
    print(df.describe())

def convert_and_add_new_attributes(df) :
    df['HasCabin'] = df['Cabin'].apply(hasCabin)

    df['Family'] = df['SibSp'] + df['Parch'] + 1
    df['Family'] = df['Family'].apply(familyDiscretizer)

    df['Title'] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
    df['Title'].replace(['Countess', 'Mne', 'Lady', 'Mlle'], 'Ms', inplace=True)
    df['Title'].replace(['Dona'], 'Mrs', inplace=True)
    df['Title'].replace(['Col', 'Major', 'Dr'], 'Col_Major_Dr', inplace=True)
    df['Title'].replace(['Jonkheer', 'Don', 'Rev', 'Capt'], 'Rip', inplace=True)

def granular_series_data(df, series) :
    seriesPerSurvived = df[[series, 'Survived']].groupby([series], as_index=False).mean().sort_values(by='Survived', ascending=False)
    visual_hr("SURVIVAL RATE IN SERIES >> " + series)
    print(seriesPerSurvived)

def fill_missing_values(df, df_combined) :
    # age_mean = (train_df['Age'].median() + test_df['Age'].median())/2
    # df['Age'].fillna(age_mean, inplace=True)
    df_age_pivot = df_combined.pivot_table(values='Age', columns=['Title'], aggfunc='mean')
    df['Age'] = df[['Age', 'Title']].apply(lambda x: df_age_pivot[x['Title']] if pd.isnull(x['Age']) else x['Age'], axis=1).astype(int)
    
    # fare_mean = (train_df['Fare'].median() + test_df['Fare'].median())/2
    # df['Fare'].fillna(fare_mean, inplace=True)
    df_fare_pivot = df_combined.pivot_table(values='Fare', columns=['Pclass'], aggfunc='mean')
    df['Fare'] = df[['Fare', 'Pclass']].apply(lambda x: df_fare_pivot[x['Pclass']] if pd.isnull(x['Fare']) else x['Fare'], axis=1).astype(int)

    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert string types to numeric groups for classifiers
def preparation(df) : 
    df["Sex"] = df["Sex"].apply(sexClassifier) #df['Sex'] = df['Sex'].map({'female': 0, 'male':1}).astype(int)
    df["Embarked"] = df["Embarked"].apply(embarkClassifier)
    df['Family'] = df['Family'].apply(familyDiscretizer)
    df['Fare'] = df['Fare'].apply(fareDiscretizer)
    df['Age'] = df['Age'].apply(ageDiscretizer)

    unique_titles = df['Title'].unique()
    title_map = dict(zip(unique_titles, range(len(unique_titles))))
    df['Title'] = df['Title'].map(title_map).astype(int)



# Read data into pandas dataframe
invaluable_columns = ['Ticket']

df = pd.read_csv("train.csv", sep = ',', header = 0)
df = df.drop(invaluable_columns, axis=1)

test_df = pd.read_csv("test.csv", sep = ',', header = 0)
test_df = test_df.drop(invaluable_columns, axis=1)

# Introduce new attributes
convert_and_add_new_attributes(df)
convert_and_add_new_attributes(test_df)

predictor_columns = ['Sex', 'Pclass', 'Title', 'Fare', 'Age', 'Family', 'Embarked', 'HasCabin']

# General information about the datasets
print_df_info(df, '<ORIGINAL> TRAIN')
print_df_info(test_df, '<ORIGINAL>  TEST')
visual_hr()

# Granular data per series
for series in predictor_columns :
    granular_series_data(df, series)

# Fill missing values using joint train and test datasets
df_combined = pd.concat([df, test_df], ignore_index=True)

fill_missing_values(df, df_combined)
fill_missing_values(test_df, df_combined)

# CLASSIFICATION
preparation(df)
preparation(test_df)

labels = df["Survived"].values
features = df[list(predictor_columns)].values

model_names = ["Decision Tree", "Extra Trees", "Random Forest", 
               "K-Nearest Neighbors", "Linear SVM", "Logistic Regression",
               "AdaBoost", "Gradient Boosting", "Gaussian Naive Bayes",
               "Neural Net - Multi-layer Perceptron"]

models = [
    DecisionTreeClassifier(),
    ExtraTreesClassifier(),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    MLPClassifier()
]

# 10-fold Cross-validation on trained models
# Get feature improtances/coefficients
for i in range(len(models)):
    predictor = models[i]
    predictor.fit(features, labels)
    et_score = cross_val_score(predictor, features, labels, cv=10, n_jobs=-1)
    print('')
    print("{0} -> ET: {1})".format(model_names[i], et_score.mean()))

    if hasattr(predictor, 'feature_importances_'):
        feature_df = pd.DataFrame({'Columns': predictor_columns, 'Importance': predictor.feature_importances_})
        print(feature_df.sort_values(['Importance'], ascending=[0]))
    if hasattr(predictor, 'coef_'):
        coef_df = pd.DataFrame({'Columns': predictor_columns, 'Coefficient': predictor.coef_[0]})
        print(coef_df.sort_values(['Coefficient'], ascending=[0]))



# Hyper-parameter tune promising predictors

# GRID SEARCH -> Random Forest Classifier
# grid_search = GridSearchCV(RandomForestClassifier(), {
#     'n_estimators': [10, 20, 50, 100],
#     'max_features': [0.6, 0.8, 1.0, 'auto'],
#     'max_depth': [5, 7],
#     'min_samples_split': [2, 3, 4],
# 	  'min_samples_leaf': [1, 2, 4],
# 	}, cv=10, verbose=3, n_jobs=-1)

# GRID SEARCH -> Extra Trees Classifier
# grid_search = GridSearchCV(ExtraTreesClassifier(), {
#     'n_estimators': [10, 20, 50, 100],
#     'max_features': [0.6, 0.8, 1.0, 'auto'],
#     'max_depth': [5, 7],
#     'min_samples_split': [2, 3, 4],
# 	  'min_samples_leaf': [1, 2, 4],
#     }, cv=10, verbose=3, n_jobs=-1)

# GRID SEARCH -> Gradient Boosting Classifier
# grid_search = GridSearchCV(GradientBoostingClassifier(), {
#     'n_estimators': [10, 20, 50, 100],
#     'max_features': [0.6, 0.8, 1.0, 'auto'],
#     'max_depth': [5, 7],
#     'min_samples_split': [2, 3, 4],
# 	  'min_samples_leaf': [1, 2, 4],
#     }, cv=10, verbose=3, n_jobs=-1)

# GRID SEARCH -> Support Vector Machine
# grid_search = GridSearchCV(SVC(), {
#     'C': [1, 5, 10, 20, 100],
#     'kernel': ['linear', 'poly', 'rbf'],
#     'degree': [1, 2, 3, 5],
#     'shrinking': [True, False],
#     'gamma': ['auto', 0.001, 0.0001],
#     'class_weight': ['balanced', None]
#     }, cv=10, verbose=3, n_jobs=-1)

# GRID SEARCH -> Multi-layer Perceptron Neural Network
# grid_search = GridSearchCV(MLPClassifier(), {
#     'hidden_layer_sizes': [20, 50, 100],
#     'activation': ['identity', 'logistic', 'tanh', 'relu'],
#     'solver': ['lbfgs', 'sgd', 'adam'],
#     'learning_rate': ['constant', 'invscaling', 'adaptive']
#     }, cv=10, verbose=3, n_jobs=-1)

# GRID SEARCH -> Logistic Regression
# grid_search = GridSearchCV(LogisticRegression(), {
#     'penalty': ['l2'],
#     'dual': [False],
#     'C': [1, 10, 100],
#     'class_weight': ['balanced', None],
#     'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
#     }, cv=10, verbose=3, n_jobs=-1)

# grid_search.fit(features, labels)

# print('Best gridsearch score -> ', grid_search.best_score_)
# print('Best classification parameters -> ', grid_search.best_params_)

# et = GradientBoostingClassifier(n_estimators=100, max_features=0.8, min_samples_leaf=4, min_samples_split=4, max_depth=3)
# et = RandomForestClassifier(n_estimators=100, max_features=0.7, min_samples_leaf=4, min_samples_split=4)
et = ExtraTreesClassifier(n_estimators=100, max_features=0.8, max_depth=7.0, min_samples_split=3, min_samples_leaf=4)
# et = SVC(gamma='auto', kernel='poly', C='100')
# et = MLPClassifier(hidden_layer_sizes=50, activation='relu', solver='lbfgs', learning_rate='constant')
# et = LogisticRegression(penalty='l2', dual=False, C='1', solver='newton-cg')
et.fit(features, labels)


# PREDICT
predictions = et.predict(test_df[predictor_columns].values)
test_df["Survived"] = pd.Series(predictions)

test_df.to_csv("survivors.csv", columns=['PassengerId', 'Survived'], index=False)
predictions = et.predict(test_df[predictor_columns].values)
test_df["Survived"] = pd.Series(predictions)

test_df.to_csv("survivors.csv", columns=['PassengerId', 'Survived'], index=False)