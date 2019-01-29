# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 14:59:46 2019

@author: bsenapat
"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont
from IPython.display import Image as PImage
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Loading the data
train = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/train.csv')
test = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/test.csv')
test_label = pd.read_csv(r'D:/Users/bsenapat/Documents/TitanicTrial1/dataset/gender_submission.csv')

# Store our test passenger IDs for easy access
PassengerId = test['PassengerId']

full_data = [train, test]

# Has_Cabin tells whether a passenger had a cabin or not
train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    
# Create new feature IsAlone from FamilySize
for dataset in full_data:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column
for dataset in full_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')


# Removes all NULLS in the Fare column
for dataset in full_data:
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

# Removes all NULLS in the Age column
for dataset in full_data:
    age_avg = dataset['Age'].mean()
    #print("age_avg",age_avg)
    
    age_std = dataset['Age'].std()
    #print("age_std",age_std)
    
    age_null_count = dataset['Age'].isnull().sum()
    #print("age_null_count",age_null_count)
    
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    # Next line has been improved to avoid warning
    dataset.loc[np.isnan(dataset['Age']), 'Age'] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# Function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Creates Title feature from names
for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
    
# Group all non-common titles into one single grouping "Rare" and others into different groups
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Converts all the categorical features into integers
for dataset in full_data:
    # Mapping Sex
    #dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
    # Mapping titles
    '''title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)'''
    
    # Mapping Embarked
    #dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    
    # Mapping Fare
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
    
    # Mapping Age
    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4

# Feature selection: remove variables no longer containing relevant information
# reason for removal:
# PassengerId, Name- Both have 100% unique values
# Ticket- Have about 76% unique values which can not be grouped
# SibSp, Parch- From both features information is retrieved and converted to FamilySize
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
train = train.drop(drop_elements, axis = 1)
test  = test.drop(drop_elements, axis = 1)

# Applying these two columns to string type so that we can one hot encode it.
train['IsAlone'] = train['IsAlone'].apply(str)
test['IsAlone'] = test['IsAlone'].apply(str)
train['Has_Cabin'] = train['Has_Cabin'].apply(str)
test['Has_Cabin'] = test['Has_Cabin'].apply(str)

########################## One Hot Encoding of Categorical features #################
train_dummies=pd.get_dummies(train)
print("Final columns in train dataset \n", train_dummies.columns)
test_dummies=pd.get_dummies(test)
print("Final columns in test dataset \n", test_dummies.columns)

# Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models'''
x_train = train_dummies.drop(['Survived'], axis=1).values 
y_train = train_dummies['Survived']
x_test = test_dummies.values

# Create Decision Tree with max_depth = 3
#decision_tree = tree.DecisionTreeClassifier(max_depth = 3)

############################### CROSS VALIDATION #################################
'''cv = KFold(n_splits=10) # Desired number of Cross Validation folds
accuracies = list()
max_attributes = len(list(test))
depth_range = range(1, max_attributes + 1)

print("cv -",cv)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    for train_fold, valid_fold in cv.split(train):
        f_train = train.loc[train_fold] # Extract train data with cv indices
        f_valid = train.loc[valid_fold] # Extract valid data with cv indices

        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), 
                               y = f_train["Survived"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), 
                                y = f_valid["Survived"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)

# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))'''

########################### RANDOMIZED SEARCH ###############################
'''param_grid = {"max_depth": [1,2,3,4,5,6,7,8,9,10],
              "min_samples_leaf": [6,7,8,9,10,11,12,13,14],
              "criterion":["gini", "entropy"]}
decision_tree = tree.DecisionTreeClassifier()

tree_cv = GridSearchCV(estimator = decision_tree,param_grid = param_grid, cv=3, n_jobs=-1)
tree_cv.fit(x_train, y_train)

print("tuned decision tree parameters: ",format(tree_cv.best_params_))
print("Best Score is: ",format(tree_cv.best_score_))


# Predicting results for test dataset
y_pred = tree_cv.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)

acc_decision_tree_train = round(tree_cv.score(x_train, y_train) * 100, 2)
print("Accuracy on train dataset",acc_decision_tree_train)

acc_decision_tree_test = round(accuracy_score(test_label['Survived'].values , submission['Survived'].values)*100, 2)
print("Accuracy on test dataset",acc_decision_tree_test)'''

decision_tree1 = tree.DecisionTreeClassifier(max_depth=3, criterion='gini')
decision_tree1.fit(x_train, y_train)
# Predicting results for test dataset
y_pred = decision_tree1.predict(x_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)

acc_decision_tree_train = round(decision_tree1.score(x_train, y_train) * 100, 2)
print("Accuracy on train dataset",acc_decision_tree_train)

acc_decision_tree_test = round(accuracy_score(test_label['Survived'].values , submission['Survived'].values)*100, 2)
print("Accuracy on test dataset",acc_decision_tree_test)

# Export our trained model as a .dot file
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(decision_tree1,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(train_dummies.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )
                                      



     

