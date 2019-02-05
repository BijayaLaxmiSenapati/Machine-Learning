# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 11:07:38 2019

@author: bsenapat
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 15:05:29 2019

@author: bsenapat
"""
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# Reading the train and the test data.
train = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/train.csv')
test_feature = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/test.csv')
test_label_with_passengerId = pd.read_csv(r'D:/Users/bsenapat/Documents/TitanicTrial1/dataset/gender_submission.csv')

test = pd.merge(test_feature, test_label_with_passengerId, on='PassengerId')
print("\n")
print("test.columns ",test.columns)

full_dataset = pd.concat([train, test], ignore_index=True, sort=True)#(it concatenates well but along with the values of index column)
print("\n")
print("full_dataset.columns ",full_dataset.columns)

# Store our test passenger IDs for easy access
PassengerId = full_dataset['PassengerId']

# Has_Cabin tells whether a passenger had a cabin or not
full_dataset['Has_Cabin'] = full_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
full_dataset['Has_Cabin'] = full_dataset["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
#print(full_dataset.columns)

# Create new feature FamilySize as a combination of SibSp and Parch
full_dataset['FamilySize'] = full_dataset['SibSp'] + full_dataset['Parch'] + 1
#print(full_dataset.columns)

# Create new feature IsAlone from FamilySize
full_dataset['IsAlone'] = 0
full_dataset.loc[full_dataset['FamilySize'] == 1, 'IsAlone'] = 1
#print(full_dataset.columns)

# Remove all NULLS in the Embarked column
full_dataset['Embarked'] = full_dataset['Embarked'].fillna('S')
#print(full_dataset.columns)

# Removes all NULLS in the Fare column
full_dataset['Fare'] = full_dataset['Fare'].fillna(full_dataset['Fare'].median())

###################### Removes all NULLS in the Age column ###############
age_avg = full_dataset['Age'].mean()   
age_std = full_dataset['Age'].std() 
age_null_count = full_dataset['Age'].isnull().sum()  
age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
full_dataset.loc[np.isnan(full_dataset['Age']), 'Age'] = age_null_random_list
full_dataset['Age'] = full_dataset['Age'].astype(int)

# Function to extract titles from passenger names
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

# Creates Title feature from names
full_dataset['Title'] = full_dataset['Name'].apply(get_title)
    
# Group all non-common titles into one single grouping "Rare" and others into different groups
full_dataset['Title'] = full_dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
full_dataset['Title'] = full_dataset['Title'].replace('Mlle', 'Miss')
full_dataset['Title'] = full_dataset['Title'].replace('Ms', 'Miss')
full_dataset['Title'] = full_dataset['Title'].replace('Mme', 'Mrs')

################### Converts all the categorical features into integers #######################
# Mapping Sex
full_dataset['Sex'] = full_dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Mapping titles
title_mapping = {"Mr": 1, "Master": 2, "Mrs": 3, "Miss": 4, "Rare": 5}
full_dataset['Title'] = full_dataset['Title'].map(title_mapping)
full_dataset['Title'] = full_dataset['Title'].fillna(0)

# Mapping Embarked
full_dataset['Embarked'] = full_dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

# Mapping Fare
full_dataset.loc[ full_dataset['Fare'] <= 7.91, 'Fare'] = 0
full_dataset.loc[(full_dataset['Fare'] > 7.91) & (full_dataset['Fare'] <= 14.454), 'Fare'] = 1
full_dataset.loc[(full_dataset['Fare'] > 14.454) & (full_dataset['Fare'] <= 31), 'Fare']   = 2
full_dataset.loc[ full_dataset['Fare'] > 31, 'Fare'] = 3
full_dataset['Fare'] = full_dataset['Fare'].astype(int)
    
# Mapping Age
full_dataset.loc[ full_dataset['Age'] <= 16, 'Age'] = 0
full_dataset.loc[(full_dataset['Age'] > 16) & (full_dataset['Age'] <= 32), 'Age'] = 1
full_dataset.loc[(full_dataset['Age'] > 32) & (full_dataset['Age'] <= 48), 'Age'] = 2
full_dataset.loc[(full_dataset['Age'] > 48) & (full_dataset['Age'] <= 64), 'Age'] = 3
full_dataset.loc[ full_dataset['Age'] > 64, 'Age'] = 4

# Feature selection: remove variables no longer containing relevant information
# reason for removal:
# PassengerId, Name- Both have 100% unique values
# Ticket- Have about 76% unique values which can not be grouped
# SibSp, Parch- From both features information is retrieved and converted to FamilySize
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch']
full_dataset = full_dataset.drop(drop_elements, axis = 1)
#print(full_dataset.columns)

#from the heat map we got to know that column Sex and Title have a big corelation value. One of the two can be removed from the dataset because both of them almost have same information.'''
# corelation of Sex with Survived=0.69
# corelation of Title with Survived=0.62
# As Sex has greater corelation with Survived we are keeping Sex column and removing title column.
full_dataset = full_dataset.drop(["Title"], axis = 1)
#print(full_dataset.columns)

print("Columns after preprocessing: ",full_dataset.columns)

# Applying these two columns to string type so that we can one hot encode it.
full_dataset['Sex'] = full_dataset['Sex'].apply(str)
full_dataset['IsAlone'] = full_dataset['IsAlone'].apply(str)
full_dataset['Has_Cabin'] = full_dataset['Has_Cabin'].apply(str)


########################## One Hot Encoding of Categorical features ################################
full_dataset_dummies=pd.get_dummies(full_dataset)
#print("full_dataset \n",full_dataset)
print("\n")
print("Columns after One Hot Encoding ",full_dataset_dummies.columns)

############################ cross validation using sklearn library(working)S##########################
from sklearn.model_selection import cross_val_score
max_attributes = len(list(full_dataset))
depth_range = range(1, max_attributes + 1)
accuracies = []
for depth in depth_range:
     tree_model = tree.DecisionTreeClassifier(max_depth = depth)
     scores = cross_val_score(tree_model, full_dataset.drop(["Survived"], axis=1).values, full_dataset[["Survived"]].values, cv=10 , scoring='accuracy')
     accuracies.append(scores.mean())
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print("\n")
print(df.to_string(index=False))

########################### GRID SEARCH (working)###############################

X = full_dataset_dummies.drop(['Survived'], axis=1).values 
y = full_dataset_dummies['Survived'].values

############ TRAIN_TEST SPLIT OF DATA ##############
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("\n")
print("type(y_train): \n",type(y_train))
print("y_train.shape: \n",y_train.shape)
print("y_train[:, None].shape: \n",y_train[:, None].shape)

y_train_2D=y_train[:, None]
pdTrain = np.concatenate((X_train,y_train_2D),axis=1)
pdTrain = pd.DataFrame.from_records(pdTrain, columns=['Age', 'Embarked', 'Fare', 'Pclass', 'FamilySize', 'Sex_0',
       'Sex_1', 'Has_Cabin_0', 'Has_Cabin_1', 'IsAlone_0', 'IsAlone_1','Survived'])
print("\n")
print("pdTrain: \n",pdTrain.shape)

a = pdTrain.groupby('Survived').count()
print("\n")
print("Dead and Survived in pdTrain \n",a)

b = pdTrain.groupby(['Sex_0','Survived']).size()
print("\n")
print("Survival rate related to Sex in pdTrain dataset \n",b)

y_test_2D=y_test[:, None]
pdTest = np.concatenate((X_test,y_test_2D),axis=1)
pdTest = pd.DataFrame.from_records(pdTest, columns=['Age', 'Embarked', 'Fare', 'Pclass', 'FamilySize', 'Sex_0',
       'Sex_1', 'Has_Cabin_0', 'Has_Cabin_1', 'IsAlone_0', 'IsAlone_1','Survived'])
print("\n")
print("pdTest: \n",pdTest.shape)

c = pdTest.groupby('Survived').count()
print("\n")
print("Dead and Survived in pdTest \n",c)

d = pdTest.groupby(['Sex_0','Survived']).size()
print("\n")
print("Survival rate related to Sex in pdTest dataset \n",d)


param_grid = {"max_depth": [3,4,5,6,7,8,9],
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10],
              "criterion":["gini", "entropy"]}
decision_tree = tree.DecisionTreeClassifier()

grid = GridSearchCV(estimator = decision_tree,param_grid = param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print("\n")
print("tuned decision tree parameters: ",format(grid.best_params_))
print("\n")
print("Best Score is: ",format(grid.best_score_))

# Predicting results for test dataset
y_train_pred = grid.predict(X_train)

acc_decision_tree_train = round(grid.score(X_train, y_train) * 100, 2)
print("\n")
print("Accuracy on pdTrain dataset(GS)",acc_decision_tree_train)

y_test_pred = grid.predict(X_test)

acc_decision_tree_test = round(grid.score(X_test, y_test) * 100, 2)
print("\n")
print("Accuracy on pdTest dataset(GS)",acc_decision_tree_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print("\n")
print("Confusion matrix of train \n",confusion_matrix(y_train, y_train_pred))
print("Precision score of train: ",precision_score(y_train,y_train_pred))
print("Recall score of train: ", recall_score(y_train,y_train_pred))
print("\n")
print("Confusion matrix of test \n",confusion_matrix(y_test, y_test_pred))
print("Precision score of test: ",precision_score(y_test, y_test_pred))
print("Recall score of test: ", recall_score(y_test, y_test_pred))