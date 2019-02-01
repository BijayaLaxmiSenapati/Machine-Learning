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


# Reading the train and the test data.
train = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/train.csv')
test_feature = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/test.csv')
test_label_with_passengerId = pd.read_csv(r'D:/Users/bsenapat/Documents/TitanicTrial1/dataset/gender_submission.csv')


test = pd.merge(test_feature, test_label_with_passengerId, on='PassengerId')
print("test.columns ",test.columns)

full_dataset = pd.concat([train, test], ignore_index=True, sort=True)#(it concatenates well but along with the values of index column)
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
full_dataset.loc[ full_dataset['Age'] <= 16, 'Age'] 					       = 0
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

######################HEAT MAP##########################################################################
colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(full_dataset.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

#from the heat map we got to know that column Sex and Title have a big corelation value. One of the two can be removed from the dataset because both of them almost have same information.
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
print("Columns after One Hot Encoding ",full_dataset_dummies.columns)

############################### CROSS VALIDATION #######################################################
cv = KFold(n_splits=3) # Desired number of Cross Validation folds
accuracies = list()
max_attributes = len(list(full_dataset))
print("max_attributes",max_attributes)
depth_range = range(1, max_attributes + 1)

print("cv -",cv)

# Testing max_depths from 1 to max attributes
# Uncomment prints for details about each Cross Validation pass
for depth in depth_range:
    fold_accuracy = []
    tree_model = tree.DecisionTreeClassifier(max_depth = depth)
    for train_fold, valid_fold in cv.split(full_dataset_dummies):
        f_train = full_dataset_dummies.loc[train_fold] # Extract train data with cv indices
        f_valid = full_dataset_dummies.loc[valid_fold] # Extract valid data with cv indices

        model = tree_model.fit(X = f_train.drop(['Survived'], axis=1), y = f_train["Survived"]) # We fit the model with the fold train data
        valid_acc = model.score(X = f_valid.drop(['Survived'], axis=1), y = f_valid["Survived"])# We calculate accuracy with the fold validation data
        fold_accuracy.append(valid_acc)

    avg = sum(fold_accuracy)/len(fold_accuracy)
    accuracies.append(avg)

# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False),"\n")

print("type(full_dataset) ",type(full_dataset))


############################ cross validation using sklearn library(working)S##########################
from sklearn.model_selection import cross_val_score
max_attributes = len(list(full_dataset))
print("max_attributes",max_attributes)
depth_range = range(1, max_attributes + 1)
accuracies = []
for depth in depth_range:
     tree_model = tree.DecisionTreeClassifier(max_depth = depth)
     scores = cross_val_score(tree_model, full_dataset.drop(["Survived"], axis=1).values, full_dataset[["Survived"]].values, cv=10 , scoring='accuracy')
     accuracies.append(scores.mean())
# Just to show results conveniently
df = pd.DataFrame({"Max Depth": depth_range, "Average Accuracy": accuracies})
df = df[["Max Depth", "Average Accuracy"]]
print(df.to_string(index=False))

########################### GRID SEARCH (working)###############################
X = full_dataset_dummies.drop(['Survived'], axis=1).values 
y = full_dataset_dummies['Survived'].values
param_grid = {"max_depth": [3,4,5,6,7,8,9],
              "min_samples_leaf": [1,2,3,4,5,6,7,8,9,10],
              "criterion":["gini", "entropy"]}
decision_tree = tree.DecisionTreeClassifier()

grid = GridSearchCV(estimator = decision_tree,param_grid = param_grid, cv=3, n_jobs=-1)
grid.fit(X, y)

print("tuned decision tree parameters: ",format(grid.best_params_))
print("Best Score is: ",format(grid.best_score_))
print("Best Model got from GridSearchCV: ",grid.best_estimator_)

# Predicting results for test dataset
'''y_pred = grid.predict(X)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)
print("y_pred",y_pred)

acc_decision_tree_train = round(grid.score(X, y) * 100, 2)
print("Accuracy on train dataset(GS)",acc_decision_tree_train)'''

########################################### Model #####################################
final_selected_model = grid.best_estimator_
final_selected_model.fit(X, y)
# Predicting results for test dataset
y_pred = grid.predict(X)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)
print("y_pred",y_pred,"\n")
print("Columns after Pre-Processing :\n",full_dataset_dummies.drop(['Survived'], axis=1).columns)
# Export our trained model as a .dot file
# dot tree1.dot -Tpng -o tree1.png command to convert to png
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(final_selected_model,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(full_dataset_dummies.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )