# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:45:36 2019

@author: bsenapat
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the train and the test data.
train = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/train.csv')
test = pd.read_csv('D:/Users/bsenapat/Documents/TitanicTrial1/dataset/test.csv')
test_label = pd.read_csv(r'D:/Users/bsenapat/Documents/TitanicTrial1/dataset/gender_submission.csv')

# Displaying a sample of the train data to get more detailed info
print(train.head())
#Summary of the data:
print(train.describe())
#Data Types of features:
print(train.dtypes)
#checks missing data is present in the features or not
print(train.apply(lambda x: x.isnull().any()))
#percentage of data missing in the features
print(pd.DataFrame({'percent_missing': train.isnull().sum() * 100 / len(train)}))
#Checking for the amount of unique values in each feature.
print(pd.DataFrame({'percent_unique': train.apply(lambda x: x.unique().size/x.size*100)}))

# Names of the features extarcted from the data
selFeatures = list(train.columns.values)
# Removing the target variable from the column values
targetCol = 'Survived'
selFeatures.remove(targetCol)
# Removing features with unique values
for i in selFeatures:
    if train.shape[0]==len(pd.Series(train[i]).unique()):
        selFeatures.remove(i)
       
# Removing features with high percentage of missing values
selFeatures.remove('Cabin')

#Visualizations:
'''import seaborn as sns
sns.set(style="ticks")
plotFeatures = [x for x in selFeatures]
plotFeatures.append("Survived")
sns.pairplot(train[plotFeatures], hue="Survived")'''

#graph of Survived based on categorical values(bar graph)
import plotly.graph_objs as go
import plotly.plotly as py
def plotGraph(plotData,msg):
    trace1 = go.Bar(
    x=plotData.columns.values,
    y=plotData.values[0],
    name='No'
    )
    trace2 = go.Bar(
        x=plotData.columns.values,
        y=plotData.values[1],
        name='Yes'
    )
    data = [trace1, trace2]
    layout = dict(
        title = msg,
        xaxis= dict(title = plotData.columns.name),
        yaxis= dict(title= 'Number of people'),
        barmode='group',
        autosize=False,
        width=800,
        height=500
    )
    fig = dict(data=data, layout=layout)
    py.iplot(fig)
    
#graph of Survived based on Pclass
'''pclass = pd.crosstab([train.Survived], train.Pclass)
plotGraph(pclass,'Survived based on Pclass')'''

#graph of Survived based on Sex
'''sex = pd.crosstab([train.Survived], train.Sex)
plotGraph(sex, 'Survived based on sex')'''

#graph of Survived based on Embarked
'''embarked = pd.crosstab([train.Survived], train.Embarked)
plotGraph(embarked, 'Survived based on embarked')'''

#graph of Survived based on SibSp
'''SibSp = pd.crosstab([train.Survived], train.SibSp)
plotGraph(SibSp, 'Survived based on SibSp')'''

#graph of Survived based on Parch
'''Parch = pd.crosstab([train.Survived], train.Parch)
plotGraph(Parch, 'Survived based on Parch')'''

#graph of Survived based on descrete values(line graph)
'''def plotLine(plotData,msg):
    trace1 = go.Scatter(
    x=plotData.columns.values,
    y=plotData.values[0],
    mode='lines',
    name='No'
    )
    trace2 = go.Scatter(
        x=plotData.columns.values,
        y=plotData.values[1],
        mode='lines',
        name='Yes'
    )
    data = [trace1, trace2]
    layout = dict(
        title = msg,
        xaxis= dict(title = plotData.columns.name),
        yaxis= dict(title= 'Number of people'),
        autosize=False,
        width=800,
        height=500
    )
    fig = dict(data=data, layout=layout)
    iplot(fig)'''

#graph of Survived based on Age    
'''Age = pd.crosstab([train.Survived],train.Age)
plotLine(Age,'Survival based on Age')'''

#graph of Survived based on Fare    
'''Fare = pd.crosstab([train.Survived],train.Fare)
plotLine(Fare,'Survival based on Fare')'''

# Also removing cabin and ticket features for the initial run.
selFeatures.remove('Ticket')
import numpy as np
from sklearn.model_selection import train_test_split

'''def handle_categorical_na(df):
    ## Imputing the null/na/nan values in 'Age' attribute with its mean value 
    df.Age.fillna(value=df.Age.mea,inplace=True)
    ## replacing the null/na/nan values in 'Embarked' attribute with 'X'
    df.Embarked.fillna(value='X',inplace=True)
    return df
seed = 7
np.random.seed(seed)
X_train, X_test, Y_train, Y_test = train_test_split(train[selFeatures], train.Survived, test_size=0.2)
print(type(X_train))'''

# Feature that tells whether a passenger had a cabin on the Titanic
'''train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

# Create new feature FamilySize as a combination of SibSp and Parch
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# Create new feature IsAlone from FamilySize
train['IsAlone'], test['IsAlone'] = 0, 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column
train['Embarked'] = train['Embarked'].fillna('S')
test['Embarked'] = test['Embarked'].fillna('S')

# Remove all NULLS in the Fare column
train['Fare'] = train['Fare'].fillna(train['Fare'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Remove all NULLS in the Age column
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(test['Age'].mean())

# Define function to extract titles from passenger names
import re
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

train['Title'] = train['Name'].apply(get_title)

print(train.drop(columns=["Survived","PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"],axis = 1))
print(test.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "SibSp", "Parch"],axis = 1))'''

######################HEAT MAP##########################
'''colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(train.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)'''

#BAR PLOT OF PCLASS AND SURVIVED FEATURE
sns.barplot(x='Pclass', y='Survived', data=train)

#GRAPH USING PCLASS, SURVIVED, SEX, EMBARKED
FacetGrid = sns.FacetGrid(train, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()

'''prediction=decision_tree.predict([[3, 1, 1, 2, 3, 1, 1, 4, 0, 1]])
#Pclass(1,2,3) Sex(1,0) Age(0,1,2,3) Parch(1,2,3,4,5,6) Fare(0,1,2,3) Embarked(0,1,2) Has_Cabin(0,1) FamilySize(1,2,3,4,5,6,7,8) IsAlone(0,1) Title(1,2,3,4,5)
print(prediction,"prediction")
prediction1=decision_tree.predict([[3, 1, 2, 0, 0, 2, 0, 1, 1, 1]])#(should give zero)
print(prediction1,"prediction1")'''

#CALCULATING ACCURACY USING SERIES TYPE ARGUMENTS
'''print("type(test_label.loc[Survived])",type(test_label.loc[:, "Survived"]))
acc_decision_tree_test2 = round(accuracy_score(test_label.loc[:, 'Survived'] , submission.loc[:, 'Survived'])*100, 2)
print("Accuracy on test dataset",acc_decision_tree_test2)'''

########################################### Model #####################################
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

final_selected_model = DecisionTreeClassifier(max_depth=3, min_sample_leaf=1,random_state=0)
final_selected_model.fit(X_train, y_train)
# Predicting results for test dataset
y_pred = final_selected_model.predict(X_test)
submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_pred
    })
submission.to_csv('submission.csv', index=False)
print("y_pred",y_pred)
acc_decision_tree_train = round(final_selected_model.score(X_test, y_test) * 100, 2)
print("Accuracy on train dataset(GS)",acc_decision_tree_train)
# Export our trained model as a .dot file
#"dot tree1.dot -Tpng -o tree1.png" command to convert to png
with open("tree1.dot", 'w') as f:
     f = tree.export_graphviz(final_selected_model,
                              out_file=f,
                              max_depth = 3,
                              impurity = True,
                              feature_names = list(full_dataset_dummies.drop(['Survived'], axis=1)),
                              class_names = ['Died', 'Survived'],
                              rounded = True,
                              filled= True )'''
