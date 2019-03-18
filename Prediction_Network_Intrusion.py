# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:17:00 2019

@author: g704874
"""

import os
import pandas as pd
import numpy as np
os.getcwd()
os.chdir("C:/IMARTICUS Learning/FromRahul")
os.getcwd()

##IMPORTING Training and test data sets
TrainData = pd.read_csv("R_Module_Day_12.1_Network_Intrusion_Train_data.csv")
TestData = pd.read_csv("R_Module_Day_12.3_Network_Intrusion_Test_data.csv")

# Create a new column. Call it "Source". Assign the value of "Train" in that column
TrainData["Source"] = "Train"
TestData["Source"] = "Test"

####COMBINE BOTH Data and call it as FullData
FullData = pd.concat([TrainData,TestData])
FullData.shape
FullData.head()

####CHECK FOR NA's or Null Values
FullData.isnull().sum()

###CHECK The summary of continous variables
FullData.describe()

####FInd uniques of Categorical variables
FullData.loc[:,FullData.dtypes == object].nunique()

####COMBINING SERVICE variable categories
# Step 1: Check "similarity" in proportions of different categories w.r.t. the dependent variable
TrainData.columns
D1 = pd.crosstab(TrainData['service'], TrainData['class'], margins = True)
D1.head()
D2 = pd.DataFrame(round(D1.normal/D1.All,3))
D2.head()

# Step 2: Make a copy of the original column "service" in the Fulldata and call it Service_New
FullData["Service_New"] = FullData.service

# Step 3: Modify "Service_New" variablex with different ifelse statements combining the identified categories 
# Category: anomaly = 0
Temp_Serv_Cat = D2.index[D2[0] == 0]
Temp_Serv_Cat
FullData.Service_New = np.where(FullData.Service_New.isin(Temp_Serv_Cat), "Serv_Cat_1", FullData.Service_New)

# Category: anomaly = 1
Temp_Serv_Cat = D2.index[D2[0] == 1]
Temp_Serv_Cat
FullData.Service_New = np.where(FullData.Service_New.isin(Temp_Serv_Cat), "Serv_Cat_2", FullData.Service_New)

# Check the number of unique categories
FullData.service.nunique()
FullData.Service_New.nunique() # Should have reduced to a reasonably lower number of categories

# Remove original "service" column
FullData2 = FullData.drop(['service'], axis = 1).copy()

#####DUMMY VARIABLE CREATION ###########
# Create dummy variables for all indep categorical variables

# Step 1: Identify categorical vars
Categ_Vars = FullData2.loc[:,FullData2.dtypes == "object"].columns
Categ_Vars

# Step 2: Create dummy vars
Dummy = pd.get_dummies(FullData2[Categ_Vars].drop(['Source'],axis=1),drop_first=True)
Dummy.shape
Dummy.columns
Dummy.dtypes

## Merge the original data set with dummy columns
FullData3 = pd.concat([FullData2,Dummy],axis = 1)
FullData3.shape
FullData3.columns

# Step 4.1: Drop all the irrelavant and categorical columns (Do NOT drop Source column - We need it for sample splitting)

Cols_To_Drop = Categ_Vars.drop('Source') # Ensure you do not consider 'Source' column in "columns to drop"

#Cols_To_Drop = Categ_Vars_list.remove('Source')

# Step 4.2
FullData4 = FullData3.drop(Cols_To_Drop, axis = 1).copy()
FullData4.shape
FullData4.columns

#####SAMPLING ########################################
# make sure you drop the source column
Train= FullData4.loc[FullData4.Source == "Train",].drop('Source',axis=1).copy()
Train.shape
Test = FullData4.loc[FullData4.Source == "Test",].drop('Source',axis =1).copy()
Test.shape

##Dividing dataset into Independent and Dependent variables
train_X = Train.drop('class_normal',axis =1)
train_y = Train['class_normal'].copy()
test_X = Test.drop('class_normal',axis =1)
test_y = Test['class_normal'].copy()

######## MODELING - DECISION TREE ####################
from sklearn.tree import DecisionTreeClassifier

tree_M1 = DecisionTreeClassifier(random_state = 100)
Model1 = tree_M1.fit(train_X, train_y)


######MODEL VISUALIZATION #############################
!pip install pydotplus
import pydotplus

# Step 2: install
# install package directly from within spyder: !pip install pydotplus
# pip is a package management system used to install and 
# manage packages written in Python
#!pip install pydotplus

# Step 3: validate if the installation has come through
#import pydotplus 

from sklearn.tree import export_graphviz

# Step 1: Download the following software (Graphviz):
# https://graphviz.gitlab.io/_pages/Download/Download_windows.html

# Step 2.1: Restart spyder
# Step 3: Python implementation
# Create DOT data (DOT format is a format to store grpahical data in text format)
dot_data = export_graphviz(Model1, out_file=None, feature_names = train_X.columns)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data)  

# Show graph (on console) # Takes a bit of time
#Image(graph.create_png())

# Write to file
graph.write_pdf("DT_Plot.pdf")


################# MODEL BUILDING - RANDOM FOREST ##########
from sklearn.ensemble import RandomForestClassifier
M2 = RandomForestClassifier(random_state=100)
M2_Model = M2.fit(train_X,train_y)
Pred1 = M2_Model.predict(test_X)

## CONFUSION MATRIX ######
from sklearn.metrics import confusion_matrix
conf1 = confusion_matrix(test_y,Pred1)
conf1

###Accuracy ###########
Accuracy = ((conf1[0][0] + conf[1][1])/test_y.shape[0]) * 100

## CLASSIFICATION Report ######
from sklearn.metrics import classification_report
report1 = classification_report(test_y,Pred1)
print(report1)

####RANDOM FOREST using RANDOM SEARCH
from sklearn.model_selection import RandomizedSearchCV

param_grid = {'n_estimators': [25, 50, 75], 'max_features': [5, 7, 9, 11], 'min_samples_split' : [1000, 2000]} # param_grid is a dictionary
RF_RS = RandomizedSearchCV(RandomForestClassifier(random_state=100), param_distributions=param_grid,  scoring='accuracy', cv=5)
# Other scoring parameters are available here: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

RF_RS_Model = RF_RS.fit(train_X, train_y)

RF_Random_Search_Df = pd.DataFrame.from_dict(RF_RS_Model.cv_results_)

# Best tuning parameters
RF_RS_Model.best_params_










