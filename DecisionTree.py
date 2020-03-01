"""
Create a Decision Stump
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import style

style.use('fivethirtyeight')
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
import scipy.stats as sps




"""
Import the DecisionTreeClassifier model.
"""

# Import the DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

###########################################################################################################

##########################################################################################################

"""
Import the Zoo Dataset
"""
from ToDataframe import *



data_path = r"C://Users//admin//Desktop//Output2.json"
TDF = ConvertJsonToDataframe()
data_df = TDF.main(data_path)
dataset = data_df.sample(frac=1)
dataset.columns = ['winnerSide', 'secondAfterBomb', 'aliveTNum', 'aliveCTNum', 'TP1dist', 'TP2dist', 'TP3dist',
                   'TP4dist', 'TP5dist', 'CTP1dist', 'CTP2dist', 'CTP3dist', 'CTP4dist', 'CTP5dist', 'TP1weapon',
                   'TP2weapon', 'TP3weapon', 'TP4weapon', 'TP5weapon', 'CTP1weapon', 'CTP2weapon',
                   'CTP3weapon', 'CTP4weapon', 'CTP5weapon']



###########################################################################################################

##########################################################################################################

"""
Split the data into a training and a testing set
"""

train_features = dataset.iloc[:80, :-1]
test_features = dataset.iloc[80:, :-1]
train_targets = dataset.iloc[:80, -1]
test_targets = dataset.iloc[80:, -1]

###########################################################################################################

##########################################################################################################

"""
Train the model
"""

tree = DecisionTreeClassifier(criterion='entropy').fit(train_features, train_targets)

###########################################################################################################

##########################################################################################################

"""
Predict the classes of new, unseen data
"""
prediction = tree.predict(test_features)

###########################################################################################################

##########################################################################################################

"""
Check the accuracy
"""

print("The prediction accuracy is: ", tree.score(test_features, test_targets) * 100, "%")

