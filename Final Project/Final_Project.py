# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 18:48:28 2021

@author: stk.akkaya
"""


#Import numerical libraries
import pandas as pd
import numpy as np

#Import graphical plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt


#Import Linear Regression Machine Learning Libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score

from sklearn.metrics import confusion_matrix



# Read csv

df = pd.read_csv ('diamonds.csv')





    
    
    
# Describe our data for each feature and use .info() for get information about our dataset

df.info()

df.describe()

# Analyse missing values

# Check duplicate values and missing data.

df_ranmat_dup=df.duplicated()    #Duplication control

df_ranmat_na=df.isna().sum()      #NaN control

df_ranmat_dup_1=df.dropna()    #Duplication control

df_ranmat_na_2=df.isnull().sum()      #NaN control

data=df

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Encoding the label
label_encoder = LabelEncoder()
data["Label_Cut"] = label_encoder.fit_transform(data["cut"]) 
data["Label_Cut"].value_counts()
categories1 = list(label_encoder.inverse_transform([0, 1, 2, 3, 4]))


data["Label_Color"] = label_encoder.fit_transform(data["color"]) 
data["Label_Color"].value_counts()
categories1 = list(label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6]))


data["Label_Clarity"] = label_encoder.fit_transform(data["clarity"]) 
data["Label_Clarity"].value_counts()
categories1 = list(label_encoder.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7]))


data["Label_Price"] = label_encoder.fit_transform(data["price"]) 
data["Label_Price"].value_counts()
categories1 = list(label_encoder.inverse_transform([0, 1, 2, 3, 4]))
data.head()


# Checking encoded labels


# Dropping unnecessary columns
clases = list(set(data.price))
data.drop(["index","cut","color","clarity","price"], axis=1, inplace=True)
data.head()


# Our label Distribution (countplot)




sns.distplot(df["carat"])
sns.distplot(df["Label_Cut"])
sns.distplot(df["Label_Color"])
sns.distplot(df["Label_Clarity"])
sns.distplot(df["depth"])
sns.distplot(df["table"])
sns.distplot(df["x"])
sns.distplot(df["y"])
sns.distplot(df["z"])

#sns.distplot(df[df.price == "Low"],color="r")
#sns.distplot(df[df.price == "Medium"],color="g")
#sns.distplot(df[df.price == "High"],color="g")
#sns.distplot(df[df.price == "Very High"],color="g")

# Locate features and label
X, y = data.iloc[: , :-1], data.iloc[: , -1]

#Scaling the data (Standardization)

X_s = preprocessing.scale(X)
X_s = pd.DataFrame(X, columns = X.columns) #converting scaled data into dataframe

y_s = preprocessing.scale(y)
y_s = pd.DataFrame(y, columns = y.columns) #ideally train, test data should be in columns


#Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_s, y_s, test_size=0.3, random_state=42)



clf = DecisionTreeClassifier #(max_depth=4, random_state=42)
clf.fit(X_train,y_train)
print("Accuracy of train:",clf.score(X_train,y_train))
print("Accuracy of test:",clf.score(X_test,y_test))

#Feature Importance
plt.figure(figsize=(12, 8))
importance = clf.feature_importances_
sns.barplot(x=importance, y=X_s.columns)
plt.show()

# Example EDA (distplot)


# Classification Report

pred = clf.predict(X_test)
print(classification_report(y_test,pred))
              
# Metrics
print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}".format(f1_score(y_test, pred,average='macro')))

# Confusion Matrix


cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels(categories, fontsize = 12)
ax.yaxis.set_ticklabels(categories, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()




