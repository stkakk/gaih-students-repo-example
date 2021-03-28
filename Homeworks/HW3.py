# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 13:32:44 2021

@author: stk.akkaya
"""
# Import necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from sklearn.datasets import make_classification


X,y=make_classification(n_samples=10000, n_features=8,n_informative=5, class_sep=2, random_state=42)


# Generate dataset using make_classification function in the sklearn. 
# Convert it into pandas dataframe.


df_ranmat = pd.DataFrame(X, columns=['Property A', 'Property B', 'Property C','Property D','Property E', 'Property F', 'Property G','Property H',])
df_ranmat.head()

# Check duplicate values and missing data.

df_ranmat_dup=df_ranmat.duplicated()    #Duplication control

df_ranmat_na=df_ranmat.isna().sum()      #NaN control

df_ranmat_rev=df_ranmat.drop_duplicates()


# Visualize data for each feature (pairplot,distplot).

#sns.scatterplot(data=df_ranmat)

#dr_dist=df_ranmat[0,1:]#.columns('Property A')
#sns.distplot(dr_dist.T[0])
#sns.distplot(df_ranmat.columns(2))
#sns.distplot(data[data.Species == "Iris-versicolor"].PetalLengthCm,color="r")
#sns.distplot(data[data.Species == "Iris-virginica"].PetalLengthCm,color="g")


sns.pairplot(df_ranmat)


# Draw correlation matrix.
df_ranmat.corr()

# Handle outliers (you can use IsolationForest, Z-score, IQR)
# Outlier detection with Z-Score
from scipy import stats

df_ranmat_z = np.abs(stats.zscore(df_ranmat))


#Split dataset into train and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Import Decision Tree, define different hyperparamters and tune the algorithm.

from sklearn.tree import DecisionTreeClassifier



clf_G_Md = DecisionTreeClassifier(max_depth=4, random_state=42)
clf_G_Md.fit(X_train,y_train)
print("Accuracy of train for Gini with 4 of max_train=4:",clf_G_Md.score(X_train,y_train))
print("Accuracy of test for Gini with 4 of max_depth=4:",clf_G_Md.score(X_test,y_test))

clf_G = DecisionTreeClassifier( random_state=42)
clf_G.fit(X_train,y_train)
print("Accuracy of train for Gini with unlimited (default)  max_train:",clf_G.score(X_train,y_train))
print("Accuracy of test for Gini with unlimited (default)  max_train:",clf_G.score(X_test,y_test))



clf_IG_Md = DecisionTreeClassifier("entropy", max_depth=4, random_state=42)
clf_IG_Md.fit(X_train,y_train)
print("Accuracy of train for Information Gain with 4 of max_train=4:",clf_IG_Md.score(X_train,y_train))
print("Accuracy of test for Information Gain with 4 of max_depth=4:",clf_IG_Md.score(X_test,y_test))

clf_IG = DecisionTreeClassifier("entropy", random_state=42)
clf_IG.fit(X_train,y_train)
print("Accuracy of train for Information Gain with unlimited (default)  max_train:",clf_IG.score(X_train,y_train))
print("Accuracy of train for Information Gain with unlimited (default)  max_train:",clf_IG.score(X_test,y_test))


# Visualize feature importances.
#Feature Importance
plt.figure(figsize=(12, 8))
importance = clf_G.feature_importances_
sns.barplot(x=importance, y=df_ranmat.columns)
plt.show()                                 #Gini için yapılmış olup diğerleri de bulunacaktır.

# Create confusion matrix and calculate accuracy, recall, precision and f1 score.

# Classification Report
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report, f1_score
pred = clf_G.predict(X_test)
print(classification_report(y_test,pred))

# Metrics
print("Precision = {}".format(precision_score(y_test, pred, average='macro')))
print("Recall = {}".format(recall_score(y_test, pred, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, pred)))
print("F1 Score = {}".format(f1_score(y_test, pred,average='macro')))

# Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(12, 8))
ax =sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels(df_ranmat.columns, fontsize = 12)
ax.yaxis.set_ticklabels(df_ranmat.columns, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


# Import XGBoostClassifier, define different hyperparamters and tune the algorithm.
import xgboost as xgb

dmatrix_train = xgb.DMatrix(data=X_train, label=y_train)
dmatrix_test = xgb.DMatrix(data=X_test, label=y_test)

param = {'max_depth':3, 
         'eta':1, 
         'objective':'multi:softprob', 
         'num_class':3}

num_round = 5
model = xgb.train(param, dmatrix_train, num_round)

preds = model.predict(dmatrix_test)

best_preds = np.asarray([np.argmax(line) for line in preds])



# Visualize feature importances.

#Feature Importance
plt.figure(figsize=(12, 8))
importance = clf_G.feature_importances_
sns.barplot(x=importance, y=df_ranmat.columns)
plt.show()   


# Create confusion matrix and calculate accuracy, recall, precision and f1 score.


print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))

from sklearn.metrics import confusion_matrix

plt.figure(figsize=(12, 8))
cm = confusion_matrix(y_test, best_preds)
ax = sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels(df_ranmat.columns, fontsize = 12)
ax.yaxis.set_ticklabels(df_ranmat.columns, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()


# Evaluate your result and select best performing algorithm for our case.

## Hyperparameter Tuning

from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  

param_dict = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2),
    'learning_rate': [0.00001,0.001,0.01,0.1,1,2],
    'n_estimators': [10,190,200,210,500,1000,2000]
    
}

xgc = XGBClassifier(booster='gbtree', learning_rate =0.01, n_estimators=200, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softprob', nthread=4, scale_pos_weight=1, seed=27)

clf = GridSearchCV(xgc,param_dict,cv=3, n_jobs = -1).fit(X_train,y_train)

print("Tuned: {}".format(clf.best_params_)) 
print("Mean of the cv scores is {:.6f}".format(clf.best_score_))
print("Train Score {:.6f}".format(clf.score(X_train,y_train)))
print("Test Score {:.6f}".format(clf.score(X_test,y_test)))
print("Seconds used for refitting the best model on the train dataset: {:.6f}".format(clf.refit_time_))



plt.figure(figsize=(12, 8))

xgb_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, xgb_pred)
ax = sns.heatmap(cm, square=True, annot=True, cbar=False)
ax.xaxis.set_ticklabels(df_ranmat.columns, fontsize = 12)
ax.yaxis.set_ticklabels(df_ranmat.columns, fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels',fontsize = 15)
ax.set_ylabel('True Labels',fontsize = 15)
plt.show()
