#Import Boston Dataset from sklearn dataset class.
from sklearn.datasets import load_boston
import pandas as pd

#Explore and analyse raw data.
Xb,yb =load_boston(return_X_y=True)

df_boston = pd.DataFrame(Xb,columns = load_boston().feature_names)
df_boston.head()


#Do preprocessing for regression.
df_boston.isna()
df_boston.isna().sum()



#Split your dataset into train and test test (0.7 for train and 0.3 for test).
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Xb,yb, test_size=0.3, random_state=42)


from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import r2_score

# Simple Linear Model

#Fit simple linear model and find coefficients
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

#print(f'Regression model coef: {regression_model.coef_}')



alpha=10

ridge_model = Ridge(alpha)
ridge_model.fit(X_train, y_train)

print(f'Ridge model coef: {ridge_model.coef_}')
#As the data has 10 columns hence 10 coefficients appear here

#for lasso regression
lasso_model = Lasso(alpha)
lasso_model.fit(X_train, y_train)

print(f'Lasso model coef: {lasso_model.coef_}')
#As the data has 10 columns hence 10 coefficients appear here

#Simple Linear Model
print("Simple Train: ", regression_model.score(X_train, y_train))
print("Simple Test: ", regression_model.score(X_test, y_test))
print('*************************')
#Lasso
print("Lasso Train: ", lasso_model.score(X_train, y_train))
print("Lasso Test: ", lasso_model.score(X_test, y_test))
print('*************************')
#Ridge
print("Ridge Train: ", ridge_model.score(X_train, y_train))
print("Ridge Test: ", ridge_model.score(X_test, y_test))
