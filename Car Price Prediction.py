
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import sklearn
df = pd.read_csv("car sale.csv")
df
df.head()
df.info()
df.describe()
df.columns
df.drop(['New_Price'], axis=1, inplace=True)
df
df.drop(['Name', 'Location', 'Unnamed: 0'], axis=1, inplace=True)
df
df.head(2)

# Data Analysis
df.isna().sum()
#Fill all the numeric rows with the median
for label, content in df.items():
    if pd.api.types.is_numeric_dtype(content): #Checks whether the provided dtype is of a numeric dtype or not.
        if pd.isnull(content).sum():
            # Fill missing numeric values with median
            df[label] = content.fillna(content.median())
df.isna().sum()

#String values to Categorical values
for label, content in df.items():
    if pd.api.types.is_string_dtype(content):
        df[label] = content.astype("category")
#categorical into numericals
for label, content in df.items():
    if not pd.api.types.is_numeric_dtype(content):
        df[label] = pd.Categorical(content).codes
df.isna().sum()
df.info()
# Data visualization
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True)
plt.show()
sns.pairplot(df)

#Splitting the train and test data

from sklearn.model_selection import train_test_split
X = df.drop("Price", axis=1)
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train
y_test

#modeling
#importing models and model Evaluators
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

models = {"Linear Regression": LinearRegression(),
          "Random Forest": RandomForestRegressor(),
          "Decision tree": DecisionTreeRegressor()}

#Fit the data
def fit_and_score(models, X_train, X_test, y_train, y_test):
    #random seed for reproducible results
    np.random.seed(101)
    model_scores={}
    for name, model in models.items():
        model.fit(X_train, y_train)
        model_scores[name] = model.score(X_test, y_test)
    return model_scores 
#Evaluating the Model Scores
model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores
#Comparing the models
model_compare = pd.DataFrame(model_scores, index=['score'])
model_compare.plot.bar();
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
model.score(X_test, y_test)
#Predictions
prediction = model.predict(X_test)
prediction

y_test

#Evaluation of model
from sklearn import metrics
print("Mean squared error: %.2f"% metrics.mean_squared_error(y_test, prediction))
print("Mean absolute error: %.2f"% metrics.mean_absolute_error(y_test, prediction))
print('R_square score: %.2f' % metrics.r2_score(y_test, prediction))

import pickle
pickle.dump(model, open('random_forest_regression_model.pkl', 'wb'))
