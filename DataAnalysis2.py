#http://blog.yhat.com/posts/logistic-regression-and-python.html
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

from sklearn.linear_model import LogisticRegression

df = pd.read_csv("D:\\Mercedes\\Mercedes\\train.csv")
df.head()

df.describe()

# frequency table cutting ID and y

pd.crosstab(df['ID'],df['y'], rownames=['ID'])

categorical_df=df.iloc[0:4209,0:8]

binary_df=df.iloc[0:4209,8:378]

ohe_features=pd.get_dummies(categorical_df)

data = ohe_features.join(binary_df)

#create training data
data_msk= np.random.rand(len(data)) < 0.8

data_training_data=data[data_msk]
data_test_data=data[~data_msk]

data_training_y=data_training_data['y'].values
np.array(data_training_y).tolist()
#array to list
data_training_set = data_training_data.drop(["y"], axis=1)



for col in data_training_set.columns:
    if col in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        data_training_set[col].astype(str).astype(int)


data_training_set['X0'].astype(str).astype(int)


np.array(data_training_y).tolist()

#logit = sm.Logit(data_training_y, data_training_set)

logistic = LogisticRegression()