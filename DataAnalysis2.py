#http://blog.yhat.com/posts/logistic-regression-and-python.html
import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np

from sklearn.linear_model import LogisticRegression

df = pd.read_csv("D:\\Mercedes\\Mercedes\\train.csv")
df_test=pd.read_csv("D:\\Mercedes\\Mercedes\\test.csv")
df.head()

df.describe()

# frequency table cutting ID and y

pd.crosstab(df['ID'],df['y'], rownames=['ID'])

categorical_df=df.iloc[0:4209,0:10]

binary_df=df.iloc[0:4209,10:378]

ohe_features=pd.get_dummies(categorical_df)

data = ohe_features.join(binary_df)

#create training data
data_msk= np.random.rand(len(data)) < 0.8

data_training_data=data[data_msk]
data_test_data=data[~data_msk]

data_training_y=data_training_data['y']
#np.array(data_training_y).tolist()   #array to list

data_training_set = data_training_data.drop(["y"], axis=1)

data_test_data_y=data_test_data['y']

data_test_data_set = data_test_data.drop(["y"], axis=1)



############Random Forest
import matplotlib.pyplot as plt
from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)
#model.fit(data_training_set, data_training_y)
model.fit(data_training_set, (data_training_y).values.ravel())   # beCAUSE OF THE ERROR RECEIVED IN ABOVE STEP OF FIT USED RAVEL
feat_names = data_training_set.columns.values

#plot
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()

#preds = data_test_data_y[model.predict(data_test_data_set)]
preds = [model.predict(data_test_data_set)]











###################removing colums
# Columns containing the unique values :  [0]
# ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']





data_training_set_new = data_training_data.drop(['y','X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'], axis=1)



data_test_data_set_new = data_test_data.drop(['y','X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347'], axis=1)

model_new = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

model_new.fit(data_training_set_new, (data_training_y).values.ravel())
preds_new = [model_new.predict(data_test_data_set_new)]

feat_names_new = data_training_set_new.columns.values
#plot
importances_new = model_new.feature_importances_
std_new = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices_new = np.argsort(importances_new)[::-1][:20]
plt.figure(figsize=(12,12))
plt.title("Feature importances")
plt.bar(range(len(indices_new)), importances_new[indices_new], color="r", align="center")
plt.xticks(range(len(indices_new)), feat_names_new[indices_new], rotation='vertical')
plt.xlim([-1, len(indices_new)])
plt.show()



######################################



for col in data_training_set.columns:
    if col in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        data_training_set[col].astype(str).astype(int)


data_training_set['X0'].astype(str).astype(int)


np.array(data_training_y).tolist()
data_training_y=pd.DataFrame(data_training_y)
#logit = sm.Logit(data_training_y, data_training_set)

logistic = LogisticRegression()