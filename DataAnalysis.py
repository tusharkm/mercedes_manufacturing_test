import pandas as pd  # data processing , file read from CSV
import numpy as np #linear algebra
import matplotlib.pyplot as plt
import math as mt
import seaborn as sns



# Input data files are available in the directory.

from subprocess import check_output
print(check_output(["ls"]).decode("utf8"))

from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA

seed =45;

train = pd.read_csv("train.csv")
test =pd.read_csv("test.csv")

#Number of columns and Rows in a CSV
len(list(train))    #378
len(train)          #4209

len(list(test))    #377
len(test)          #4209

#or
print("Train shape : ", test.shape)  #Train shape :  (4209, 378)
print("Train shape : ", train.shape)

train.head()

#plot output variable y
plt.figure(figsize=(8,6))
plt.scatter(range(train.shape[0]), np.sort(train.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.show()

# plot distribution of y
plt.figure(figsize=(12,8))
sns.distplot(train.y.values, bins=50, kde=False)
plt.xlabel('y value', fontsize=12)
plt.show()


# data type of all the variables present in the dataset.
dtype_df = train.dtypes.reset_index()  #data type of all variable
dtype_df.columns = ["Count", "Column Type"]    #give column header
dtype_df.groupby("Column Type").aggregate('count').reset_index() #count the variable based on column type


#op
# Column Type  Count
# 0       int64    369
# 1     float64      1
# 2      object      8


# missing values
missing_df = train.isnull().sum(axis=0).reset_index()
#A 2-dimensional array has two corresponding axes: the first running vertically downwards across rows (axis 0), and the second running horizontally across columns (axis 1).
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count']>0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df




# unique values
unique_values_dict = {}

for col in train.columns:
    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        unique_value = str(np.sort(train[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist[:]

for unique_val, columns in unique_values_dict.items():
    print("Columns containing the unique values : ", unique_val)
    print(columns)
    print("--------------------------------------------------")

# Columns containing the unique values :  [0]
# ['X11', 'X93', 'X107', 'X233', 'X235', 'X268', 'X289', 'X290', 'X293', 'X297', 'X330', 'X347']

# Explore the categorical columns present in the dataset.

var_name = "X0"
col_order = np.sort(train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.stripplot(x=var_name, y='y', data=train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


# Explore the categorical columns present in the dataset with boxplot.
var_name = "X2"
col_order = np.sort(train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()



# Explore the categorical columns present in the dataset with violinplot.
var_name = "X3"
col_order = np.sort(train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.violinplot(x=var_name, y='y', data=train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


var_name = "X4"
col_order = np.sort(train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.violinplot(x=var_name, y='y', data=train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()

# Explore the categorical columns present in the dataset with boxplot.

var_name = "X5"
col_order = np.sort(train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


var_name = "X6"
col_order = np.sort(train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


var_name = "X8"
col_order = np.sort(train[var_name].unique()).tolist()
plt.figure(figsize=(12,6))
sns.boxplot(x=var_name, y='y', data=train, order=col_order)
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()


#Binary Variables: bar graph
zero_count_list = []
one_count_list = []
cols_list = unique_values_dict['[0, 1]']
N = len(cols_list)

ind = np.arange(N)
width = 0.35
plt.figure(figsize=(6,100))

p1 = plt.barh(ind, zero_count_list, width, color='red')
p2 = plt.barh(ind, one_count_list, width, left=zero_count_list, color="blue")
plt.yticks(ind, cols_list)
plt.legend((p1[0], p2[0]), ('Zero count', 'One Count'))
plt.show()


# ID variable:
var_name = "ID"
plt.figure(figsize=(12,6))
sns.regplot(x=var_name, y='y', data=train, scatter_kws={'alpha':0.5, 's':30})
plt.xlabel(var_name, fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of y variable with "+var_name, fontsize=15)
plt.show()

#IDs are distributed across train and test.
plt.figure(figsize=(6,10))
train['eval_set'] = "train" # create column of eval_set with test and train
test['eval_set'] = "test"
full_df = pd.concat([train[["ID","eval_set"]], test[["ID","eval_set"]]], axis=0)

plt.figure(figsize=(12,6))
sns.violinplot(x="eval_set", y='ID', data=full_df)
plt.xlabel("eval_set", fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Distribution of ID variable with evaluation set", fontsize=15)
plt.show()


#Random Forest model and check the important variables.
from sklearn import ensemble
model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

train_y = train['y'].values
train_X = train.drop(["ID", "y", "eval_set",'X0','X1','X2','X3','X4','X5','X6','X8','X10'], axis=1)
train_X.head()

model.fit(train_X, train_y)
feat_names = train_X.columns.values

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



####################

###########
# ID       y  X0 X1  X2 X3 X4 X5 X6 X8  ...   X375  X376  X377  X378  X379  \
# 0   0  130.81   k  v  at  a  d  u  j  o  ...      0     0     1     0     0
# 1   6   88.53   k  t  av  e  d  y  l  o  ...      1     0     0     0     0
# 2   7   76.26  az  w   n  c  d  x  j  x  ...      0     0     0     0     0
# 3   9   80.62  az  t   n  f  d  x  l  e  ...      0     0     0     0     0


# get id and target variables, iloc gives the data(http://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/)
trainId = train.iloc[0:4209,0]
testId =test.iloc[0:4209,0]
trainOp=train.iloc[0:4209,1]

#remove the column id and target(https://chrisalbon.com/python/pandas_dropping_column_and_rows.html)
train=train.drop('ID',axis=1)
train=train.drop('y',axis=1)
test=test.drop('ID',axis=1)

#Merge 2 test and train dataset (https://stackoverflow.com/questions/14988480/pandas-version-of-rbind)
df_all=train.append(pd.DataFrame(data = test), ignore_index=True)

len((df_all))      #8418

df_all    #[8418 rows x 376 columns]


# split groups of categorical and binary features
#categorical_variable=['X0','X1','X2','X3','X4','X5','X6','X8','X10']


categorical_df=df_all.iloc[0:8418,0:10]  #[8418 rows x 8 columns]


binary_df=df_all[0:8418,10:376]  #[8418 rows x 368 columns]

#One-hot encoding
ohe_features=pd.get_dummies(categorical_df)  #[8418 rows x 211 columns]

#concat colums with ohe
df_binary_ohe=pd.concat([binary_df,ohe_features],axis=1)  #[8418 rows x 579 columns]
df_all_ohe=pd.concat([df_all,ohe_features],axis=1) #[8418 rows x 587 columns]

