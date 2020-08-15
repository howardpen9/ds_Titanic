# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# TOP)C


https://code.visualstudio.com/docs/python/data-science-tutorial


Tatanic Analysis:
Aug 15, 2020


# %%
import pandas as pd
import numpy as np
data = pd.read_csv('data.csv')


# %%
data.describe()


# %%
data.describe()


# %%
data.replace('?', np.nan, inplace= True)
data = data.astype({"age": np.float64, "fare": np.float64})


# %%
## 轉置性別。目前 Gender 為 String 
data.replace({'male': 1, 'female': 0}, inplace=True)


# %%
data.head(5)


# %%
# 計算存活率
data.corr().abs()[["survived"]]


# %%
# Calculate whether the Parents & Childrens are the key metrics
data['relatives_in_mind'] = data.apply (lambda row: int((row['sibsp'] + row['parch']) > 0), axis=1)
data.corr().abs()[["survived"]]


# %%
data = data[['sex', 'pclass','age','relatives_in_mind','fare','survived']].dropna()


# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[['sex','pclass','age','relatives_in_mind','fare']]
, data.survived
, test_size=0.2
, random_state=0)


# %%
from sklearn.preprocessing import StandardScaler
## 分 Traning Sets / Test Set
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


# %%
from sklearn.naive_bayes import GaussianNB
## Choose a better Algo to 
model = GaussianNB()
model.fit(X_train, y_train)


# %%
from sklearn import metrics

predict_test = model.predict(X_test)
print(metrics.accuracy_score(y_test, predict_test))

# %% [markdown]
# 
# # Use a neural network to increase accuracy 
# 
# 
# A neural network is a model that uses **weights and activation functions**, *modeling* aspects of human neurons, to determine an outcome based on provided inputs.

# %%
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()


model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu', input_dim = 5))
model.add(Dense(5, kernel_initializer = 'uniform', activation = 'relu'))
model.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# %% [markdown]
# **The rectified linear unit (relu) activation function** is used as a good general activation function for the first two layers, while the sigmoid activation function is required for the final layer as the output you want (of whether a passenger survives or not) needs to be scaled in the range of 0-1 (the probability of a passenger surviving).

# %%
model.summary()


# %%
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=50)

# %% [markdown]
# ### With the model built and trained its now time to see how it performs against the test data.

# %%
y_pred = model.predict_classes(X_test)
print(metrics.accuracy_score(y_test, y_pred))


# %%



# %%



# %%



# %%
import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(ncols=5, figsize=(30,5))
sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs[0])
sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[1])
sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[2])
sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[3])
sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[4])

