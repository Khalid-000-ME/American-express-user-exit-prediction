# -*- coding: utf-8 -*-
"""

Corresponding COLAB file is located at
    https://colab.research.google.com/drive/1zmpzrbO1EkdbHM3hneqatq_ADJmlU62T

"""

"""
Importing the libraries
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


"""
Importing Dataset
"""

data_set = pd.read_csv("data_set.csv")
X = data_set.iloc[:, :-1].values
y = data_set.iloc[:, -1].values

"""
Encoding Categorical Data
"""

label_encoder = LabelEncoder()
X[:, 2] = label_encoder.fit_transform(X[:, 2])


ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

"""
Splitting data into Test set & Training Set
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
Feature Scaling
"""

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

"""
**ANN**
"""

"""
Initialization
"""

ann = tf.keras.models.Sequential()

"""
Adding input layer as first hidden layer
"""

ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

"""
Adding second hidden layer
"""

ann.add(tf.keras.layers.Dense(units=5, activation='relu'))

"""
Adding output layer
"""

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

"""
**ANN TRAINING**
"""

"""
ANN Compiling
"""

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )

"""
ANN Training
"""

ann.fit(X_train, y_train, batch_size=32, epochs=120)


"""
Predicting y values using test data
"""

y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_test), 1), y_test.reshape(len(y_pred), 1)), 1))

"""
Confusion Matrix
"""

cm = confusion_matrix(y_test, y_pred)

"""
Accuracy
"""

accuracy_score(y_test, y_pred)
