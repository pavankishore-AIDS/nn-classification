# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![](nn.png)

## DESIGN STEPS

- STEP 1: Import the necessary packages & modules
- STEP 2: Load and read the dataset
- STEP 3: Perform pre processing and clean the dataset
- STEP 4: Encode categorical value into numerical values using ordinal/label/one hot encoder modules
- STEP 5: Split and Scale the data to training and testing
- STEP 6: Train the data using Dense module in tensorflow
- STEP 7: Gather the trainling loss and classification metrics


## PROGRAM
```
Program by: Pavan Kishore.M
Reg no: 212221230076
```
```python
#importing packages 
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
```
```python
#loading the dataset
df_initial = pd.read_csv('customers.csv')
```
```python
# EDA
df_initial.columns
df_initial.dtypes
df_initial.shape
df_initial.isnull().sum()
```
```python
#data cleaning
df_cleaned = df_initial.dropna(axis=0)
df_cleaned.isnull().sum()
```
```python
# EDA before Data Encoding
df_cleaned['Gender'].unique()
df_cleaned['Ever_Married'].unique()
df_cleaned['Graduated'].unique()
df_cleaned['Profession'].unique()
df_cleaned['Spending_Score'].unique()
df_cleaned['Var_1'].unique()
df_cleaned['Segmentation'].unique()
```
```python
#data encoding
categories_lst=[['Male', 'Female'],
            ['No', 'Yes'],
            ['No', 'Yes'],
            ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor','Homemaker', 'Entertainment', 'Marketing', 'Executive'],
            ['Low', 'High', 'Average']]
enc = OrdinalEncoder(categories=categories_lst)
df1=df_cleaned.copy()
df1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']] = enc.fit_transform(df1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])
df1.dtypes

le = LabelEncoder()
df1['Segmentation'] = le.fit_transform(df1['Segmentation'])
df1.dtypes

df1.describe()
df1['Segmentation'].unique()
X = df1[['Gender',
         'Ever_Married',
         'Age',
         'Graduated',
         'Profession',
         'Work_Experience',
         'Spending_Score',
         'Family_Size']].values

y1=df1[['Segmentation']].values
oh_enc = OneHotEncoder()
oh_enc.fit(y1)
y1.shape
y=oh_enc.transform(y1).toarray()
y.shape

y1[0]
y[0]
X.shape
```
```python
#splitting the data
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
```
```python
#scaling the data
scaler = MinMaxScaler()
scaler.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

X_train_scaled[:,2] = scaler.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler.transform(X_test[:,2].reshape(-1,1)).reshape(-1)
```
```python
# Creating the model
ai_brain = Sequential([
    Dense(8,input_shape=(8,)),
    Dense(16, activation ='relu'),
    Dense(16),
    Dense(8, activation ='relu'),
    Dense(4,activation='softmax')
])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2)
```
```python
#running the model
ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,batch_size=256,
             validation_data=(X_test_scaled,y_test))
```
```python
#getting metrics
metrics = pd.DataFrame(ai_brain.history.history)
metrics.head()
```
```python
#plotting loss & accuracy
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
```
```python
metrics[['loss','val_loss']].plot()
metrics[['accuracy','val_accuracy']].plot()
```
```python
#classification metrics
print(confusion_matrix(y_test_truevalue,x_test_predictions))

print(classification_report(y_test_truevalue,x_test_predictions))
```
```python
#Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)

print(le.inverse_transform(x_single_prediction))
```


## Dataset Information
![1](https://github.com/pavankishore-AIDS/nn-classification/assets/94154941/79cf3190-31cb-4619-bf8b-f4c594ae6fdb)


## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![2](https://github.com/pavankishore-AIDS/nn-classification/assets/94154941/e84eee74-61d5-47c3-9a59-5b6c0f317197)



### Classification Report
![3](https://github.com/pavankishore-AIDS/nn-classification/assets/94154941/37baf503-c6f0-4bc3-b9fc-d6b68e9f29f3)


### Confusion Matrix
![4](https://github.com/pavankishore-AIDS/nn-classification/assets/94154941/1414695a-5166-49c1-8fd5-c627c2c07d6e)


### New Sample Data Prediction
![5](https://github.com/pavankishore-AIDS/nn-classification/assets/94154941/9842270b-9ae5-40d0-bc67-3c9808bc528b)

## RESULT
Thus a Neural Network Classification Model is created and executed successfully
