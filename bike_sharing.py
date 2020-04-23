#!/usr/bin/env python
# coding: utf-8


#import libraries
import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


bike_df = pd.read_csv('bike.csv')


# getting rid of unneccessary columns
unneccessary_cols = ['dteday','casual','registered','instant']
bike_df.drop(labels=unneccessary_cols,axis=1,inplace=True)



# splitting data
from sklearn.model_selection import train_test_split
X = bike_df.drop('cnt',axis=1).values
y = bike_df['cnt'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)


# scaling data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



# create neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')


# training
model.fit(X_train,y_train,epochs=10000,verbose=1)


# evaluation
riders_predictions = model.predict(X_test)


results_df = pd.DataFrame(data={'Actual':y_test,'Predicted':riders_predictions[:,0]})

result_img = sns.lmplot(x='Actual',y='Predicted',data=results_df)
result_img.savefig('result.png')



