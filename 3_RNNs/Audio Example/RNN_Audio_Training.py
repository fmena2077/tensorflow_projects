# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 19:26:39 2021

@author: francisco mena
"""

import os
os.chdir('C:\\Users\\franc\\projects\\letstf2gpu\\3_ RNNs')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
np.random.seed(42)
tf.random.set_seed(42)

print( tf.config.list_physical_devices('GPU') )


#%% Read data

# Let's read 10k lines, since it's an example. The whole training takes a long time
df = pd.read_csv('RNN_Audio_CleanData.csv', nrows = 1E5,
                 usecols = ['L500k', 'L1k', 'L2k', 'L3k', 'L4k', 'L6k', 'L8k'])
print(df.shape)

#%% Let's visualize

fig, axes = plt.subplots(2,2, sharex=(True), sharey=(True), figsize = (8,6))
axes[0,0].plot(df.iloc[0], ".-")
axes[1,0].plot(df.iloc[1], ".-")
axes[0,1].plot(df.iloc[2], ".-")
axes[1,1].plot(df.iloc[3], ".-")

#Let's look at several at the same time
plt.figure()
plt.plot(df.sample(20).T)

#There-s a lot of variability

#%% Split and Normalize

df.columns

# X = df[['L2k', 'L4k', 'L6k']].copy()
# y = df.copy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(df[['L2k', 'L4k', 'L6k']], df, test_size=0.2, random_state=42, shuffle = True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, shuffle = True)


scale = StandardScaler()
X_train = scale.fit_transform(X_train)
X_val = scale.transform(X_val)
X_test = scale.transform(X_test)

# let's save some memory
del df


#%% Convert from matrix to time series

X_train = X_train[:,:,np.newaxis]
X_val = X_val[:,:,np.newaxis]
X_test = X_test[:,:,np.newaxis]
y_train = y_train.values[:,:,np.newaxis]
y_val = y_val.values[:,:,np.newaxis]
y_test = y_test.values[:,:,np.newaxis]


#%% Let's build the Base Model

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


#Linear model

model_base = keras.models.Sequential([
            keras.layers.InputLayer(X_train.shape[1:]),
            keras.layers.Flatten(),
            keras.layers.Dense(y_train.shape[1])
    ])

model_base.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])

model_base.summary()

#%% Train base model

EPOCHS = 20 
BATCH = 32

# earlycall = keras.callbacks.EarlyStopping(patience = 5)

history_base = model_base.fit(X_train, y_train,
                              validation_data = (X_val, y_val),
                              batch_size = BATCH,
                              epochs = EPOCHS,
                              # callbacks = [earlycall]
                              )

#%% 

plt.plot( pd.DataFrame(history_base.history)[['loss', 'val_loss']] )
print( model_base.evaluate( X_test, y_test))    #3.2356

#%% let's try now the LSTM model

from tensorflow.keras.layers import LSTM, InputLayer, Bidirectional

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential([
    InputLayer(X_train.shape[1:]),
    Bidirectional( LSTM(5, return_sequences = True) ),
    Bidirectional( LSTM(10, return_sequences = True) ),
    Bidirectional( LSTM(10) ),
    keras.layers.Dense(7)    
    ])


model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])

model.summary()


#%%

history = model.fit(X_train, y_train,
                    validation_data = (X_val, y_val),
                    batch_size = BATCH,
                    epochs = EPOCHS,
                    # callbacks = [earlycall]
                    )

#%%

plt.plot( pd.DataFrame(history.history)[['loss', 'val_loss']] )
print( model.evaluate( X_test, y_test))    #3.26814


#%%
"""
Conclusion:
    A basic linear model seems to give just as good results 
"""

#%%
#%%
