9# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 19:26:39 2021
https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset

@author: francisco mena
"""

import os
os.chdir('C:\\Users\\franc\\projects\\letstf2gpu\\3_ RNNs')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

print( tf.config.list_physical_devices() )



#%% Functions to preprocess data
def preprocess(line):
    #8 features plus the target
    defaults = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    fields = tf.io.decode_csv(line, record_defaults=defaults)    #converts cvs line to tensor, using default values when a value is missing
    
    
    x_mean = np.array([5.45593655e+00, 1.42428122e+03, 2.95886657e+00,
                       3.89175860e+00,  2.86245478e+01,  5.45593655e+00,  1.09963474e+00,
            1.42428122e+03,  2.95886657e+00,  3.56464315e+01, -1.19584363e+02
            ]) #obtained previously from the dataset
    x_std = np.array([2.55049054e+00, 1.09580719e+03, 2.36148218e+00,
                      1.90935552e+00, 1.26414621e+01, 2.55049054e+00, 4.65480175e-01,
           1.09580719e+03, 2.36148218e+00, 2.13465865e+00, 2.00101922e+00
           ]) #obtained previously from the dataset
    
    fields = [(fields[i] - x_mean[i])/x_std[i]  for i in range(len(fields))]
    
    x = tf.stack(fields[:3])
    y = tf.stack(fields[:])
    
    x = tf.expand_dims(x, axis=-1)
    y = tf.expand_dims(y, axis=-1)
    
    return x, y




# Read data
def read_csv_files(filepaths, repeat=1, nskips = 0, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(nskips),    #Skip nskips lines, in case you have a header
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)



#%% Read data

filepath = os.path.join( os.getcwd(), 'datasets\\audio')

train_filepath = [os.path.join(filepath, x) for x in os.listdir(filepath) if 'train' in x]
valid_filepath = [os.path.join(filepath, x) for x in os.listdir(filepath) if 'val' in x]
test_filepath = [os.path.join(filepath, x) for x in os.listdir(filepath) if 'test' in x]

trainset = read_csv_files(train_filepath, repeat=None, n_parse_threads=tf.data.AUTOTUNE) 
validset = read_csv_files(valid_filepath, n_parse_threads=tf.data.AUTOTUNE)
testset = read_csv_files(test_filepath, n_parse_threads=tf.data.AUTOTUNE)



#%% Let's see a couple of items

for X_batch, y_batch in trainset.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()
    
    print("X shape =", X_batch.shape)
    print("y shape =", y_batch.shape)
    


#%% Let's visualize

Xsample = X_batch.numpy()
Ysample = y_batch.numpy()


fig, axes = plt.subplots(2,2, sharex=(True), sharey=(True), figsize = (8,6))
axes[0,0].plot(Ysample[0,:,0], ".-")
axes[1,0].plot(Ysample[1,:,0], ".-")
axes[0,1].plot(Ysample[2,:,0], ".-")
axes[1,1].plot(Ysample[3,:,0], ".-")

#Let's look at several at the same time
plt.figure()
plt.plot(Ysample[:,:,0].T)

#There-s a lot of variability


#%% Convert from 2D tensor to time series 3D tensor



#%% Let's build the Base Model

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


#Linear model

model_base = keras.models.Sequential([
            keras.layers.InputLayer(X_batch.shape[1:]),
            keras.layers.Flatten(),
            keras.layers.Dense(y_batch.shape[1])
    ])

model_base.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])

model_base.summary()

#%% Train base model

EPOCHS = 20 
BATCH = 32

# earlycall = keras.callbacks.EarlyStopping(patience = 5)

history_base = model_base.fit(trainset,
                              validation_data = validset,
                              batch_size = BATCH,
                              epochs = EPOCHS,
                              # callbacks = [earlycall]
                              )

#%% 

plt.plot( pd.DataFrame(history_base.history)[['loss', 'val_loss']] )
print( model_base.evaluate( testset ))    #3.2356

#%% let's try now the LSTM model

from tensorflow.keras.layers import LSTM, InputLayer, Bidirectional

tf.keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


model = keras.models.Sequential([
    InputLayer(X_batch.shape[1:]),
    Bidirectional( LSTM(5, return_sequences = True) ),
    Bidirectional( LSTM(10, return_sequences = True) ),
    Bidirectional( LSTM(10) ),
    keras.layers.Dense(7)    
    ])


model.compile(loss = 'mae', optimizer = 'adam', metrics = ['mae'])

model.summary()


#%%

history = model.fit(trainset,
                    validation_data = validset,
                    batch_size = BATCH,
                    epochs = EPOCHS,
                    # callbacks = [earlycall]
                    )

#%%

plt.plot( pd.DataFrame(history.history)[['loss', 'val_loss']] )
print( model.evaluate( testset ))    #3.26814


#%%
"""
Conclusion:
    A basic linear model seems to give just as good results 
"""

#%%
#%%
