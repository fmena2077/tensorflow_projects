# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 19:30:07 2021
Read data from different csv files using Tensorflow Dataset

@author: francisco mena
"""

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.data import AUTOTUNE


#%%
tf.test.is_built_with_cuda()
tf.config.list_physical_devices()

tf.random.set_seed(42)



#%% Preprocess function

def preprocess(line):
    #8 features plus the target
    defaults = [0., 0., 0., 0., 0., 0., 0., 0., tf.constant([], dtype=tf.float32)]
    fields = tf.io.decode_csv(line, record_defaults=defaults)    #converts cvs line to tensor, using default values when a value is missing
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])

    x_mean = [3.89175860e+00,  2.86245478e+01,  5.45593655e+00,  1.09963474e+00,
            1.42428122e+03,  2.95886657e+00,  3.56464315e+01, -1.19584363e+02
            ] #obtained previously from the dataset
    x_std = [1.90935552e+00, 1.26414621e+01, 2.55049054e+00, 4.65480175e-01,
           1.09580719e+03, 2.36148218e+00, 2.13465865e+00, 2.00101922e+00
           ] #obtained previously from the dataset
    
    return (x - x_mean)/x_std, y




# Read data
def read_csv_files(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


#%%
os.chdir("C:\\Users\\franc\\projects\\letstf2gpu\\CNN")

filepath = os.path.join( os.getcwd(), 'datasets\\housing')

train_filepath = [os.path.join(filepath, x) for x in os.listdir(filepath) if 'train' in x]
valid_filepath = [os.path.join(filepath, x) for x in os.listdir(filepath) if 'valid' in x]
test_filepath = [os.path.join(filepath, x) for x in os.listdir(filepath) if 'test' in x]

trainset = read_csv_files(train_filepath, repeat=None, n_parse_threads=tf.data.AUTOTUNE) 
validset = read_csv_files(valid_filepath, n_parse_threads=tf.data.AUTOTUNE)
testset = read_csv_files(test_filepath, n_parse_threads=tf.data.AUTOTUNE)



#%% Let's see a couple of items

for X_batch, y_batch in trainset.take(2):
    print("X =", X_batch)
    print("y =", y_batch)
    print()
    
X_batch.shape


#%% Now that we have the data, let's model

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape= X_batch.shape[1:]),
    keras.layers.Dense(1),
])

model.compile(loss = 'mse', optimizer = keras.optimizers.SGD(lr=1e-3), metrics = ["mae"])

#%% Train
earlystop = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights=True)

batch_size=32
history = model.fit(trainset, 
                    steps_per_epoch = 300,    #should be len(X_train)//batch_size
                    epochs = 100,
                    validation_data = validset)


#%%
model.evaluate(testset)

ypred = model.predict(testset)
ypred = [x[0] for x in ypred]

ylist = []
for x,y in iter(testset):
    y = [xx[0] for xx in y]
    y = [xx.numpy() for xx in y]
    ylist = ylist + y

#%%
import pandas as pd
import matplotlib.pyplot as plt

pd.Series(ypred).describe()
pd.Series(ylist).describe()


plt.figure(figsize = (6,6))
plt.scatter(ylist, ypred)
plt.xlim(0,8)
plt.ylim(0,8)
plt.xlabel('Real Value', fontsize = 12)
plt.ylabel('Predicted Value', fontsize = 12)
