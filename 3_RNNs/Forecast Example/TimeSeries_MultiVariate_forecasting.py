# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 16:32:53 2022

Script to read time series data from a CSV file and forecast 10 timesteps into the future

@author: Francisco Mena
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow .keras import optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import random

print(tf.config.list_physical_devices())

#Set seed
random.seed(2700)
np.random.seed(2700)

#%% Read data


os.chdir("C:\\Users\\franc\\projects\\letstf2gpu\\EY")
currdir = os.getcwd()
filepath = os.path.join(currdir, "data", "ENGINE_1_clean.csv")

df = pd.read_csv(filepath, index_col=None)

print(df.head())

#%% Train Test split

n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

#%% Normalization

meanx = train_df.mean()
stdx = train_df.std()

train_df = (train_df - meanx)/stdx 
val_df = (val_df - meanx)/stdx 
test_df = (test_df - meanx)/stdx 

#%% From dataframe to 3-dim tensor for RNN

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
               train_df = train_df, val_df = val_df, test_df = test_df,
               label_columns=None):
        """
        Function to create windows for time series

        Parameters
        ----------
        input_width : width of input data

        label_width : width of target 
            
        shift : how many time steps after the end of input data, to start target (label)
            
        train_df : train dataframe
            
        val_df : validation dataframe
            
        test_df : test dataframe
            
        label_columns : name of label/target 

        Returns
        -------
        Creates object

        """
        
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                        enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])
    
    
def split_window(self, features):
    """
    Function to split the time series into inputs and labels
    """
        
    inputs = features[:, self.input_slice, :]
    labels = features[:, self.labels_slice, :]
    if self.label_columns is not None:
      labels = tf.stack(
          [labels[:, :, self.column_indices[name]] for name in self.label_columns],
          axis=-1)
    
    # Slicing doesn't preserve static shape information, so set the shapes
    # manually. This way the `tf.data.Datasets` are easier to inspect.
    inputs.set_shape([None, self.input_width, None])
    labels.set_shape([None, self.label_width, None])
    
    return inputs, labels

WindowGenerator.split_window = split_window

#%%

w = WindowGenerator(input_width=30, label_width=10, shift=10,
                     label_columns=['Temperature_Engine'])
print( w )

#%%
# Stack three slices, the length of the total window.
example_window = tf.stack([np.array(train_df[:w.total_window_size]),
                           np.array(train_df[100:100+w.total_window_size]),
                           np.array(train_df[200:200+w.total_window_size])])

example_inputs, example_labels = w.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')


#%% Let's visualize it

w.example = example_inputs, example_labels

def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
    inputs, labels = self.example
    plt.figure(figsize=(12, 8))
    plot_col_index = self.column_indices[plot_col]
    max_n = min(max_subplots, len(inputs))
    for n in range(max_n):
        plt.subplot(max_n, 1, n+1)
        plt.ylabel(f'{plot_col} [normed]')
        plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                 label='Inputs', marker='.', zorder=-10)
    
        if self.label_columns:
            label_col_index = self.label_columns_indices.get(plot_col, None)
        else:
            label_col_index = plot_col_index
    
        if label_col_index is None:
            continue
    
        plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
        if model is not None:
            predictions = model(inputs)
    
            if len(predictions.shape)<3:
                plt.scatter(self.label_indices, predictions[n, :],
                              marker='X', edgecolors='k', label='Predictions',
                              c='#ff7f0e', s=64)
            else:    
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                          marker='X', edgecolors='k', label='Predictions',
                          c='#ff7f0e', s=64)
    
        if n == 0:
          plt.legend()

    plt.xlabel('Time [h]')

WindowGenerator.plot = plot

w.plot(plot_col = "Temperature_Engine")

#%% To Dataset

# From dataframe to tf data

def make_dataset(self, data):
    data = np.array(data, dtype=np.float32)
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=32,)

    ds = ds.map(self.split_window)

    return ds

WindowGenerator.make_dataset = make_dataset


@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example


#%% Let's see some of the data

print( w.train.element_spec )

#%%

for example_inputs, example_labels in w.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')
  
#%% TRAINING

#Since we'll train a bunch of models, let's build a training function


def compile_and_fit(model, window, patience=2, EPOCHS = 10, model_name = "my_checkpoint.h5"):

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model_checkpoint = keras.callbacks.ModelCheckpoint(
    model_name, save_best_only=True)

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping, model_checkpoint])
    return history


#%% Lets start with a simple dense model, no regularization


# single_step_window = WindowGenerator(
#     input_width=1, label_width=1, shift=1,
#     label_columns=['Temperature_Engine'])
# single_step_window

#%% DENSE MODEL
tf.keras.backend.clear_session() 

dense = keras.Sequential([
    keras.layers.Flatten(),     #We need a flatten first cause the data is shaped [batch, timesteps, features]
    keras.layers.Dense(units = 24, activation = "relu"),
    keras.layers.Dense(units = 12, activation = "relu"),
    keras.layers.Dense(units = 6, activation = "relu"),
    keras.layers.Dense(units = 10),
    ])


history = compile_and_fit(dense, window = w)

val_performance = {}
performance = {}

#%%
val_performance['Dense'] = dense.evaluate(w.val)
performance['Dense'] = dense.evaluate(w.test, verbose=0)


#%% CONV MODEL

conv_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(30,),   #The convolution needs to be the same size as the input window to work
                           activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=10),
])

history_conv = compile_and_fit(conv_model, window = w)


#%%

val_performance['Conv'] = conv_model.evaluate(w.val)
performance['Conv'] = conv_model.evaluate(w.test, verbose=0)


#%% CONV MODEL with Flatten

conv_model_smallKernel = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=32,
                           kernel_size=(3,),   #The convolution needs to be the same size as the input window to work
                           activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=10),
])

history_conv_smallKernel = compile_and_fit(conv_model_smallKernel, window = w)

#%%

val_performance['Conv_SmallKernel'] = conv_model_smallKernel.evaluate(w.val)
performance['Conv_SmallKernel'] = conv_model_smallKernel.evaluate(w.test, verbose=0)


#%% LSTM MODEL

tf.keras.backend.clear_session()

lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(16, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.LSTM(16, return_sequences=False),
    tf.keras.layers.Dense(units=10)
])

print('Input shape:', w.example[0].shape)
print('Output shape:', lstm_model(w.example[0]).shape)

history_lstm = compile_and_fit(lstm_model, w)


#%%

val_performance['LSTM'] = lstm_model.evaluate(w.val)
performance['LSTM'] = lstm_model.evaluate(w.test, verbose=0)


#%%

x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [Engine Temperature, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()

#%%

for name, value in performance.items():
  print(f'{name:12s}: {value[1]:0.4f}')

#%%

w.plot(model = lstm_model, plot_col = "Temperature_Engine")

w.plot(model = dense, plot_col = "Temperature_Engine")

ypred = lstm_model.predict(w.val)

ypred.shape

#%%
