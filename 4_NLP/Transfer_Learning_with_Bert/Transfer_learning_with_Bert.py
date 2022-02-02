# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 19:36:32 2022

NOTE: Ideas from https://www.tensorflow.org/text/tutorials/classify_text_with_bert

@author: Francisco Mena
"""

import pandas as pd
import numpy as np
import string
import nltk
import tensorflow as tf
from official.nlp import optimization

import os


#%%
print( tf.test.is_built_with_cuda() )
print( tf.config.list_physical_devices() )
print( tf.__version__)

tf.random.set_seed(42)


#%%

df = pd.read_csv('prueba.csv', delimiter = ';', names = ['Conversation_Id', 'Authors', 'Class', 'Conversation' ])

df.isna().sum()

df.dropna(inplace = True)

df.head()

#%%

df["Conversation"][0]
df["Class"].value_counts() 
df["Class"].value_counts(normalize = True) 
#problems with imbalanced dataset

#%%

df2 = df[["Conversation", "Class"]]

df2.dtypes
df2["Class"] = df2["Class"].astype(int)

target = df2.pop("Class")

#%% Cleaning up

df2 = df2.squeeze()

df_conv = df2.apply(lambda x: x.lower().replace('|', ' '))
print(df_conv)

from num2words import num2words
df_conv = df_conv.apply(lambda x: ' '.join([num2words(word) if word.isnumeric() and int(word)<1000000000 else word for message in x.split('|') for word in message.split(' ')]))
print(df_conv)

df_conv = df_conv.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
print(df_conv)


import json
with open('abbreviations.json', 'r') as fp:
    abbr_dict = json.load(fp)

df_conv = df_conv.apply(lambda x: ' '.join([abbr_dict[word] if word in abbr_dict.keys() else word for word in x.split(' ') ]))

df_conv.dtypes

#%%

X = df_conv.copy()
y = target.copy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle = True)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train, shuffle = True)

print(y_val.value_counts())
print(y_val.value_counts(normalize = True))

print(y_test.value_counts())
print(y_test.value_counts(normalize = True))


#%% undersampling cause of size of data


from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0)
X_resampled, y_resampled = rus.fit_resample(X_train.values.reshape(-1,1), y_train)



#%%
print(y_resampled.value_counts())
print(X_resampled.shape)


X_resampled = np.squeeze(X_resampled)


#size
#%%

import tensorflow_text as text
import tensorflow_hub as hub

## Tensorflow Hub model options

#bert_model_name = 'small_bert/bert_en_uncased_L-4_H-512_A-8'

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2'

print(f'BERT model selected           : {tfhub_handle_encoder}')
print(f'Preprocess model auto-selected: {tfhub_handle_preprocess}')

#%%
# We create a layer to preprocess the text within the model
bert_preprocess_model = hub.KerasLayer(tfhub_handle_preprocess)

#%%
#Let's test it
test_text = ['ask me 5 questions and i will answer them']
text_preprocessed = bert_preprocess_model(test_text)
print(f'Keys       : {list(text_preprocessed.keys())}')
print(f'Shape      : {text_preprocessed["input_word_ids"].shape}')
print(f'Word Ids   : {text_preprocessed["input_word_ids"][0, :12]}')
print(f'Input Mask : {text_preprocessed["input_mask"][0, :12]}')
print(f'Type Ids   : {text_preprocessed["input_type_ids"][0, :12]}')

#%% This will be the model

bert_model = hub.KerasLayer(tfhub_handle_encoder)

bert_results = bert_model(text_preprocessed)
print(f'Loaded BERT: {tfhub_handle_encoder}')
print(f'Pooled Outputs Shape:{bert_results["pooled_output"].shape}')
print(f'Pooled Outputs Values:{bert_results["pooled_output"][0, :12]}')
print(f'Sequence Outputs Shape:{bert_results["sequence_output"].shape}')
print(f'Sequence Outputs Values:{bert_results["sequence_output"][0, :12]}')


#%% Now that we have all these layers ready, we can build a classifier model


def build_classifier_model():

    #We first preprocess the text which creates the tokens, then encode, which will
    #be the numeric input to the classifier layer
    text_input = tf.keras.layers.Input(shape = (), dtype = tf.string, name = 'text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name = 'preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name = 'BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(1, activation = None, name = 'classifier')(net)
    
    return tf.keras.Model(text_input, net)


#%% Let's see if it works

classifier_model = build_classifier_model()
classifier_model.summary()

bert_raw_result = classifier_model(tf.constant(test_text)) #tf constant cause input has to be tensor
print(tf.sigmoid(bert_raw_result)) #pass it through a sigmoid 


loss = tf.keras.losses.BinaryCrossentropy(from_logits=True) #
metrics = tf.metrics.BinaryAccuracy()

# epochs = 5
epochs = 15
# steps_per_epoch = tf.data.experimental.cardinality(X_train).numpy() #entire dataset per epoch
# steps_per_epoch = len(X_train) #entire dataset per epoch
steps_per_epoch = len(X_resampled) #entire dataset per epoch
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps) #warmup is used for the learning rate

init_lr = 3e-5
optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')


#%% OPTINAL class weights
# negcount = y_train.value_counts()[0]
# poscount = y_train.value_counts()[1]

# # Scaling by total/2 helps keep the loss to a similar magnitude.
# # The sum of the weights of all examples stays the same.
# weight_for_0 = (1 / negcount) * (len(y_train) / 2.0)
# weight_for_1 = (1 / poscount) * (len(y_train) / 2.0)

# CLASSWEIGHTS = {0: weight_for_0, 1: weight_for_1}

#%% compile

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)


print(f'Training model with {tfhub_handle_encoder}')
# history = classifier_model.fit(df_conv, target,
#                                # validation_data=val_dataset,
#                                epochs=epochs)

BATCH = 32
#Let's train!
earlystop = tf.keras.callbacks.EarlyStopping(patience = 5, restore_best_weights= True)

history = classifier_model.fit(X_resampled, y_resampled,
                               validation_data= (X_val, y_val),
                               # class_weight = CLASSWEIGHTS,
                               callbacks = [earlystop],
                               batch_size = BATCH,
                               epochs=epochs)


#%% Visualize

pd.DataFrame( history.history)[["loss", "val_loss"]].plot()
pd.DataFrame( history.history)[["binary_accuracy", "val_binary_accuracy"]].plot()


#%% Evaluation

loss, accuracy = classifier_model.evaluate(X_test, y_test)

ypred = classifier_model.predict(X_test)

#%% Metrics

pd.Series(np.squeeze(tf.sigmoid(ypred))).describe()

ypred_values = np.squeeze(tf.sigmoid(ypred))

ypred_values = (ypred_values>0.5).astype(int)


from sklearn.metrics import classification_report, plot_confusion_matrix, confusion_matrix
from sklearn.dummy import DummyClassifier


# print( classification_report(y_test, pd.Series(ypred_values, index = y_test)) )

#%%
dumm = DummyClassifier(strategy="stratified")
dumm.fit(X_resampled, y_resampled)
ydum = dumm.predict(X_test)


print( classification_report(y_test,  ydum))
print( classification_report(y_test, pd.Series(ypred_values, index = y_test)) )

#%%

print( confusion_matrix(y_test,  ydum))
print( confusion_matrix(y_test, pd.Series(ypred_values, index = y_test)) )



#%% SAVE MODEL

classifier_model.save('saved_model/my_model')

"""
To load:
    
new_model = tf.keras.models.load_model('saved_model/my_model')

"""

#%% Alternative, save the weights

classifier_model.save_weights('saved_model/my_checkpoint')






