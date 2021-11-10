
"""
Use TF datasets to load data, use tf.iamges to augment, and CCN with regularization  
Author : Francisco Mena
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.data import AUTOTUNE
import tensorflow_datasets as tfds

tf.test.is_built_with_cuda()
tf.config.list_physical_devices()

tf.random.set_seed(42)
keras.backend.clear_session()

#%%

# train_ds, test_ds = tfds.load('cifar10', split=['train','test'], 
#                                as_supervised = True, 
#                                batch_size = -1)
#%% Obtener data


# tfds.disable_progress_bar()
(train_set, test_set), ds_info = tfds.load('cifar10', 
                              split=['train', 'test'], 
                               shuffle_files=True,
                              as_supervised = True,     #gives pairs of data, label. Otherwise returns dictionary
                              with_info = True,          # if False it only returns train,test
#                              batch_size=32
                              )
print(ds_info)

#%%
#fig = tfds.show_examples(train_set, ds_info)

#%% Visualization
# example, label = train_ds.take(2)

# import matplotlib.pyplot as plt
# plt.imshow(example)

#%% Create a preprocessing Pipeline with Tensorflow

#First, the functions
def resize_and_rescale(image, label):
    image = tf.cast(image, tf.float32)  #convert to float32 to save memory
    image = (image / 255.)   #scale

    return image, label


def augment(image, label):
    
    #Note: you can replace for stateless_random_brightness to set seed
    image, label = resize_and_rescale(image, label)
    image = tf.image.random_brightness(image, max_delta=0.3) #random brightness
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_flip_up_down(image)   #Flipping upside down
    # image = tf.image.rot90(image)
    image = tf.clip_by_value(image, clip_value_min = 0, clip_value_max = 1)
    return image, label
    

#%%
#%%
# Preprocessing pipeline
BATCH_SIZE = 32
# train_set = train_set.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
train_set = train_set.map(augment, num_parallel_calls=AUTOTUNE)
train_set = train_set.shuffle(1000)
train_set = train_set.batch(BATCH_SIZE)
train_set = train_set.prefetch(AUTOTUNE)



test_set = test_set.map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
test_set = test_set.batch(BATCH_SIZE)
test_set = test_set.prefetch(AUTOTUNE)

#%% Data Augmentation


# from tensorflow.keras.preprocessing.image import ImageDataGenerator

# datagen = ImageDataGenerator(rotation_range= 20, #rotate the image
#                    width_shift_range= 0.1,      # shift width
#                    height_shift_range= 0.1,     # shift height
#                    horizontal_flip=True,        # flip horizontally
#                    validation_split=0.2
#                    )        

# datagen.fit(X_train)

#%% Let's build a convolutional AlexNet

keras.backend.clear_session()

model = keras.models.Sequential([
    keras.layers.InputLayer((32,32,3)),
    keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'relu',
                        kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
        
    keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu',
                        kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'relu',
                        kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.4),
    
    keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'relu',
                        kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation = 'relu', kernel_initializer="lecun_normal",
                       kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation = 'softmax')
    ])

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam", metrics = "accuracy")


#%%
earlystop = tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)


history = model.fit(train_set, 
                    validation_data = test_set,
                    epochs = 30,
                    # verbose = 2
                    )


#%%
model.evaluate(test_set)

### Without augmentation and 10 epochs: 0.75

### With augmentation and 30 epochs: 0.79


#%% Plot evolution
import matplotlib.pyplot as plt

history.history.keys()
plt.figure()
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.show()

history.history.keys()
plt.figure()
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']]

#%% Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ypred_proba = model.predict(test_set)
ypred = np.argmax(ypred_proba, axis = 1)

# print( classification_report(y_test, ypred))
# print( confusion_matrix(y_test, ypred))


#%%
