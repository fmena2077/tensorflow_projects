

"""
Script using a simple CNN to clasiffy color images
Author : Francisco Mena
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers

tf.test.is_built_with_cuda()
tf.config.list_physical_devices()

tf.random.set_seed(42)


#%% Obtener data

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train.astype("float32")/255.
X_test =  X_test.astype("float32")/255.


num_classes = len(np.unique(y_train))

#%% Visualization
example = X_train[0]

import matplotlib.pyplot as plt
plt.imshow(example)


#%% Let's build a convolutional AlexNet

model = keras.models.Sequential([
    keras.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = 'elu', 
                        kernel_regularizer = regularizers.l2(1e-4), input_shape = X_train.shape[1:]),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.2),
        
    keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'elu', 
                        kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.3),
    
    keras.layers.Conv2D(filters = 128, kernel_size = 3, padding = 'same', activation = 'elu', 
                        kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size = (2,2)),
    keras.layers.Dropout(0.4),
    
    keras.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = 'elu', 
                        kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.BatchNormalization(),
    
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation = 'elu', kernel_initializer="lecun_normal",
                       kernel_regularizer = regularizers.l2(1e-4)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation = 'softmax')
    ])

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "adam", metrics = "accuracy")


#%%
# earlystop = tf.keras.callbacks.EarlyStopping(patience = 30, restore_best_weights = True)

num_batch = 64
num_epochs = 100

history = model.fit(X_train, y_train, 
                    validation_split = 0.2,
                    batch_size = num_batch,
                    epochs = num_epochs,
                    verbose = 2
                    )


print(model.evaluate(X_test, y_test))

#%% Plot evolution

history.history.keys()
plt.figure()
pd.DataFrame(history.history).plot()
plt.show()

#%% Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

ypred_proba = model.predict(X_test)
ypred = np.argmax(ypred_proba, axis = 1)

# print( classification_report(y_test, ypred))
# print( confusion_matrix(y_test, ypred))
print( accuracy_score(y_test, ypred))


#%%
