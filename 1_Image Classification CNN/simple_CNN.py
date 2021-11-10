"""
Script using a simple CNN to clasiffy black&white images

Author: Francisco Mena
"""


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D


tf.test.is_built_with_cuda()
tf.config.list_physical_devices()

#%% Obtener data

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]
X_train.shape

Xmean = X_train.mean(axis=0, keepdims=True)
Xstd = X_train.std(axis=0, keepdims=True)

X_train = (X_train - Xmean) / Xstd
X_valid = (X_valid - Xmean) / Xstd
X_test = (X_test - Xmean) / Xstd

# Add a dimension to include the # of channels (here 1, black&white images)
X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]

#%% Let's build a convolutional AlexNet

model = keras.models.Sequential([
    keras.layers.Conv2D(filters = 64, kernel_size = 5, padding = 'same', activation = 'relu', input_shape =X_train.shape[1:]),
    keras.layers.MaxPool2D(pool_size = 2),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Conv2D(filters= 256, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv2D(filters= 256, kernel_size=3, padding='same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation= 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation= 'relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation= 'softmax')
    ])

model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = "nadam", metrics = "accuracy")


#%%
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid), verbose = 2)

print(model.evaluate(X_test, y_test))

#%% Evaluation
from sklearn.metrics import classification_report, confusion_matrix

ypred_proba = model.predict(X_test)
ypred = np.argmax(ypred_proba, axis = 1)

print( classification_report(y_test, ypred))
print( confusion_matrix(y_test, ypred))


#%%
