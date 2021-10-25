# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 19:36:20 2021

Load different models and see how they classify an image

https://www.tensorflow.org/hub/tutorials/image_classification

@author: franc
"""

import tensorflow as tf
import tensorflow_hub as hub

import requests
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np

#%% Choose an image

image_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Bengal_tiger_%28Panthera_tigris_tigris%29_female_3_crop.jpg"

def load_image(image_url, target_height, target_width):
    
    """Returns an image with shape [1, height, width, num_channels]."""
    user_agent = {'User-agent': 'Cob Sample (https://tensorflow.org)'}
    response = requests.get(image_url, headers=user_agent)
    image = Image.open(BytesIO(response.content))   #BytesIO manages memory better than load
    
    #Process image size and scale
    image = np.array(image) #image to array
    img_reshape = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]]) #Add one dimension to account for the batch
    image_norm = tf.image.convert_image_dtype(img_reshape, tf.float32)  #This will normalize between 0 and 1

    #Resize to the model image size
    image_ready = tf.image.resize_with_pad(image_norm, target_height, target_width)
    
    return image_ready

#%% Choose model
# https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2

model_name = "efficientnetv2-s"
model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
model_image_size = 384

#download labels of ImageNet and creates a maps
labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)

classes = []
i = 0
with open(downloaded_file) as f:
  labels = f.readlines()
  classes = [l.strip() for l in labels[1:]]
  i += 1

##Load the chosen model and run

classifier = hub.load(model_url)

image = load_image(image_url, model_image_size, model_image_size)

input_shape = image.shape

#Let's test the model with a random tensor of same shape
warmup_input = tf.random.uniform(input_shape, 0, 1.0)
%time warmup_logits = classifier(warmup_input).numpy()
warmup_logits.shape

#Prediction
predict_logits = classifier(image)
predict_logits = tf.nn.softmax(predict_logits).numpy()   # normalize to 0-1 range, and convert to numpy
predicted_label = np.argmax(predict_logits)

print(classes[predicted_label] ) #Correctly predicts a tiger!

#Let's choose the 5 predictions with higher probs. TF has a convenient func for sorting
classes_predicted = tf.argsort(predict_logits, direction="DESCENDING")[0][:5].numpy()

for idx in classes_predicted:
    print('Index ' + str(idx) + ' - Label: ' + classes[idx]  + ' - Probability: ' + str(predict_logits[0,idx]))


#%% Let's try two more models

model_name = "resnet_v1_50"
model_url = "https://tfhub.dev/google/imagenet/resnet_v1_50/classification/5"
model_image_size = 224


classifier = hub.load(model_url)

# image = load_image(image_url, model_image_size, model_image_size)

# input_shape = image.shape


#Prediction
%time predict_logits = classifier(image)
predict_logits = tf.nn.softmax(predict_logits).numpy()   # normalize to 0-1 range, and convert to numpy
predicted_label = np.argmax(predict_logits)

print(classes[predicted_label] ) #Correctly predicts a tiger!
print('\n')

#Let's choose the 5 predictions with higher probs. TF has a convenient func for sorting
classes_predicted = tf.argsort(predict_logits, direction="DESCENDING")[0][:5].numpy()

for idx in classes_predicted:
    print('Index ' + str(idx) + ' - Label: ' + classes[idx]  + ' - Probability: ' + str(predict_logits[0,idx]))

# Interestingly, ResNet predicts a cheetah instead of a tiger

#%% #Inception

model_name = "inception_v3"
model_url = "https://tfhub.dev/google/imagenet/inception_v3/classification/5"
model_image_size = 299


classifier = hub.load(model_url)

# image = load_image(image_url, model_image_size, model_image_size)

# input_shape = image.shape


#Prediction
%time predict_logits = classifier(image)
predict_logits = tf.nn.softmax(predict_logits).numpy()   # normalize to 0-1 range, and convert to numpy
predicted_label = np.argmax(predict_logits)

print(classes[predicted_label] ) #Correctly predicts a tiger!
print('\n')

#Let's choose the 5 predictions with higher probs. TF has a convenient func for sorting
classes_predicted = tf.argsort(predict_logits, direction="DESCENDING")[0][:5].numpy()

for idx in classes_predicted:
    print('Index ' + str(idx) + ' - Label: ' + classes[idx]  + ' - Probability: ' + str(predict_logits[0,idx]))


#%%