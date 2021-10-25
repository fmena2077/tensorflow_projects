# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 17:27:08 2021
https://towardsdatascience.com/realtime-multiple-person-2d-pose-estimation-using-tensorflow2-x-93e4c156d45f
https://github.com/Mjrovai/TF2_Pose_Estimation/blob/master/10_Pose_Estimation_Images.ipynb
https://www.youtube.com/watch?v=Uui-7bag7Pk

Pose Estimation Tutorial

@author: franc
"""

# Importing necessary libraries
import os
os.chdir('C:\\Users\\franc\\projects\\letstf2gpu\\1b_ Object Classification\\Pose estimation\\tf-pose-estimation')

import sys
import time
import logging
import numpy as np
import matplotlib
# import tkinter
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2


#%% Load an image 

image_path = './images/apink1.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.grid()
plt.show()

image.shape

#%%  Choosing model

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

model='mobilenet_thin'
resize='432x368'
w, h = model_wh(resize)


#%%  Get Estimator e

e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))


#%% Let's resize the image

humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)

"""
e.heatMat contiene la probabilidad de cada pixel de pertenecer a uno de los 18 joints del cuerpo,
o ninguno de ellos (elemento 19).
Por ejemplo, e.heatMat[0][0] tiene valor 0.99 en el último elemento (None), indicando que 
este pixel no pertenece a uno de los 18 joints.

"""
#%%
# Key points found by the model
max_prob = np.amax(e.heatMat[:, :, :-1], axis=2)
plt.imshow(max_prob)
plt.grid();

#%%

# Key points over original image
plt.figure(figsize=(15,8))
bgimg = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)
bgimg = cv2.resize(bgimg, (e.heatMat.shape[1], e.heatMat.shape[0]), interpolation=cv2.INTER_AREA)
plt.imshow(bgimg, alpha=0.5)
plt.imshow(max_prob, alpha=0.5)
plt.colorbar()
plt.grid();


#%%
#Finally, let's draw the skeleton

image_og = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
plt.figure(figsize=(15,8))
plt.imshow(image_og)


#%%
# Let's obtain only the skeleton

# image = common.read_imgfile(image_path, None, None)
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
black_background = np.zeros(image.shape)
skeleton = TfPoseEstimator.draw_humans(black_background, humans, imgcopy=False)
plt.figure(figsize=(15,8))
plt.imshow(skeleton);
plt.grid(); 
# plt.axis('off');

#%%
#el objeto "humans" posee las coordenadas de los joints. Para obtenerlas:

keypoints = str(str(str(humans[0]).split('BodyPart:')[1:]).split('-')).split(' score=')
len(keypoints) #18 joints

keypoints_list=[]
for i in range (len(keypoints)-1): 
    pnt = keypoints[i][-11:-1]
    pnt = tuple(map(float, pnt.split(', ')))
    keypoints_list.append(pnt)

keypoints_list

#%% 
#Transformamos a coordenadas de la imagen
keypts_array = np.array(keypoints_list)
keypts_array = keypts_array*(image.shape[1],image.shape[0])
keypts_array = keypts_array.astype(int)
keypts_array

plt.figure(figsize=(10,10))
plt.axis([0, image.shape[1], 0, image.shape[0]])  
plt.scatter(*zip(*keypts_array))
ax=plt.gca() 
ax.set_ylim(ax.get_ylim()[::-1]) 
ax.xaxis.tick_top() 
plt.grid();


#%%
#Pongamoslo todo junto
plt.figure(figsize=(10,10))
plt.axis([0, image.shape[1], 0, image.shape[0]])  
plt.scatter(*zip(*keypts_array), s=200, color='orange', alpha=0.6)
img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img)
ax=plt.gca() 
ax.set_ylim(ax.get_ylim()[::-1]) 
ax.xaxis.tick_top() 
plt.grid();

for i, txt in enumerate(keypts_array):
    ax.annotate(i, (keypts_array[i][0]-5, keypts_array[i][1]+5))



#%%
#Definamos funciones para facilitar el análisis

def get_human_pose(image_path, showBG = True):
    image = common.read_imgfile(image_path, None, None)
  
    if image is None:
        sys.exit('Image can not be read, path=%s' % image)

    t = time.time()
    
    humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
    elapsed = time.time() - t

    #logger.info('inference image: %s in %.4f seconds.' % (image, elapsed))
    if showBG == False:
        image = np.zeros(image.shape)
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    return image, humans


def show_keypoints(image, hum, human=1, color='orange', showBG = True):
    if human == 0: human = 1
    num_hum = len(hum)
    keypoints = str(str(str(hum[human-1]).split('BodyPart:')[1:]).split('-')).split(' score=')
    keypoints_list=[]
    for i in range (len(keypoints)-1): 
        pnt = keypoints[i][-11:-1]
        pnt = tuple(map(float, pnt.split(', ')))
        keypoints_list.append(pnt)

    keypts_array = np.array(keypoints_list)
    keypts_array = keypts_array*(image.shape[1],image.shape[0])
    keypts_array = keypts_array.astype(int)
    keypts_array

    plt.figure(figsize=(10,10))
    plt.axis([0, image.shape[1], 0, image.shape[0]])  
    plt.scatter(*zip(*keypts_array), s=200, color=color, alpha=0.6)
    if showBG:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      
    plt.imshow(image)
    ax=plt.gca() 
    ax.set_ylim(ax.get_ylim()[::-1]) 
    ax.xaxis.tick_top() 
    plt.title('Keypoints Person [{}] from {} humans detected\n'.format(human, num_hum))
    plt.grid();

    for i, txt in enumerate(keypts_array):
        ax.annotate(i, (keypts_array[i][0]-5, keypts_array[i][1]+5))
            
    return keypts_array


#%%

path1 = os.path.join( os.getcwd(), 'images\\apink2.jpg')

img, hum = get_human_pose(path1)
show_keypoints(img, hum, 1)


#%%

path1 = os.path.join( os.getcwd(), 'images\\apink3.jpg')

img, hum = get_human_pose(path1)
show_keypoints(img, hum, 1)

#%%
    

path1 = os.path.join( os.getcwd(), 'images\\ski.jpg')

img, hum = get_human_pose(path1)
show_keypoints(img, hum, 1)



