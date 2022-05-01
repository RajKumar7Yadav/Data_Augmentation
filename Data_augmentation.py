# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:46:25 2022

@author: raj.yadav
"""

import os
import cv2
import numpy as np
from skimage import io
from keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',cval=125) #cval is not given then image shifted part will be black

# #                       Augumentation for a single class
 
# for augmenting single image
image=io.imread('C/exp1.jpg')
image=image.reshape((1,)+image.shape)


i=0
for batch in datagen.flow(image,batch_size=32,save_to_dir='data_aug_example',save_format='jpg',save_prefix='aug'):
    i=i+1
    if i>10:
        break

# for agumenting multiple images in a folder
import glob
images=[]
files=glob.glob('/data_aug_example/*.jpg')    
for filename in files:
    img=cv2.imread(filename)
    img=cv2.resize(img, (244,244))
    images.append(img)
img_array=np.array(images,dtype='float32')    

i=0
for batch in datagen.flow(img_array,batch_size=6,save_to_dir='data_aug',save_format='jpg',save_prefix='aug'):
    i=i+1
    if i>10:
        break


#                       Augumentation for a Multi class

i=0
for batch in datagen.flow_from_directory(directory='data_aug_example',
                                         batch_size=6,
                                         save_to_dir='augmentated',save_format='jpg',save_prefix='aug'):
    i=i+1
    if i>10:
        break
