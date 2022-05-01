# -*- coding: utf-8 -*-
"""
Created on Sun May  1 15:42:01 2022

@author: raj.yadav
"""
import os
import cv2
import numpy as np

# Copying model neural network from Binary_classification.py file
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout

INPUT_SHAPE=(244,244,3)

model=Sequential()
#what is input and input layer
# 1st convlutional layer
model.add(Conv2D(32,(3,3),activation=('relu'),input_shape=(244,244,3)))
model.add(MaxPool2D(pool_size=(2,2)))
# 2nd convolutional layer
model.add(Conv2D(64,(3,3),activation=('relu')))
model.add(MaxPool2D(pool_size=(2,2)))
# 3rd convolutional layer
model.add(Conv2D(128,(3,3),activation=('relu')))
model.add(MaxPool2D(2,2))

model.add(Flatten())
model.add(Dense(512,activation=('relu')))
model.add(Dense(1,activation=('sigmoid')))

#model.summary()

from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(
    horizontal_flip=True)


test_datagen=ImageDataGenerator(rescale=1/255.)
"""
can store the data like we did in Data_augumentation.py file. Then we have to store the augmented data & then 
we will use the data from augmented folder.But it'll consume more memory, so instead we are just storing the
data in train_genrator & test_generator and resuing it. If wanted to save the use 'save_to_dir' param 
train_datagen.flow_from_directory(save_to_dir='folder_name') and then will have to read again from that folder
store it in train(train.append(img)) list and then pass it to model.fit()
"""
train_genrator=train_datagen.flow_from_directory(directory='train_new',
                                                 batch_size=8,
                                                 target_size=(244,244),
                                                 class_mode='binary')

test_genrator=test_datagen.flow_from_directory(directory='test_new',
                                                 batch_size=8,
                                                 target_size=(244,244),
                                                 class_mode='binary')


model.compile(loss=('binary_crossentropy'),optimizer='rmsprop',metrics=('accuracy'))

#adding a checkpoint to save best model
import os
from keras.callbacks import ModelCheckpoint

MODEL_PATH='C:/Users/raj.yadav/Projects/Self_try/Binary_classifier/Models'
filepath=os.path.join(MODEL_PATH,"Cellphone_data_aug.h5")
checkpoint=ModelCheckpoint(filepath,monitor='val_accuracy',verbose=1,save_best_only=True,mode='max',period=1)
callback_list=[checkpoint]

model_train=model.fit(train_genrator,batch_size=32,epochs=5,validation_data=test_genrator,callbacks=callback_list)
