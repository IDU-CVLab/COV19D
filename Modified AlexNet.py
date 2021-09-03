#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 19:52:07 2021

@author: idu
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow import keras
from keras.models import load_model
from keras import backend as K
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50
from PIL import Image

#######################################Generating data with rescaling and binary labels from the images
batch_size = 32
SIZE = 128


train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/trainn/', 
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode='rgb',
        class_mode='binary')


print('****************')
for cls, idx in train_generator.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')


val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validationn/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode = 'rgb',
        class_mode='binary')

y_train = train_generator.classes
y_val = val_generator.classes 

y_train
y_val


#########################3

steps_per_epoch = train_generator.samples//batch_size

in_h = in_w = SIZE

###### Modified AlexNet Model for Feature Exctraction

Mod_AlexNet = tf.keras.models.Sequential([
    # 1st conv
  tf.keras.layers.Conv2D(10, (11,11),strides=(4,4), activation='relu', input_shape=(in_h, in_w, 3)),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2,2)),
    # 2nd conv
  tf.keras.layers.Conv2D(18, (11,11),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
     # 3rd conv
  tf.keras.layers.Conv2D(18, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 4th conv
  tf.keras.layers.Conv2D(18, (3,3),strides=(1,1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
    # 5th Conv
  tf.keras.layers.Conv2D(12, (3, 3), strides=(1, 1), activation='relu',padding="same"),
  tf.keras.layers.BatchNormalization(),
  tf.keras.layers.MaxPooling2D(2, strides=(2, 2)),
  # To Flatten layer
  tf.keras.layers.Flatten(),
  # To FC layer 1
  tf.keras.layers.Dense(512, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  #To FC layer 2
  tf.keras.layers.Dense(10, activation='relu'),
    # add dropout 0.5 ==> tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(2)
])
    
    
Mod_AlexNet.compile(optimizer='adadelta', loss="binary_crossentropy", metrics=['accuracy'])

Mod_AlexNet.summary()

Mod_AlexNet.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=15)

Mod_AlexNet.evaluate(val_generator)

feature_extractor=Mod_AlexNet.predict(train_generator)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

feature_extractor.shape

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
RF_model.fit(features, y_train)

RF_model.fit(features, y_train)

X_val_feature = Mod_AlexNet.predict(val_generator)
X_val_features = X_val_feature.reshape(X_val_feature.shape[0], -1)

prediction_RF = RF_model.predict(X_val_features)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_val, prediction_RF))
print ("F1_score (COVID) = ", metrics.f1_score(y_val, prediction_RF, pos_label = 0))
print ("F1_score (NON-COVID) = ", metrics.f1_score(y_val, prediction_RF,pos_label = 1))
print ("Macro F1_score = ", metrics.f1_score(y_val, prediction_RF, average='macro'))

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, prediction_RF)
cm
sns.heatmap(cm, annot=True)

