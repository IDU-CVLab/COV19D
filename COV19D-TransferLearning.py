#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 28 15:50:19 2021

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
        '/home/idu/Desktop/COV19D/train/', 
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode='rgb',
        class_mode='binary')


print('****************')
for cls, idx in train_generator.class_indices.items():
    print('Class #{} = {}'.format(idx, cls))
print('****************')

for _ in range(5):
    img, label = next(train_generator)
    print(img.shape)
    #print(label[0])
    #print(train_generator.classes[0])
    plt.imshow(img[0])
    plt.show()
    

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validation/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        color_mode = 'rgb',
        class_mode='binary')

y_train = train_generator.classes
y_val = val_generator.classes 

y_train
y_val


###################################### Transfer Learning Models####################################333

ResNet_model = ResNet50(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
VGG_model = VGG16(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
Mobile_Net = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))
Model_Xcep = tf.keras.applications.Xception(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))

for layer in ResNet_model.layers:
	layer.trainable = False
ResNet_model.summary()

for layer in VGG_model.layers:
	layer.trainable = False
VGG_model.summary()  

for layer in Mobile_Net.layers:
	layer.trainable = False
Mobile_Net.summary() 

for layer in Model_Xcep.layers:
	layer.trainable = False
Model_Xcep.summary()

feature_extractor=ResNet_model.predict(train_generator)
feature_extractor=VGG_model.predict(train_generator)
feature_extractor=Mobile_Net.predict(train_generator)
feature_extractor=Model_Xcep.predict(train_generator)

features = feature_extractor.reshape(feature_extractor.shape[0], -1)

########3 The Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
RF_model.fit(features, y_train)

########################################################### CHecking accuracy for Transfer learning##########33

X_val_feature = ResNet_model.predict(val_generator)
X_val_feature = Mobile_Net.predict(val_generator)
X_val_feature = Model_Xcep.predict(val_generator)
X_val_features = X_val_feature.reshape(X_val_feature.shape[0], -1)

#Now predict using the trained RF model. 
prediction_RF =RF_model.predict(X_val_features)


from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_val, prediction_RF))
print ("F1_score (COVID) = ", metrics.f1_score(y_val, prediction_RF, pos_label = 0))
print ("F1_score (NON-COVID) = ", metrics.f1_score(y_val, prediction_RF,pos_label = 1))
print ("Macro F1_score = ", metrics.f1_score(y_val, prediction_RF, average='macro'))

