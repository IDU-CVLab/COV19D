#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:28:03 2021

@author: idu
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import backend as K
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from keras.callbacks import Callback, LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.optimizers import RMSprop, SGD, Adam
from keras import layers, models
from tensorflow.keras import activations

####### Generatiiing Data
batch_size = 64
SIZE = 512

train_datagen = ImageDataGenerator(rescale=1./255)
                                   
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validation/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

y_train = train_generator.classes
y_val = val_generator.classes

################ A CNN Model for image classification
def make_model():
   
    model = models.Sequential()
    
    model.add(layers.Conv2D(16,(3,3),input_shape=(SIZE,SIZE,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
      
    model.add(layers.Dense(1,activation='sigmoid'))
    
    
    return model


model = make_model()

###################################### Training the model
n_epochs= 70

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

history=model.fit(train_generator, 
                  steps_per_epoch=250,
                  validation_data=val_generator,
                  validation_steps=78,
                  verbose=2,
                  epochs=n_epochs,)
                 # callbacks=[lr_rate_scheduler])

########################33 Evaluating
print (history.history.keys())
            
Train_accuracy = history.history['accuracy']
print(Train_accuracy)
print(np.mean(Train_accuracy))
val_accuracy = history.history['val_accuracy']
print(val_accuracy)
print( np.mean(val_accuracy))
print( np.max(val_accuracy))

epochs = range(1, len(Train_accuracy)+1)

plt.figure(figsize=(12,6))
plt.plot(epochs, Train_accuracy, 'g', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.ylim(0.45,1)
plt.xlim(0,50)
plt.legend()

plt.show()

val_recall = history.history['val_recall']
print(val_recall)
avg_recall = np.mean(val_recall)
avg_recall

val_precision = history.history['val_precision']
avg_precision = np.mean(val_precision)
avg_precision

epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, val_recall, 'g', label='Validation Recall')
plt.plot(epochs, val_precision, 'b', label='Validation Prcision')
plt.title('Validation recall and Validation Percision')
plt.xlabel('Epochs')
plt.ylabel('Recall and Precision')
plt.legend()
plt.ylim(0,1)

plt.show()

Macro_F1score = (2*avg_precision*avg_recall)/ (avg_precision + avg_recall)
Macro_F1score


