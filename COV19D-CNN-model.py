#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 22:19:35 2021

@author: idu
"""
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import keras
from keras.models import load_model
from keras import backend as K

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model,models
from tensorflow.keras.layers import LeakyReLU
import tensorflow_addons as tfa


from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

batch_size = 32
SIZE = 128

train_datagen = ImageDataGenerator(rescale=1./255)
                                   
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train/', 
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validation/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        classes = ['covid','non-covid'],
        class_mode='binary')

y_train = train_generator.classes
y_val = val_generator.classes 

def make_model():
    '''
    Define your model architecture here.
    Returns `Sequential model`
    '''
    
    model = models.Sequential()
    
    model.add(layers.Conv2D(16,(3,3),input_shape=(SIZE,SIZE,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))   
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.MaxPooling2D((2,2)))
    
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Dropout(0.2))
      
    model.add(layers.Dense(1,activation='sigmoid'))
    
    
    return model


model = make_model()
model.summary()

total_sample=train_generator.n
n_epochs = 20
steps_per_epoch_train = len(train_generator) // batch_size
steps_per_epoch_val = len(val_generator) // batch_size

INIT_LR = 0.0005
def lr_scheduler(epoch):
        return INIT_LR ** epoch
    
model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(learning_rate=0.0001),
             metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])
#train_generator.reset()
#val_generator.reset()
# Training the model
history=model.fit(train_generator, 
                            steps_per_epoch=200,
                            validation_data=val_generator,
                            validation_steps=75,
                            verbose=2,
                            epochs=n_epochs,
                            callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler)])


print (history.history.keys())

Train_accuracy = history.history['accuracy']
print(Train_accuracy)
print(np.mean(Train_accuracy))
val_accuracy = history.history['val_accuracy']
print(val_accuracy)
print( np.mean(val_accuracy))

epochs = range(1, len(Train_accuracy)+1)

plt.figure(figsize=(12,6))
plt.plot(epochs, Train_accuracy, 'g', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Validation accuracy')
plt.legend()

plt.show()

epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, Train_accuracy, 'g', label='Training loss')
plt.plot(epochs, Train_accuracy, 'b', label='Validation loss')
plt.title('Training and validation binary loss')
plt.xlabel('Epochs')
plt.ylabel('Training Accuracy')
plt.legend()

plt.show()


val_recall = history.history['val_recall']
avg_recall = np.mean(val_recall)
avg_recall

val_precision = history.history['val_precision']
avg_precision = np.mean(val_precision)
avg_precision

Macro_F1score = (2*avg_precision*avg_recall)/ (avg_precision + avg_recall)
Macro_F1score
