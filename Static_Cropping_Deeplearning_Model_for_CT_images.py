#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 01:34:25 2021

@author: idu
"""

### Importing libraries
import skimage
from skimage import color, filters
import numpy as np
import os, glob
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers, models
from tensorflow.keras import activations
from tensorflow.keras.models import Sequential
from termcolor import colored  

#import visualkeras
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
#from collections import defaultdict

#from PIL import ImageFont

from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img

import pandas as pd

import csv
from sklearn.utils import class_weight 
from collections import Counter

#from tensorflow.keras.models import model_from_json
from keras.callbacks import ModelCheckpoint


#########
####### Generatiiing Data

#batch_size = 32
batch_size = 128
#batch_size = 128
SIZE = 512

train_datagen = ImageDataGenerator(rescale=1./255,)
                              #height_shift_range=0.2,
                              #rotation_range=5, 
                              #shear_range = 0.02,
                              #fill_mode='nearest',
                              #vertical_flip=True,
                              #horizontal_flip=True) 
                                                                
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train/',  ## COV19-CT-DB Training set
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validation/',  ## COV19-CT-DB Validation set
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

################ CNN Model Architecture
def make_model():
   
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(SIZE,SIZE,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    return model

model = make_model()

###################################### Compiling and Training the model
n_epochs= 70

model.compile(loss='binary_crossentropy',
              optimizer = 'Adam',
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

#early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5)

history=model.fit(train_generator,
                  steps_per_epoch=250,
                  validation_data=val_generator,
                  validation_steps=78,
                  verbose=2,
                  epochs=n_epochs,)
                  #class_weight=class_weights,
                  #callbacks=[early_stopping_cb])


###################### THE METHOD WITH SLICE PROCESSING AND HYPERPARAMETERS TUNNING ####################################
#####################3###############################################################3#############################33###
    #########################################################################################################3
         ############################################################################################

########## Cropping slices and removing none-representative slices in the CT volume
            
t = 0.45 #Histogram Threshold
#### Cropping lungs as ROI and removing upper and lowermost of the slices 
count = []
folder_path = '/home/idu/Desktop/COV19D/validation/covid' 
#Change this directory to the directory where you need to do preprocessing for images
#Inside the directory must folder(s), which have the images inside them
for fldr in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, fldr)
        for filee in os.listdir(sub_folder_path):
            file_path = os.path.join(sub_folder_path, filee)
            img = cv2.imread(file_path)
            #Grayscale images
            img = skimage.color.rgb2gray(img) 
            # First cropping an image
            #%r = cv2.selectROI(img) 
            #Select ROI from images before you start the code 
            #Reference: https://learnopencv.com/how-to-select-a-bounding-box-roi-in-opencv-cpp-python/
            #{Last access 15th of Dec, 2021}
            # Crop image using r
            img_cropped = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
            # Thresholding and binarizing images
            # Reference: https://datacarpentry.org/image-processing/07-thresholding/
            #{Last access 15th of Dec, 2021}
            # Gussian Filtering
            img = skimage.filters.gaussian(img_cropped, sigma=1.0)
            # Binarizing the image
            img = img < t
            count = np.count_nonzero(img)  ### COunting number of bright pixels in the binarized slices
            #print(count)
            if count > 3400:
             img_cropped = np.expand_dims(img_cropped, axis=2)
             img_cropped = array_to_img (img_cropped)
               # Replace images with the image that includes ROI
             img_cropped.save(str(file_path), 'JPEG')
             #print('saved')
            else:
                # Remove non-representative slices
             os.remove(str(file_path))
             #print('removed')
             # Check that there is at least one slice left
            if not os.listdir(str(sub_folder_path)):
              print(str(sub_folder_path), "Directory is empty")
            count = []

#################### Importing Cropped Images from Directory
h=227 # Height of the cropped images
w=300  # Width of the cropped images
batch_size=128
train_datagen = ImageDataGenerator(rescale=1./255, 
                              vertical_flip=True,
                              horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train/',  ## COV19-CT-DB Training set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/validation/',  ## COV19-CT-DB (Test) Validation set
        target_size=(h, w),
        batch_size=batch_size,
        color_mode='grayscale',
        classes = ['covid','non-covid'],
        class_mode='binary')

#### The CNN model

def make_model():
   
    model = models.Sequential()
    
    # Convulotional Layer 1
    model.add(layers.Conv2D(16,(3,3),input_shape=(h,w,1), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 2
    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 3
    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())   
    model.add(layers.MaxPooling2D((2,2)))
    
    # Convulotional Layer 4
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2,2)))
    
    # Fully Connected Layer
    model.add(layers.Flatten())
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.1))
    
    # Dense Layer  
    model.add(layers.Dense(1, activation='sigmoid'))
    
    
    return model

model = make_model()

n_epochs= 70

# Compiling the model using SGD optimizer with a learning rate schedualer
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule),
              metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=7)

###Learning Rate decay
def decayed_learning_rate(step):
  return initial_learning_rate * decay_rate ^ (step / decay_steps)

initial_learning_rate = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=100000,
    decay_rate=0.96,
    staircase=True)

#filepath1="/home/idu/Desktop/COV19D/saved-models/CNN-model-with-modified-images.h5"
#filepath2="/home/idu/Desktop/COV19D/saved-models/CNN-model-with-modified-images.jason"
#model_json = model.to_json()
#with open("model.json", "w") as json_file:
    #json_file.write(model_json)
#model.save_weights("/home/idu/Desktop/COV19D/saved-models/CNN-model-with-modified-images.h5")
#checkpoint = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

## Class weight
counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
class_weights

training_steps = 280462 // batch_size
val_steps = 63559 // batch_size

history=model.fit(train_generator,
                  steps_per_epoch=training_steps,
                  validation_data=val_generator,
                  validation_steps=val_steps,
                  verbose=2,
                  epochs=n_epochs,
                  callbacks=[early_stopping_cb, checkpoint],
                  class_weight=class_weights)
                  
# Saving the model
model.save('/home/idu/Desktop/COV19D/saved-models/CNN-model-with-modified-images.h5')

model = keras.models.load_model('/home/idu/Desktop/COV19D/saved-models/CNN-model-with-modified-images.h5')

############################## Making Predictions at patient level 

## Choosing the directory where the test/validation data is at
folder_path = '/home/idu/Desktop/COV19D/validation/non-covid/'
extensions0 = []
extensions1 = []
extensions2 = []
extensions3 = []
extensions4 = []
extensions5 = []
extensions6 = []
extensions7 = []
extensions8 = []
extensions9 = []
extensions10 = []
extensions11 = []
extensions12 = []
extensions13 = []
covidd = []
noncovidd = []
coviddd = []
noncoviddd = []
covidddd = []
noncovidddd = []
coviddddd = []
noncoviddddd = []
covidd6 = []
noncovidd6 = []
covidd7 = []
noncovidd7 = []
covidd8 = []
noncovidd8 = []
results =1
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)
        c=load_img(file_path, color_mode='grayscale', target_size=(227,300))
        c=img_to_array(c)
        c= np.expand_dims(c, axis=0)
        c /= 255.0
        result = model.predict_proba(c) #Probability of 1 (non-covid)
        if result > 0.97:  # Class probability threshod is 0.97
           extensions1.append(results)
        else:
           extensions0.append(results)
        if result > 0.90:  # Class probability threshod is 0.90 
           extensions3.append(results)
        else:
           extensions2.append(results) 
        if result > 0.70:   # Class probability threshod is 0.70
           extensions5.append(results)
        else:
           extensions4.append(results)
        if result > 0.40:   # Class probability threshod is 0.40
           extensions7.append(results)
        else:
           extensions6.append(results)
        if result > 0.50:   # Class probability threshod is 0.50
           extensions9.append(results)
        else:
           extensions8.append(results)
        if result > 0.15:   # Class probability threshod is 0.15
           extensions11.append(results)
        else:
           extensions10.append(results)  
        if result > 0.05:   # Class probability threshod is 0.05
           extensions13.append(results)
        else:
           extensions12.append(results)
    #print(sub_folder_path, end="\r \n")
    ## The majority voting at Patient's level
    if len(extensions1) > * len(extensions0):
      print(fldr, colored("NON-COVID", 'red'), len(extensions1), "to", len(extensions0))
      noncovidd.append(fldr)  
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions0), "to", len(extensions1))
      covidd.append(fldr)    
    if len(extensions3) > * len(extensions2):
      print (fldr, colored("NON-COVID", 'red'), len(extensions3), "to", len(extensions2))
      noncoviddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions2), "to", len(extensions3))
      coviddd.append(fldr)
    if len(extensions5) > * len(extensions4):
      print (fldr, colored("NON-COVID", 'red'), len(extensions5), "to", len(extensions4))
      noncovidddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions5), "to", len(extensions4))
      covidddd.append(fldr)
    if len(extensions7) > * len(extensions6):
      print (fldr, colored("NON-COVID", 'red'), len(extensions7), "to", len(extensions6))
      noncoviddddd.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions6), "to", len(extensions7))
      coviddddd.append(fldr)
    if len(extensions9) > * len(extensions8):
      print (fldr, colored("NON-COVID", 'red'), len(extensions9), "to", len(extensions8))
      noncovidd6.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions8), "to", len(extensions9))
      covidd6.append(fldr)
    if len(extensions11) > * len(extensions10):
      print (fldr, colored("NON-COVID", 'red'), len(extensions11), "to", len(extensions10))
      noncovidd7.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions10), "to", len(extensions11))
      covidd7.append(fldr)
    if len(extensions13) > len(extensions12):
      print (fldr, colored("NON-COVID", 'red'), len(extensions13), "to", len(extensions12))
      noncovidd8.append(fldr)
    else:
      print (fldr, colored("COVID", 'blue'), len(extensions12), "to", len(extensions13))
      covidd8.append(fldr)
       
    extensions0=[]
    extensions1=[]
    extensions2=[]
    extensions3=[]
    extensions4=[]
    extensions5=[]
    extensions6=[]
    extensions7=[]
    extensions8=[]
    extensions9=[]
    extensions10=[]
    extensions11=[]
    extensions12=[]
    extensions13=[]

#Checking the results
print(len(covidd))
print(len(coviddd))
print(len(covidddd))
print(len(coviddddd))
print(len(covidd6))
print(len(covidd7))
print(len(covidd8))
print(len(noncovidd))
print(len(noncoviddd))
print(len(noncovidddd))
print(len(noncoviddddd))
print(len(noncovidd6))
print(len(noncovidd7))
print(len(noncovidd8))
print(len(covidd+noncovidd))
print(len(coviddd+noncoviddd))
print(len(covidddd+noncovidddd))
print(len(coviddddd+noncoviddddd))
print(len(covidd6+noncovidd6))
print(len(covidd7+noncovidd7))
print(len(covidd8+noncovidd8))