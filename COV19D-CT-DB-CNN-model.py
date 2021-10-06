#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 00:28:03 2021

@author: idu
"""

import os, glob
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import backend as K
from tensorflow.keras.optimizers import Adam
from keras import layers, models
from tensorflow.keras import activations

####### Generatiiing Data

# batch_size = 64
# batch_size = 64
batch_size = 128
SIZE = 512

train_datagen = ImageDataGenerator(rescale=1./255)                             
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

y_train = train_generator.classes
y_val = val_generator.classes

#### Display images from the train_generator
for _ in range(5):
    img, label = next(train_generator)
    print(img.shape)
    #print(label[0])
    print(train_generator.classes[0])
    plt.imshow(img[0])
    plt.show()

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
    model.add(layers.Dense(1,activation='sigmoid'))
    
    
    return model


model = make_model()

###################################### Compiling and Training the model
n_epochs= 70

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

history=model.fit(train_generator, 
                  steps_per_epoch=250,
                  validation_data=val_generator,
                  validation_steps=78,
                  verbose=2,
                  epochs=n_epochs)


########################33 Evaluating
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

###### Save the model
model.save("model.h5")

######################## Visualization of the model


import visualkeras
from collections import defaultdict
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from collections import defaultdict


color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = 'blue'
color_map[ZeroPadding2D]['fill'] = 'blue'
color_map[Dropout]['fill'] = 'blue'
color_map[MaxPooling2D]['fill'] = 'blue'
color_map[Dense]['fill'] = 'blue'
color_map[Flatten]['fill'] = 'blue'

visualkeras.layered_view(model, spacing=5, legend=True, color_map=color_map)

visualkeras.layered_view(model, legend=True)

########################3 Visualizing CNN Layer Filters
model.summary()

layer = model.layers #Conv layers at 0, 4, 8, 12
print(layer)
filters, biases = model.layers[0].get_weights()
print(layer[0].name, filters.shape)

# plot filters
fig1=plt.figure(figsize=(8, 12))
columns = 4
rows = 4
n_filters = columns * rows
for i in range(1, n_filters +1):
    f = filters[:, :, :, i-1]
    fig1 =plt.subplot(rows, columns, i)
    fig1.set_xticks([])  #Turn off axis
    fig1.set_yticks([])
    plt.imshow(f[:, :, 0], cmap='gray') #Show only the filters from 0th channel (R)
    #ix += 1
plt.show()    


#### Now plot filter outputs    
from keras.models import Model
#Define a new truncated model to only include the conv layers of interest
conv_layer_index = [0, 4, 8, 12]  #TO define a shorter model
outputs = [model.layers[i].output for i in conv_layer_index]
model_short = Model(inputs=model.inputs, outputs=outputs)
print(model_short.summary())

from keras.preprocessing.image import load_img, img_to_array
img = load_img('/home/idu/Desktop/COV19D/val-small/covid/ct_scan_0/170.jpg', target_size=(512, 512)) 
from skimage.color import rgb2gray
img = rgb2gray(img)
# convert the image to an array
img = img_to_array(img)
# expand dimensions to match the shape of model input
img = np.expand_dims(img, axis=0)

# Generate feature output by predicting on the input image
Prid_output = model_short.predict(img)

columns = 4
rows = 4
for ftr in Prid_output:
    #pos = 1
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig =plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='gray')
        #pos += 1
    plt.show()
    

img2 = load_img('/home/idu/Desktop/COV19D/val-small/non-covid/ct_scan_0/190.jpg', target_size=(512, 512)) 

# convert the image to an array
img2 = img_to_array(img2)
# expand dimensions to match the shape of model input
img2 = np.expand_dims(img2, axis=0)

# Generate feature output by predicting on the input image
Prid_output2 = model_short.predict(img2)

columns = 4
rows = 4
for ftr in Prid_output2:
    #pos = 1
    fig=plt.figure(figsize=(12, 12))
    for i in range(1, columns*rows +1):
        fig =plt.subplot(rows, columns, i)
        fig.set_xticks([])  #Turn off axis
        fig.set_yticks([])
        plt.imshow(ftr[0, :, :, i-1], cmap='gray')
        #pos += 1
    plt.show()
    
################################3 WROKING ON TEST DATASET
    
test_datagen = ImageDataGenerator(rescale=1./255)
                                   
test_generator = test_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/test/',  
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        classes = ['covid','non-covid'],
        class_mode='binary')

y_test = test_generator.classes


img2 = load_img('/home/idu/Desktop/COV19D/test/covid/109.jpg', target_size=(128, 128)) 

img2 = img_to_array(img2)
# expand dimensions to match the shape of model input
img2 = np.expand_dims(img2, axis=0)

# Generate feature output by predicting on the input image
prediction = model.predict(img2)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_test, prediction))
print ("F1_score (COVID) = ", metrics.f1_score(y_test, prediction, pos_label = 0))
print ("F1_score (NON-COVID) = ", metrics.f1_score(y_test, prediction,pos_label = 1))
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
print("Average Precision", average_precision_score(y_test, prediction))
print("Average Recall", recall_score(y_test, prediction))
print ("Macro F1_score = ", metrics.f1_score(y_test, prediction, average='macro'))

##### Further evaluation

eval = model.evaluate (test_generator)
