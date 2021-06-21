import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import nibabel as nib

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow import keras
from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as K
import seaborn as sns
from keras.applications.vgg16 import VGG16

##########################################Extracting .rar files 
import rarfile
import unrar
!unrar x '/home/kenan/Desktop/COV19D/rar/train/covid.rar' '/home/kenan/Desktop/COV19D/train/'
!unrar x '/home/kenan/Desktop/COV19D/rar/train/non-covid.rar' '/home/kenan/Desktop/COV19D/train/'

!unrar x '/home/kenan/Desktop/COV19D/rar/validation/covid.rar' '/home/kenan/Desktop/COV19D/validation/'
!unrar x '/home/kenan/Desktop/COV19D/rar/validation/non-covid.rar' '/home/kenan/Desktop/COV19D/validation/'


########################################Configuring the train and validation paths
one_image = '/home/kenan/Desktop/COV19D/train/covid/ct_scan_0/0.jpg'
image = Image.open(one_image)
# summarize some details about the image
print(image.format)
print(image.size)
print(image.mode)

#######################################Generating data with rescaling and binary labels from the images
batch_size = 32
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        '/home/kenan/Desktop/COV19D/train/', 
        color_mode = "grayscale",
        target_size=(128, 128),  # All images are 512 * 512
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1/255)
val_generator = val_datagen.flow_from_directory(
        '/home/kenan/Desktop/COV19D/validation/', 
        color_mode = "grayscale",
        target_size=(128, 128),  
        batch_size=batch_size,
        classes = ['covid','non-covid'],
        class_mode='binary')

y_train = train_generator.classes
y_val = val_generator.classes 

###############################################33

####################################333
SIZE = 128
#Load model wothout classifier/fully connected layers
VGG_model = VGG16(include_top=False, weights=None, input_shape=(SIZE, SIZE, 1))
#Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
for layer in VGG_model.layers:
	layer.trainable = False
VGG_model.summary()  #Trainable parameters will be 0

#Now, let us use features from convolutional network for RF
feature_extractor=VGG_model.predict(train_generator)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
RF_model.fit(features, y_train) 
########################################################### Checking accuracy for Transfer learning##########33
#Send test data through same feature extractor process
X_val_feature = VGG_model.predict(val_generator)
X_val_features = X_val_feature.reshape(X_val_feature.shape[0], -1)

#Now predict using the trained model
prediction_RF = RF_model.predict(X_val_features)
#Inverse le transform to get original label back. 
#prediction_RF = le.inverse_transform(prediction_RF)

#Print overall accuracy
from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_val, prediction_RF))
print ("Macro f1_score = ", metrics.f1_score(y_val, prediction_RF))


#Confusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, prediction_RF)
#print(cm)
sns.heatmap(cm, annot=True)

# Displaying an image in the train_generator
batch=next(train_generator) 
print(batch[0].shape)
img=batch[0][0] 
print (img.shape)
plt.imshow(img)

input_img = np.expand_dims(img, axis=0) #Expand dims so the input is (num images, x, y, c)
input_img_feature=VGG_model.predict(input_img)
input_img_features=input_img_feature.reshape(input_img_feature.shape[0], -1)
prediction_RF = RF_model.predict(input_img_features)[0] 
print("The prediction for this image is: ", prediction_RF)
print("The actual label for this image is: ", y_train[0])
