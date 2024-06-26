#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 13:14:14 2023

@author: idu
"""
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical

import matplotlib.pyplot as plt
from PIL import Image
from termcolor import colored  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import numpy as np
import pandas as pd
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence, to_categorical

import matplotlib.pyplot as plt
from PIL import Image
from termcolor import colored  
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img


########################### Adding Noise to The Original Images in the validation set [Gusiian Noise] ########################
###########################^^############################^^^^################################################^^^^^^^^^
   

# Function to add Gaussian noise to an image
def add_gaussian_noise(image, mean=0, sigma=25):
    """Add Gaussian noise to an image"""
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    noisy_image = image + gaussian_noise
    noisy_image = np.clip(noisy_image, 0, 255)  # Ensure pixel values are valid
    return noisy_image

# Function to load an image
def load_image(image_path, target_size=(224, 224)):
    """Load a grayscale image"""
    img = load_img(image_path, target_size=target_size, color_mode="grayscale")
    img_array = img_to_array(img)
    return img_array

# Function to save an image
def save_image(image, save_path):
    """Save a grayscale image to the specified path in JPEG format"""
    save_img(save_path, image, data_format='channels_last', file_format='jpeg')

# Path to validation set folder
validation_set_path = '/home/idu/Desktop/COV19D/val'
output_path = '/home/idu/Desktop/COV19D/val-noise-added'

# Function to add noise to images and save them
def add_noise_and_save_images(image_paths, output_path, validation_set_path):
    for image_path in image_paths:
        try:
            # Determine output path for noisy image
            relative_path = os.path.relpath(image_path, start=validation_set_path)
            noisy_image_save_path = os.path.join(output_path, relative_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(noisy_image_save_path), exist_ok=True)
            
            # Load image
            image = load_image(image_path)
            
            # Add Gaussian noise
            noisy_image = add_gaussian_noise(image)
            
            # Save noisy image
            save_image(noisy_image, noisy_image_save_path)
            print(f"Saved noisy image: {noisy_image_save_path}")
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

# Get all image paths recursively from covid and non-covid directories
covid_image_paths = []
non_covid_image_paths = []

for root, dirs, files in os.walk(os.path.join(validation_set_path, 'covid')):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            covid_image_paths.append(os.path.join(root, file))

for root, dirs, files in os.walk(os.path.join(validation_set_path, 'non-covid')):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg')):
            non_covid_image_paths.append(os.path.join(root, file))

# Process and save noisy covid images
print("Processing and saving COVID images:")
add_noise_and_save_images(covid_image_paths, output_path, validation_set_path)

# Process and save noisy non-covid images
print("\nProcessing and saving Non-COVID images:")
add_noise_and_save_images(non_covid_image_paths, output_path, validation_set_path)



#######################################################################################
######################## Processing Newly Created Noisey IMages ############################################
#######################################################################################

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





####################################################################################
######################## Light Weight CNN Model for classification of Noisey Images 
####################################################################################

# Loaing our saved model

model = keras.models.load_model("/home/idu/Desktop/COV19D/saved-models/CNN model/imageprocess-sliceremove-cnn.h5")


#Making predictions on the validation set of noisey images COV19-CT-DB

## Choosing the directory where the test/validation data is at
from termcolor import colored  # Importing colored for colored console output
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

folder_path = '/home/idu/Desktop/COV19D/val-noise-added/covid'  # Change as needed

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

results = 1
for fldr in os.listdir(folder_path):
    sub_folder_path = os.path.join(folder_path, fldr)
    for filee in os.listdir(sub_folder_path):
        file_path = os.path.join(sub_folder_path, filee)

        try:
            c = load_img(file_path, color_mode='grayscale', target_size=(227, 300)) ## The image input size expected
            c = img_to_array(c)
            c = np.expand_dims(c, axis=0)
            c /= 255.0
            result = model.predict(c)  # Probability of 1 (non-covid)

            if result > 0.97:  # Class probability threshold is 0.97
                extensions1.append(results)
            else:
                extensions0.append(results)
            if result > 0.90:  # Class probability threshold is 0.90
                extensions3.append(results)
            else:
                extensions2.append(results)
            if result > 0.70:  # Class probability threshold is 0.70
                extensions5.append(results)
            else:
                extensions4.append(results)
            if result > 0.40:  # Class probability threshold is 0.40
                extensions7.append(results)
            else:
                extensions6.append(results)
            if result > 0.50:  # Class probability threshold is 0.50
                extensions9.append(results)
            else:
                extensions8.append(results)
            if result > 0.15:  # Class probability threshold is 0.15
                extensions11.append(results)
            else:
                extensions10.append(results)
            if result > 0.05:  # Class probability threshold is 0.05
                extensions13.append(results)
            else:
                extensions12.append(results)
        except Exception as e:
            print(f"Error processing image {file_path}: {e}")
            continue

    # The majority voting at Patient's level
    if len(extensions1) > len(extensions0):
        print(fldr, colored("NON-COVID", 'red'), len(extensions1), "to", len(extensions0))
        noncovidd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions0), "to", len(extensions1))
        covidd.append(fldr)
    if len(extensions3) > len(extensions2):
        print(fldr, colored("NON-COVID", 'red'), len(extensions3), "to", len(extensions2))
        noncoviddd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions2), "to", len(extensions3))
        coviddd.append(fldr)
    if len(extensions5) > len(extensions4):
        print(fldr, colored("NON-COVID", 'red'), len(extensions5), "to", len(extensions4))
        noncovidddd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions5), "to", len(extensions4))
        covidddd.append(fldr)
    if len(extensions7) > len(extensions6):
        print(fldr, colored("NON-COVID", 'red'), len(extensions7), "to", len(extensions6))
        noncoviddddd.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions6), "to", len(extensions7))
        coviddddd.append(fldr)
    if len(extensions9) > len(extensions8):
        print(fldr, colored("NON-COVID", 'red'), len(extensions9), "to", len(extensions8))
        noncovidd6.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions8), "to", len(extensions9))
        covidd6.append(fldr)
    if len(extensions11) > len(extensions10):
        print(fldr, colored("NON-COVID", 'red'), len(extensions11), "to", len(extensions10))
        noncovidd7.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions10), "to", len(extensions11))
        covidd7.append(fldr)
    if len(extensions13) > len(extensions12):
        print(fldr, colored("NON-COVID", 'red'), len(extensions13), "to", len(extensions12))
        noncovidd8.append(fldr)
    else:
        print(fldr, colored("COVID", 'blue'), len(extensions12), "to", len(extensions13))
        covidd8.append(fldr)

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

# Checking the results
# print(len(covidd))
# print(len(coviddd))
print(len(covidddd))
print(len(coviddddd))
print(len(covidd6))
print(len(covidd7))
# print(len(covidd8))
# print(len(noncovidd))
# print(len(noncoviddd))
print(len(noncovidddd))
print(len(noncoviddddd))
print(len(noncovidd6))
print(len(noncovidd7))
# print(len(noncovidd8))
# print(len(covidd+noncovidd))
# print(len(coviddd+noncoviddd))
print(len(covidddd+noncovidddd))
print(len(coviddddd+noncoviddddd))
print(len(covidd6+noncovidd6))
print(len(covidd7+noncovidd7))
# print(len(covidd8+noncovidd8))
