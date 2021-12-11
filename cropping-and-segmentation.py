#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 21:55:28 2021

@author: kenaa
"""

import numpy as np
import glob
import matplotlib.pyplot as plt
import skimage.io
import skimage.color
import skimage.filters
from skimage.segmentation import chan_vese
from skimage.io import imread
from skimage.color import rgb2gray
import cv2

img = cv2.imread('/home/kenaa/Desktop/COV19D/validation/covid/ct_scan_0/195.jpg')
# Displaying the original Image
fig, ax = plt.subplots()
plt.imshow(img, cmap='gray')
plt.show()
# Print image shape
print(img.shape) 
# Cropping an image
img = img[60:450, 0:512]
gray_image = skimage.color.rgb2gray(img)

# Blurring the image using Guassian
blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)

# Plotting the Blurred Image 
fig, ax = plt.subplots()
plt.imshow(blurred_image, cmap='gray')
plt.show()

# create a histogram of the blurred grayscale image
histogram, bin_edges = np.histogram(blurred_image, bins=256, range=(0.0, 1.0))
histogram.max()

# Plotting the histogram
plt.plot(bin_edges[0:-1], histogram)
plt.title("Grayscale Histogram")
plt.xlabel("grayscale value")
plt.ylabel("pixels")
plt.xlim(0, 1.0)
plt.show()

# create a mask based on the threshold
t = 0.4
binary_mask = blurred_image < t

# Plotting the mask
fig, ax = plt.subplots()
plt.imshow(binary_mask, cmap='gray')
plt.show()
