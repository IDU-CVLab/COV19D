# COV19D
The project includes working on a chest CT-scan series of images and masks in order to develop an automated classification algorithm for Covid-19/non-Covid-19 diagnosis. The project is towards the international COV19D cmmpetetion described in section B at https://mlearn.lincoln.ac.uk/mia-cov19d/.
Images are all in JPG fomrat.
## The codes
The codes try to work on either homogenious model or hybrid models:          
1. CNN Model: 
<br /> A) The code '3D_Image_Classification.ipynb' introduces a classification method based on a CNN model with Macro f1 loss function. The results in terms of accuracy, macro f1 loss function, and macro f1 matrics over 200 epoches were shown. The value of macro f1 didn't show significant improvement while training the model. On the contrary, macro f1 scores kept oscillating about 7.0 over the training epochs.
<br /> B) The Code '3D_Image_Classification_binary_cross_loss.ipynb' introduces the same CNN model with binary crossentropy as a losss function. The results shows the improvment on the accuracy and loss of the CNN model over 20 epochs.
2. Transfer Learning Hybrid Model:
<br /> The code 'TRansfer_Learning_VGG16_RF.ipynb' introduces a transfer learning method using VGG16 model for feature exctraction and Random Forest ched  to classify the images. Only 60 CT images were used for training and the same number of images for validation, half of the images were Covid and the other half were non-covid. The Macro f1 score reached about 0.57.
