# COV19D
The project includes working on a chest CT-scan series of images aiming to develop an automated classification algorithm for Covid-19/non-Covid-19 diagnosis. The project is towards the international COV19D cmmpetetion described in section B at https://mlearn.lincoln.ac.uk/mia-cov19d/.
Images are all in JPG fomrat.
## The codes
The codes use CNN model and transfer learning methods:  <br />        
1. CNN Model: 
<br /> A) The code '3D_Image_Classification.ipynb' introduces a classification method based on a CNN model with Macro f1 loss function. The results in terms of accuracy, macro f1 loss function, and macro f1 matrics over 200 epoches were shown. The value of macro f1 didn't show significant improvement while training the model. On the contrary, macro f1 scores kept oscillating about 7.0 over the training epochs.  <br />
<br /> B) The Code '3D_Image_Classification_binary_cross_loss.ipynb' introduces the same CNN model with binary crossentropy as a losss function. The results shows the improvment on the accuracy and loss of the CNN model over 20 epochs. <br /><br />
2. Transfer Learning and a classifier Model - VGG16 model:
<br /> The code 'Transfer_Learning_VGG16_RF.ipynb' introduces a transfer learning method using VGG16 model for feature exctraction and Random Forest  to classify the images. Only 60 CT images were used for training and the same number of images for validation, half of the images were Covid and the other half were non-covid. The Macro f1 score reached about 0.58. <br /><br />
3. Customized-CNN-Model-and-Tranfer-Learning-Model.py; Customized CNN model for image classification. The model achieved accuracy of about 70% and Macro F1 score of about 0.650. The proposed model is compared to a transfer learning model, MobileNet, for feature exctraction followed by a random forest classifier to take the final decision.

