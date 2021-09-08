# COV19D
The project includes working on a chest CT-scan series of images aiming to develop an automated classification algorithm for Covid-19/non-Covid-19 diagnosis. The project is towards the international COV19D competetion described in section B at https://mlearn.lincoln.ac.uk/mia-cov19d/.

## The codes
The codes use CNN model and transfer learning methods:  <br />        
1.  The Code '3D_Image_Classification_binary_loss_function.ipynb' is a CNN model with binary crossentropy as a losss function. The results shows the improvments on the accuracy  of the CNN model over 20 epochs. <br /><br />
2. The code 'Customized-CNN-Model-and-Tranfer-Learning-Model.py' is a customized CNN model for image classification. The model achieved accuracy of about 70% and Macro F1 score of about 0.650. The proposed model is compared to a transfer learning model, MobileNet, for feature exctraction followed by a random forest classifier to take the final decision.

![Optional Text](../master/Figures/trainandtestacc.png)
