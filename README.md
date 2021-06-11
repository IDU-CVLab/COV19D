# COV19D
The project includes working on a chest CT-scan series of images and masks in order to develop an automated classification algorithm for Covid-19/non-Covid-19 diagnosis. The project is towards the international COV19D cmmpetetion described in section B at https://mlearn.lincoln.ac.uk/mia-cov19d/.
## The codes
The codes try to work on either homogenious model or hybrid models:          
1. CNN Model: The code '3D_Image_Classification.ipynb' introduces a siple classification method based on a CNN model. The results in terms of accuracy, macro f1 loss function, and macro f1 matrics over 200 epoches were shown. The value of macro f1 didn't show significant improvement neither on the training nor on the validation set while training the model. The macro f1 scores ocilated around an average of about 7 over the 200 epochs.
2. Transfer Learning Hybrid Model:
