# COV19D
The project includes working on a chest CT-scan series of images and masks in order to develop an automated classification algorithm for Covid-19/non-Covid-19 diagnosis. The project is towards the international COV19D cmmpetetion described in section B at https://mlearn.lincoln.ac.uk/mia-cov19d/.
## The codes
The codes try to work on either homogenious model or hybrid models:          
1. CNN Model: 
<br /> The code '3D_Image_Classification.ipynb' introduces a classification method based on a CNN model with Macro f1 loss function. The results in terms of accuracy, macro f1 loss function, and macro f1 matrics over 200 epoches were shown. The value of macro f1 didn't show significant improvement while training the model. On the contrary, macro f1 scores kept oscillating about 7.0 over the spochs.
2. Transfer Learning Hybrid Model:
