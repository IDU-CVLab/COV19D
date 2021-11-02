# COV19-CT-DB
The project includes working on a chest CT-scan series of images aiming to develop an automated classification algorithm solution for Covid-19/non-Covid-19 diagnosis. The dataset is called COV19-CT-DB obtained at AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (MIA-COV19D), https://mlearn.lincoln.ac.uk/mia-cov19d/. <br/>

## THE CODES
The codes develop CNN models as well as transfer learning models as follows:  <br />        
1.  The Code '3D_Image_Classification_binary_loss_function.ipynb' is a CNN model with binary crossentropy as a losss function. <br />
2. The code 'Customized-CNN-Model-and-Tranfer-Learning-Model.py' is a customized CNN model with a linearly changing learning rate for image classification. The proposed model is compared to a transfer learning model, MobileNet, for feature exctraction followed by a random forest classifier to take the final decision. The lattar model achieves less macro F1 score. <br/>  

3. The code 'COV19-CT-DB-CNN-Model.py' Proposes a less_hand engineered CNN model Architecture for automated COVID-19 diagnosis <br/>
      The CNN model achitechture is: <br/>
<p align="center">
  <img src="https://github.com/IDU-CVLab/COV19D/blob/main/Figures/CNN-Model-Architecture.png" />
</p>      
<br/>
<b> Dependencies: </b><br/>
-numpy == 1.19.5 <br/>
-matplotlib == 3.3.4 <br/>
-tensorflow == 2.5.0 <br/>




