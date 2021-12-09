# COV19-CT-DB
The project includes working on a chest CT-scan series of images aiming to develop an automated classification algorithm solution for Covid-19/non-Covid-19 diagnosis. The dataset is called COV19-CT-DB obtained at AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (MIA-COV19D), https://mlearn.lincoln.ac.uk/mia-cov19d/. <br/>

## THE CODES
To replicate the codes the following must be noted:
1. To run the code properly you would need training set of images and validation set of images.
2. The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator <br /> 

The codes develop CNN models as well as transfer learning models as follows:  <br />        
1.  The Code '3D_Image_Classification_binary_loss_function.ipynb' is a CNN model with binary crossentropy as a losss function. <br />
2. The code 'Customized-CNN-Model-and-Tranfer-Learning-Model.py' is a customized CNN model with a linearly changing learning rate for image classification. The proposed model is compared to transfer learning models, MobileNet, for feature exctraction followed by a random forest classifier to take the final decision. <br/>  

3. The code 'COV19-CT-DB-CNN-Model.py' Proposes a less hand-engineered CNN model Architecture for automated COVID-19 diagnosis <br/>
      The CNN model achitechture is: <br/>
<p align="center">
  <img src="https://github.com/IDU-CVLab/COV19D/blob/main/Figures/CNN-Model-Architecture.png" />
</p>      
<br/>
<b> Dependencies: </b><br/>
-numpy == 1.19.5 <br/>
-matplotlib == 3.3.4 <br/>
-tensorflow == 2.5.0 <br/>
<br />  
If you use this model, please refernce the following arXiv paper: <br />  
@misc{morani2021deep,
      title={Deep Learning Based Automated COVID-19 Classification from Computed Tomography Images}, 
      author={Kenan Morani and Devrim Unay},
      year={2021},
      eprint={2111.11191},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}


