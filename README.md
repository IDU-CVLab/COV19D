# COV19-CT-DB
The project includes working on a chest CT-scan series of images aiming to develop an automated classification algorithm solution for Covid-19/non-Covid-19 diagnosis. The dataset is called COV19-CT-DB obtained at AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (MIA-COV19D), https://mlearn.lincoln.ac.uk/mia-cov19d/. <br/>

## THE CODES
To replicate the codes, the following must be noted:
1. To run the code properly you would need training set of images and validation set of images.
2. The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator <br /> 

The codes were developed in two versions as follows:  <br />        

### 1. Version1 
Version1 is the code 'COV19-CT-DB-CNN-Model.py' Proposes a less hand-engineered CNN model Architecture for automated COVID-19 diagnosis. <br/>The CNN model achitechture is: <br/>
<p align="center">
  <img src="https://github.com/IDU-CVLab/COV19D/blob/main/Figures/CNN-Model-Architecture.png" />
</p>      
<br/>
<b> Dependencies: </b><br/>
-numpy == 1.19.5 <br/>
-matplotlib == 3.3.4 <br/>
-tensorflow == 2.5.0 <br/>
<br />  
ArXiv paper (version 1) can be found at https://arxiv.org/abs/2111.11191

<br /> <br/> 
### 2. Version2 
Version2 is the code "Static_Cropping_Deeplearning_Model_for_CT_images.py"  <br/> 
Uses the above mentioned CNN model with images preprocessed before training. The preprocessing includes a static rectangular croping to the Region of Interest (ROI) in the CT sclices and statitical methods for uppermost and lowermost removal of the slcies in each CT scan image. <br />
The code can be devided in two parts: <br/><br/>
_Part1_. The code without slice processing and parameters tuning: <br />
● The code can be found at https://github.com/IDUCVLab/COV19D/blob/main/COV19-CT-DB-CNN-model.py  <br />
● The code is written fully in Python using Spyder 3 IDE (.py code) and you should have the appropriate software/tools to use python code.  <br />
● Dependencies used to build the code are: <br />
▪ numpy == 1.19.5 <br />
▪ matplotlib == 3.3.4 <br />
▪ tensorflow == 2.5.0 <br />
● To run the code properly you would need a training set of images and a validation set of <br />
images.
● The images must be put in the appropriate directories. With that, the directory of training <br />
and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for
‘imagedatagenerator’ and ‘flow_from_directory’ at
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator <br />
● No further instructions are needed to run the code. <br /> <br/>
_PartII_. The code with rectangle-shape cropping and hyperparameters tuning: <br />
● Dependencies used to build the code are: <br />
▪ CV2 == 4.5.4 <br />
▪ sklearn == 0.24.2 <br />
 Slice preprocessing can be found at our lab’s github at https://github.com/IDUCVLab/ 
Images_Preprocessing. [Guided GRad-Cam Visualization, and Images processing 1 and 2]. <br />
 Gradcam visualization follows the code reference at: <br/>
https://stackoverflow.com/questions/66911470/how-to-apply-grad-cam-on-my-trainedmodel
{Last Access 13.12.2021} <br/>
 Blurring, binarization and segmentation follows the code reference: <br/>
https://datacarpentry.org/image-processing/07-thresholding/ {Last Access 15.12.2021}
 Region of interest cropping follows the code references: <br/>
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html {Last Access
15.12.2021} <br/>
https://stackoverflow.com/questions/56467902/select-a-static-roi-on-webcam-video-onpython-
opencv {Last Access 15.12.2021} <br/>
https://stackoverflow.com/questions/15341538/numpy-opencv-2-how-do-i-crop-nonrectangular-
region {Last Access 15.12.2021} <br/>
arXiv paper version 2 can be found at https://arxiv.org/abs/2111.11191v2 <br/> <br/>
If you use this method please reference: <br/>
@article{morani2021deep,<br />
  title={Deep Learning Based Automated COVID-19 Classification from Computed Tomography Images},<br />
  author={Morani, Kenan and Unay, Devrim},<br />
  journal={arXiv preprint arXiv:2111.11191},<br />
  year={2021}<br />
}
