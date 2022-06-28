## COV19-CT-DB Database

[![DOI:10.48550/arXiv.2111.11191](http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg)](https://doi.org/10.48550/arXiv.2111.11191)

* The project includes working on a chest CT-scan series of images aiming to develop an automated classification algorithm solution for Covid-19/non-Covid-19 diagnosis. The dataset is called COV19-CT-DB obtained at AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (MIA-COV19D), https://mlearn.lincoln.ac.uk/mia-cov19d/. <br/>
* The team (IDU-CVLab) is on the leaderboard [here](https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/c/6133/files/2022/03/iccv_cov19d_leaderboard.pdf). <br/>

## THE CODES
To replicate the codes, the following must be noted:
1. To run the code properly you would need a training set of images and a validation set of images.
2. The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator <br /> 

The algorithm was developed in two versions or steps as follows:  <br />        

### 1. Version1 
**The Code 'COV19-CT-DB-CNN-Model.py'** <br/>
The algorithm introduces a less hand-engineered CNN model Architecture for automated COVID-19 diagnosis. <br/>The CNN model achitechture is: <br/>
<p align="center">
  <img src="https://github.com/IDU-CVLab/COV19D/blob/main/Figures/CNN-Model-Architecture.png" />
</p>      
<br/>
<b> Dependencies: </b><br/>
-numpy == 1.19.5 <br/>
-matplotlib == 3.3.4 <br/>
-tensorflow == 2.5.0 <br/>

### 2. Version2 
**The Code 'Static_Cropping_Deeplearning_Model_for_CT_images.py'** <br/>
The work makes use of the above mentioned CNN model with images preprocessed before training. The preprocessing includes a static rectangular croping to the Region of Interest (ROI) in the CT sclices and statitical methods for uppermost and lowermost removal of the slcies in each CT scan image. <br />
The code can be devided in two parts: <br/><br/>
_PartI_. The code without slice processing and parameters tuning [as in version1]. <br />

_PartII_. The code with rectangle-shape cropping and hyperparameters tuning. <br />
● Further dependencies used for this part are: <br />
▪ CV2 == 4.5.4 <br />
▪ sklearn == 0.24.2 <br />

## Cite 
● If you use this method, please cite: <br/>
@article{Morani2021DeepLB, <br/>
  title={Deep Learning Based Automated COVID-19 Classification from Computed Tomography Images}, <br/>
  author={Kenan Morani and Devrim {\"U}nay}, <br/>
  journal={ArXiv}, <br/>
  year={2021}, <br/>
  volume={abs/2111.11191} <br/>
} 


