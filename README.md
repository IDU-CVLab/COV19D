## Database

[![DOI:10.48550/arXiv.2111.11191](http://img.shields.io/badge/DOI-10.1101/2021.01.08.425840-B31B1B.svg)](https://doi.org/10.48550/arXiv.2111.11191)   
* The dataset is called COV19-CT-DB obtained at [AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (ICCV 2021 Workshop: MIA-COV19D 2021)](https://mlearn.lincoln.ac.uk/mia-cov19d/). The project is conducted using chest CT-scan series of images aiming to develop an automated solution for Covid-19/non-Covid-19 diagnosis. <br/>
* The team (IDU-CVLab) introduces a light weight solution for COVID-19 diagnosis and is listed on the leaderboard [here](https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/c/6133/files/2022/03/iccv_cov19d_leaderboard.pdf). <br/>

## Proposed CNN Model
The algorithm introduces a less hand-engineered CNN model Architecture for automated COVID-19 diagnosis. <br/>The CNN model achitechture is: <br/>
<p align="center">
  <img src="https://github.com/IDU-CVLab/COV19D/blob/main/Figures/CNN-Model-Architecture.png" />
</p>      


## Full Method
* The work makes use of the above mentioned CNN model with slices preprocessed before training. 
* Slices preprocessing ,as [here](https://github.com/IDU-CVLab/Images_Preprocessing) ,includes a static rectangular croping to the Region of Interest (ROI) in the CT sclices, and statitical methods for uppermost and lowermost removal of the slcies in each CT scan image, as in the attached paper. <br />

<br/>
<b> Dependencies: </b><br/>
▪ numpy == 1.19.5 <br/>
▪ matplotlib == 3.3.4 <br/>
▪ tensorflow == 2.5.0 <br/>
▪ CV2 == 4.5.4 <br />
▪ sklearn == 0.24.2 <br />

## Training the Model
To replicate the codes, the following must be noted:
1. To run the code properly you would need a training set of images and a validation set of images.
2. The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator <br />       

## Citation
● If you find the method useful this method, please considering citing: <br/>
@article{Morani2021DeepLB, <br/>
  title={Deep Learning Based Automated COVID-19 Classification from Computed Tomography Images}, <br/>
  author={Kenan Morani and Devrim {\"U}nay}, <br/>
  journal={ArXiv}, <br/>
  year={2021}, <br/>
  volume={abs/2111.11191} <br/>
}

## Collaboration
* Please get in touch if you wish to collaborate or wish to request the pre-trained models.
