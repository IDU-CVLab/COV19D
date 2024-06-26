[![DOI:10.1080/21681163.2023.2219765](http://img.shields.io/badge/DOI-10.1080/21681163.2023.2219765-B31B1B.svg)](https://doi.org/10.1080/21681163.2023.2219765) 

## Database  
* The dataset is named COV19-CT-DB. It is obtained at [AI-enabled Medical Image Analysis Workshop and Covid-19 Diagnosis Competition (ICCV 2021 Workshop: MIA-COV19D 2021)](https://mlearn.lincoln.ac.uk/mia-cov19d/). The project is conducted using chest CT-scan series of images aiming to develop an automated solution for Covid-19/non-Covid-19 diagnosis. <br/>
* The team (IDU-CVLab) introduces a light weight solution for COVID-19 diagnosis and is listed on the leaderboard [here](https://cpb-eu-w2.wpmucdn.com/blogs.lincoln.ac.uk/dist/c/6133/files/2022/03/iccv_cov19d_leaderboard.pdf). <br/>

## Methodology
### Proposed CNN Model
The algorithm introduces a less hand-engineered CNN model Architecture for automated COVID-19 diagnosis. <br/>The CNN model achitechture is in picture below: <br/>
<p align="center">
  <img src="https://github.com/IDU-CVLab/COV19D/blob/main/Figures/CNN-Model-Architecture.png" />
</p>       

</br>
* The work makes use of the above-mentioned CNN model, with slices preprocessing before training. <br/>   

### Slice Processing     
Slices preprocessing -as in [here](https://github.com/IDU-CVLab/Images_Preprocessing)- includes a static rectangular cropping, and non-representative slcies removal in each CT scan. For more details, please refer to the peer-reviewed paper. <br /> 

### Robustness to Noise
Gussian noise was added to the images in the original validation set and then images are processed as mentioned above and the pretrained and saved CNN model was tested on the newly-created noisey images. This step aims to check our method's performance against noisey data to prove the solution's robustness. The python code is named "Noisey-Images-mage-Processing-Light-CNN-Model.py".

### Training the Model
To replicate the codes, the following must be noted:
* To run the code properly you would need a training set of images and a validation set of images.
* The images must be put in the appropriate directories. With that, the directory of training and validation images included in the code should be changed to match the directory where your image datasets are located. This method is following the documentation for ‘imagedatagenerator’ and ‘flow_from_directory’ at https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
* Please note: this is a binary classificaiton task. To replicate the code using muti-class classificaiton data, you need to modify the model's output to suit your task.

## Dependencies
▪ numpy == 1.19.5 <br/>
▪ matplotlib == 3.3.4 <br/>
▪ tensorflow == 2.5.0 <br/>
▪ CV2 == 4.5.4 <br />
▪ sklearn == 0.24.2 <br />   

## Citation
If you find the this method useful, please consider citing: <br/> <br/>
>@article{doi:10.1080/21681163.2023.2219765, <br/>
author = {Kenan Morani and D. Unay}, <br/>
title = {Deep learning-based automated COVID-19 classification from computed tomography images}, <br/>
journal = {Computer Methods in Biomechanics and Biomedical Engineering: Imaging \& Visualization}, <br/>
volume = {0}, <br/>
number = {0}, <br/>
pages = {1-16}, <br/>
year  = {2023}, <br/>
publisher = {Taylor & Francis}, <br/>
doi = {10.1080/21681163.2023.2219765}, <br/>
