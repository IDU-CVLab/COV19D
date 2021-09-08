# COV19D
The project includes working on a chest CT-scan series of images aiming to develop an automated classification algorithm solution for Covid-19/non-Covid-19 diagnosis. The dataset is called COV19-CT-DB obtained during the international COV19D competetion described in section B at https://mlearn.lincoln.ac.uk/mia-cov19d/.

## THE CODES
The codes use CNN model and transfer learning methods:  <br />        
1.  The Code '3D_Image_Classification_binary_loss_function.ipynb' is a CNN model with binary crossentropy as a losss function. The results shows the improvments on the accuracy  of the CNN model over 20 epochs. <br />
2. The code 'Customized-CNN-Model-and-Tranfer-Learning-Model.py' is a customized CNN model for image classification. The model achieved accuracy of about 70% and Macro F1 score of about 0.650. The proposed model is compared to a transfer learning model, MobileNet, for feature exctraction followed by a random forest classifier to take the final decision. <br/>  

3. The code 'COV19-CT-DB-Customized-CNN-Model-.py' Proposes a learning rate schedualer to the model. The results are:
Macro F1 score is 0.924, accuracy, recall and precision results can be seen in the following two figures: <br/>
![Training and Testing Accuracy Figure](../master/Figures/trainandtestacc.png)               <br />
![Testing_Recall_and_Teasting_Precision_Figure](../master/Figures/recalandprecision.png)

