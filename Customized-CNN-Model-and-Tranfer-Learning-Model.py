import os, glob
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from keras import backend as K
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

####### Generatiiing Data
batch_size = 32
SIZE = 128

train_datagen = ImageDataGenerator(rescale=1./255)
                                   
train_generator = train_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/train-small/',  ## 236364 CT images for training - (COV19D-CT-DB)
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        classes = ['covid','non-covid'],
        class_mode='binary')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        '/home/idu/Desktop/COV19D/val-small/',  ## 51808 CT images for teating (validation) - (COV19D-CT-DB)
        target_size=(SIZE, SIZE),
        batch_size=batch_size,
        color_mode='rgb',
        classes = ['covid','non-covid'],
        class_mode='binary')

y_train = train_generator.classes
y_val = val_generator.classes

################ A CNN Model for image classification
def make_model():
   
    model = models.Sequential()
    
    model.add(layers.Conv2D(16,(3,3),input_shape=(SIZE,SIZE,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(32,(3,3), padding="same"))  
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))   
    model.add(layers.MaxPooling2D((2,2)))
    
    model.add(layers.Conv2D(128,(3,3), padding="same"))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.MaxPooling2D((2,2)))
    
    
    model.add(layers.Flatten())
    
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(0.1))
    model.add(layers.Dropout(0.2))
      
    model.add(layers.Dense(1,activation='sigmoid'))
    
    
    return model


model = make_model()

######### Customizing the model

Initial_LR = 0.0007
def lr_scheduler(epoch):
        return Initial_LR ** epoch
    
model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(learning_rate=0.0001),
             metrics=[tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),'accuracy'])

history=model.fit(train_generator, 
                  steps_per_epoch=250,
                  validation_data=val_generator,
                  validation_steps=78,
                  verbose=2,
                  epochs=n_epochs,
                  callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler)])


print (history.history.keys())
            
Train_accuracy = history.history['accuracy']
print(Train_accuracy)
print(np.mean(Train_accuracy))
val_accuracy = history.history['val_accuracy']
print(val_accuracy)
print( np.mean(val_accuracy))

epochs = range(1, len(Train_accuracy)+1)

plt.figure(figsize=(12,6))
plt.plot(epochs, Train_accuracy, 'g', label='Training acc')
plt.plot(epochs, val_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.ylim(0.55,0.8)
plt.legend()

plt.show()

val_recall = history.history['val_recall_1']
avg_recall = np.mean(val_recall)
avg_recall

val_precision = history.history['val_precision_1']
avg_precision = np.mean(val_precision)
avg_precision

epochs = range(1, len(Train_accuracy)+1)
plt.figure(figsize=(12,6))
plt.plot(epochs, val_recall, 'g', label='Validation Recall')
plt.plot(epochs, val_precision, 'b', label='Validation Prcision')
plt.title('Validation recall and Validation Percision')
plt.xlabel('Epochs')
plt.ylabel('Recall and Precision')
plt.legend()
plt.ylim(0,1)

plt.show()



Macro_F1score = (2*avg_precision*avg_recall)/ (avg_precision + avg_recall)
Macro_F1score


############## The MobileNet Model
Mobile_Net = tf.keras.applications.mobilenet.MobileNet(include_top=False, weights='imagenet', input_shape=(SIZE, SIZE, 3))

for layer in Mobile_Net.layers:
	layer.trainable = False
Mobile_Net.summary() 

feature_extractor=Mobile_Net.predict(train_generator)
features = feature_extractor.reshape(feature_extractor.shape[0], -1)

RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)
RF_model.fit(features, y_train)

X_val_feature = Mobile_Net.predict(val_generator)
X_val_features = X_val_feature.reshape(X_val_feature.shape[0], -1)

prediction_RF =RF_model.predict(X_val_features)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(y_val, prediction_RF))
print ("F1_score (COVID) = ", metrics.f1_score(y_val, prediction_RF, pos_label = 0))
print ("F1_score (NON-COVID) = ", metrics.f1_score(y_val, prediction_RF,pos_label = 1))
print("Average Precision", average_precision_score(y_val, prediction_RF))
print("Average Recall", recall_score(y_val, prediction_RF))
print ("Macro F1_score = ", metrics.f1_score(y_val, prediction_RF, average='macro'))