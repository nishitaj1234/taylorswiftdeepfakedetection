import pandas as pd
import numpy as np
import sklearn.model_selection
import keras

from keras.models import load_model

originalfile=pd.read_csv("Original/taylor_original2.txt")
#originalfile2=pd.read_csv("Original/taylor_original1.txt")
originalfile3=pd.read_csv("Original/taylor_original3.txt")

deepfakefile=pd.read_csv("Deepfakes/taylorfake3.txt")

originaldata=np.array(originalfile.drop("class",1))
originallabels=np.array(originalfile["class"])
#originaldata2=np.array(originalfile2.drop("class",1))
#originallabels2=np.array(originalfile2["class"])
originaldata3=np.array(originalfile3.drop("class",1))
originallabels3=np.array(originalfile3["class"])


deepfakedata=np.array(deepfakefile.drop("class",1))
deepfakelabels=np.array(deepfakefile["class"])
print(len(deepfakedata))




'''print(originallabels)
print(deepfakelabels)
print(deepfakelabels2)'''
alloriginallabels=np.concatenate([originallabels,originallabels3])
alloriginaldata=np.concatenate([originaldata,originaldata3])
print(len(alloriginaldata))

'''
alldeepfakelabels=np.concatenate([deepfakelabels,deepfakelabels2])
alldeepfakedata=np.concatenate([deepfakedata,deepfakedata2])'''

data=np.concatenate([deepfakedata,alloriginaldata])
labels=np.concatenate([deepfakelabels,alloriginallabels])
print(len(deepfakelabels),len(alloriginallabels))




#generate another deepfake based on given video. include in code, variable called all data and all labels
Xtraindata, Xtestdata, Ytraindata, Ytestdata=sklearn.model_selection.train_test_split(data,labels,test_size=0.2)


                        
                        
model=keras.Sequential([#keras.layers.Flatten(input_shape=(137,)),
                        keras.layers.Dense(512,activation='relu'),
                        keras.layers.Dense(256,activation='relu'),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(2,activation='softmax')])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])

model.fit(Xtraindata,Ytraindata,epochs=50)

model.save('deepfaketaylor.h5')




