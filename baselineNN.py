import numpy as np
from array import array
import math
import matplotlib.pyplot as plt
import pdb
from numpy import round
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from keras import backend as T
from sklearn.preprocessing import StandardScaler

fileDir = "/scratch/murtazas/CS231N-data/inputArraysVSmallCR/"
#   /scratch/murtazas/CS231N-data/inputArraysVSmallCR
#   /nfs/slac/g/atlas/u02/murtazas/PileUPJetID/inputArraysVSmallCR/
plotDir = "/home/murtazas/CS231N-PileUp-Jet-ID/AllPlots/"
#   /home/murtazas/CS231N-PileUp-Jet-ID/AllPlots
#   /nfs/slac/g/atlas/u02/murtazas/PileUPJetID/Plots/

train_Rpt  = np.load(fileDir + "train_jet_Rpt_j0_EM.npy")
CV_Rpt     = np.load(fileDir + "CV_jet_Rpt_j0_EM.npy")
test_Rpt     = np.load(fileDir + "test_jet_Rpt_j0_EM.npy")
train_j0pt = np.load(fileDir + "train_recopts_j0_EM.npy") 
CV_j0pt    = np.load(fileDir + "CV_recopts_j0_EM.npy")
test_j0pt    = np.load(fileDir + "test_recopts_j0_EM.npy")  
train_isPU = np.load(fileDir + "train_isPU_j0_EM.npy") 
CV_isPU    = np.load(fileDir + "CV_isPU_j0_EM.npy")
test_isPU    = np.load(fileDir + "test_isPU_j0_EM.npy")
train_rwi  = np.load(fileDir + "train_revisedWeights_j0_EM.npy")
CV_rwi     = np.load(fileDir + "CV_revisedWeights_j0_EM.npy")
test_rwi     = np.load(fileDir + "test_revisedWeights_j0_EM.npy")

train_y = train_isPU.astype(int)
CV_y    = CV_isPU.astype(int)
test_y  = test_isPU.astype(int)

n_train = train_Rpt.size
n_CV    = CV_Rpt.size
n_test  = test_Rpt.size

train_X = np.column_stack((train_j0pt,train_Rpt))
CV_X    = np.column_stack((CV_j0pt,CV_Rpt))
test_X  = np.column_stack((test_j0pt,test_Rpt))
whole_X = np.concatenate((train_X, CV_X), axis=0)
whole_X = np.concatenate((whole_X, test_X), axis=0)
#scale   = StandardScaler()
#train_X = scale.fit_transform(train_X)
#CV_X    = scale.transform(CV_X)
#test_X  = scale.transform(test_X)
#whole_X = scale.transform(whole_X)

seed = 7
np.random.seed(seed)

# create model
model = Sequential()
model.add(Dense(5, input_shape=(2,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
print("[INFO] compiling model...")
adad = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#rmsprop
#adam
# RMSPE
# mean_squared_error
# mean_absolute_percentage_error
# mean_absolute_error
# sgd = SGD(lr=0.01)
# model.compile(loss="categorical_crossentropy", optimizer=sgd,
# 	metrics=["accuracy"])
history = model.fit(train_X, train_y, epochs=30, batch_size=100,verbose =2,sample_weight=train_rwi,validation_data=(CV_X, CV_y,CV_rwi))
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plotDir + "learningCurve_Loss_baselineNN.jpg")
plt.close()

plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plotDir + "learningCurve_Accuracy_baselineNN.jpg")

whole_class_predictions = model.predict(whole_X, batch_size=100, verbose=1)
class_predictions = model.predict(CV_X, batch_size=100, verbose=1)
#scores = model.evaluate(CV_X, CV_y, batch_size=1000)
np.save(fileDir + "classPredictions_whole_baselineNN.npy", whole_class_predictions)
np.save(fileDir + "classPredictions_CV_baselineNN.npy", class_predictions)