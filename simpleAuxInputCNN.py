import numpy as np
from array import array
import math
import matplotlib.pyplot as plt
import pdb
from numpy import round
import keras
from keras.models import Sequential
#from keras.layers import Dense, Dropout, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Conv2D
#from keras.layers import Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.constraints import maxnorm
from keras import backend as T
from sklearn.preprocessing import StandardScaler
from keras.layers.normalization import BatchNormalization

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
train_image = np.load(fileDir + "train_pixel_image_clus_trks_j0_EM.npy")
CV_image   = np.load(fileDir + "CV_pixel_image_clus_trks_j0_EM.npy")
test_image = np.load(fileDir + "test_pixel_image_clus_trks_j0_EM.npy")

train_y = train_isPU.astype(int)
CV_y    = CV_isPU.astype(int)
test_y  = test_isPU.astype(int)

n_train = train_Rpt.size
n_CV    = CV_Rpt.size
n_test  = test_Rpt.size

train_X = train_image
CV_X    = CV_image
test_X  = test_image
whole_X = np.concatenate((train_X, CV_X), axis=0)
whole_X = np.concatenate((whole_X, test_X), axis=0)

seed = 7
np.random.seed(seed)

# create model
model = Sequential()

model.add(BatchNormalization(axis=1, input_shape=(3, 10, 10)))
model.add(Conv2D(10, 10, strides=(1, 1), padding='valid', activation='relu',data_format='channels_first'))
#model.add(BatchNormalization(axis=1))
#model.add(Conv2D(8, 3, strides=(1, 1), padding='valid', activation='relu',data_format='channels_first'))
#model.add(BatchNormalization(axis=1))
#model.add(Conv2D(16, 4, strides=(1, 1), padding='valid', activation='relu',data_format='channels_first'))
model.add(Flatten())
#model.add(Dense(5))
#model.add(Activation('sigmoid'))
#model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(1))
model.add(Activation('sigmoid'))

print("[INFO] compiling model...")
#model.compile(loss=RMSPE2, optimizer='adam')
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
history = model.fit(train_X, train_y, epochs=50, batch_size=100,verbose =2,sample_weight=train_rwi,validation_data=(CV_X, CV_y,CV_rwi))
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plotDir + "learningCurve_Loss_simpleCNN.jpg")
plt.close()

plt.figure(2)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(plotDir + "learningCurve_Accuracy_simpleCNN.jpg")

whole_class_predictions = model.predict(whole_X, batch_size=100, verbose=1)
class_predictions = model.predict(CV_X, batch_size=100, verbose=1)
#scores = model.evaluate(CV_X, CV_y, batch_size=1000)
np.save(fileDir + "classPredictions_whole_simpleCNN.npy", whole_class_predictions)
np.save(fileDir + "classPredictions_CV_simpleCNN.npy", class_predictions)
