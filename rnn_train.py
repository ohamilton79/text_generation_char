# LSTM Network to generate text inspired by Oliver Twist by Charles Dickens
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, Callback
from keras.utils import np_utils
import re

from model import loadModel
from dataset import summariseDataset, getTrainingData
from rnn_test import performTest

class ValidationCallback(Callback):
    #At the end of each epoch, test the network
    def on_epoch_end(self, epoch, logs=None):
        performTest("weights/weights-{}.hdf5".format(epoch+1))

#Constants
filename = "corpus.txt"
seqLength = 40
stride = 3

#Allow memory to grow on GPU
gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

#Load ASCII text and return a mapping from characters to integers,
#and retrieve the number of total characters and unique characters
rawText, charToInt, nChars, nVocab = summariseDataset(filename)

#Get the training data from the raw text
X, Y = getTrainingData(rawText, seqLength, charToInt, stride, nChars, nVocab)

#Get the RNN model and output a summary
model = loadModel(seqLength, nVocab)
model.summary()

#Define the checkpoint to save the best weights
filepath="weights/weights-{epoch:d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, mode='min')

#Fit the model using the training data
model.fit(X, Y, epochs=300, batch_size=32, callbacks=[checkpoint, ValidationCallback()], shuffle=True)
