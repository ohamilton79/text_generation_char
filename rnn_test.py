import sys
import re
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

from model import loadModel
from dataset import summariseDataset, getTestData

def sample(preds, temperature=1.0):
    """Use the temperature to add variety to the predicted
    sentences
    """
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def performTest(weightsFilename):
    #Constants
    filename = "corpus.txt"
    seqLength = 40
    temperature = 0.5

    #Allow memory to grow on GPU
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    #Load ASCII text and return a mapping from characters to integers,
    #and retrieve the number of total characters and unique characters
    rawText, charToInt, nChars, nVocab = summariseDataset(filename)
    #Create a mapping from integers to characters
    intToChar = dict((i, c) for c, i in charToInt.items())

    #Get a random piece of test data
    pattern = getTestData(rawText, seqLength, charToInt, nChars, nVocab)

    #Get the RNN model
    model = loadModel(seqLength, nVocab)

    #Load the network weights
    filename = weightsFilename
    #"weights-improvement-35-1.1789-bigger.hdf5"#"weights-improvement-50-1.0037-bigger.hdf5"
    model.load_weights(filename)

    result = ''.join([intToChar[np.argmax(value)] for value in pattern])

    #Generate characters using the seed until a full stop is predicted
    currentChar = None
    while currentChar != ".":
        #Predict the next character using the RNN
        prediction = model.predict(np.array([pattern]), verbose=0)[0]
        index = sample(prediction, temperature)
        currentChar = intToChar[index]
        #One-hot encode the outputted character
        oneHot = np.zeros((nVocab))
        oneHot[index] = 1.0
        #Update seed using newly generated character
        pattern[0:seqLength-1] = pattern[1:seqLength]
        pattern[seqLength - 1] = oneHot
        result += currentChar

    print("Generated text:\n{}".format(result))
