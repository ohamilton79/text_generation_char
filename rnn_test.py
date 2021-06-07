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
    maxSeqLength = 100
    seedLength = 40
    temperature = 0.1
    maskValue = -10

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
    pattern = getTestData(rawText, seedLength, maxSeqLength, maskValue, charToInt, nChars, nVocab)
    #print(pattern.shape)
    #Get the RNN model
    model = loadModel(maxSeqLength, nVocab, maskValue)

    #Load the network weights
    filename = weightsFilename
    #"weights-improvement-35-1.1789-bigger.hdf5"#"weights-improvement-50-1.0037-bigger.hdf5"
    model.load_weights(filename)

    result = ''.join([intToChar[np.argmax(pattern[0, i])] for i in range(seedLength)])

    #Generate characters using the seed until the end is reached or the max size is exceeded
    currentChar = None
    charIndex = seedLength
    maxPredictionLen = 300
    while not re.search('(?<!(.{1}\smr|\smrs|.{2}\sm|\smme|mlle))\\.\s', result) and len(result) < maxPredictionLen:
        #Predict the next character using the RNN
        #paddedPattern = np.pad(pattern, ((0, 0), (0, maxSeqLength - len(result)), (0, 0)), constant_values=maskValue)
        prediction = model.predict(pattern, verbose=0)[0]
        index = sample(prediction, temperature)
        currentChar = intToChar[index]
        #One-hot encode the outputted character
        oneHot = np.zeros((nVocab))
        oneHot[index] = 1.0
        #print("Initial pattern: {}".format(pattern))
        #Update pattern using newly generated character
        #pattern[0:seqLength-1] = pattern[1:seqLength]
        if charIndex >= maxSeqLength:
            pattern[0, 0:maxSeqLength-1] = pattern[0, 1:maxSeqLength]
            pattern[0, maxSeqLength-1] = oneHot
            #print(pattern)
        else:
            pattern[0, charIndex] = oneHot
        #print(pattern)
        result += currentChar
        charIndex += 1

    print("Generated text:\n{}".format(result))
