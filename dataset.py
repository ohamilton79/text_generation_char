import numpy as np
import re
from random import randint

def summariseDataset(filename):
    #Read the dataset file
    rawText = open(filename, 'r', encoding='utf-8').read()
    #Convert text to lowercase
    rawText = rawText.lower()
    #Remove line breaks and indentation
    rawText = rawText.replace('\n', ' ')
    rawText = rawText.replace('\t', '')
    #Remove any double spaces created by removing line breaks / indentation
    rawText = rawText.replace('  ', ' ')
    #Create a mapping from characters to integers
    chars = sorted(list(set(rawText)))
    charToInt = dict((c, i) for i, c in enumerate(chars))
    #Get the total number of characters, and number of unique characters
    nChars = len(rawText)
    nVocab = len(chars)
    
    return rawText, charToInt, nChars, nVocab

def getTrainingData(rawText, maxSeqLength, seedLength, maskValue, charToInt, stride, nChars, nVocab):
    sentences = []
    nextChars = []
    seqLength = randint(seedLength, maxSeqLength)
    #Sample character sequences to use as training data
    i = 0
    while i < nChars - seqLength:
        seqIn = rawText[i:i + seqLength]
        seqOut = rawText[i + seqLength]
        sentences.append([charToInt[char] for char in seqIn])
        nextChars.append(charToInt[seqOut])

        #Get next character sequence
        i += 5
        #Get next sequence length
        seqLength = randint(seedLength, maxSeqLength)
        #print(len(seqIn))

    #print("Sentence length: {}".format(len(sentences)))
    X = np.zeros((len(sentences), maxSeqLength, nVocab), dtype=np.float32)
    Y = np.zeros((len(sentences), nVocab), dtype=np.float32)

    #One-hot encode the network's inputs and outputs
    for i, sentence in enumerate(sentences):
        for j in range(maxSeqLength):
            if j < len(sentence):
                char = sentence[j]
                X[i, j, char] = 1.0
            else:
                X[i, j] = np.full((nVocab), maskValue)

        Y[i, nextChars[i]] = 1.0
        
    return X, Y

def getTestData(rawText, seedLength, maxSeqLength, maskValue, charToInt, nChars, nVocab):
    #print(rawText[0:1000])
    #Split the corpus into a list of sentences, ignoring periods for title abbreviations
    sentences = re.sub("(?<!(.{1}\smr|\smrs|.{2}\sm|\smme|mlle))\\.\s", ".//", rawText).split("//")
    #Keep only the sentences of at least the required length
    sentences = [sentence for sentence in sentences if len(sentence) > seedLength]
    #print(sentences[0:5])
    #Pick a random sentence index to choose
    i = np.random.randint(0, len(sentences)-1)
    #print(sentences[i])
    #Use the random number to get a random seed
    seqIn = sentences[i][0:seedLength]
    seqOut = sentences[i][seedLength]
    sentence = [charToInt[char] for char in seqIn]

    pattern = np.zeros((1, seedLength, nVocab), dtype=np.float32)
    #One hot encode the network inputs
    for j, char in enumerate(sentence):
        pattern[0, j, char] = 1.0
    #Pad the pattern so it matches the maximum sequence length
    paddedPattern = np.pad(pattern, ((0, 0), (0, maxSeqLength - seedLength), (0, 0)), constant_values=maskValue)
    return paddedPattern
