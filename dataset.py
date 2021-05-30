import numpy as np
import re

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

def getTrainingData(rawText, seqLength, charToInt, stride, nChars, nVocab):
    sentences = []
    nextChars = []
    #Sample character sequences to use as training data
    for i in range(0, nChars - seqLength, stride):
        seqIn = rawText[i:i + seqLength]
        seqOut = rawText[i + seqLength]
        sentences.append([charToInt[char] for char in seqIn])
        nextChars.append(charToInt[seqOut])

    X = np.zeros((len(sentences), seqLength, nVocab), dtype=np.float32)
    Y = np.zeros((len(sentences), nVocab), dtype=np.float32)

    #One-hot encode the network's inputs and outputs
    for i, sentence in enumerate(sentences):
        for j, char in enumerate(sentence):
            X[i, j, char] = 1.0

        Y[i, nextChars[i]] = 1.0
        
    return X, Y

def getTestData(rawText, seqLength, charToInt, nChars, nVocab):
    #print(rawText[0:1000])
    #Split the corpus into a list of sentences, ignoring periods for title abbreviations
    sentences = re.sub("(?<!(.{1}\smr|\smrs|.{2}\sm|\smme|mlle))\\.\s", ".//", rawText).split("//")
    #Keep only the sentences of at least the required length
    sentences = [sentence for sentence in sentences if len(sentence) > seqLength]
    #print(sentences[0:5])
    #Pick a random sentence index to choose
    i = np.random.randint(0, len(sentences)-1)
    #print(sentences[i])
    #Use the random number to get a random seed
    seqIn = sentences[i][0:seqLength]
    seqOut = sentences[i][seqLength]
    sentence = [charToInt[char] for char in seqIn]

    pattern = np.zeros((seqLength, nVocab), dtype=np.float32)
    #One hot encode the network inputs
    for j, char in enumerate(sentence):
        pattern[j, char] = 1.0
        
    return pattern
