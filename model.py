import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizers import Adam

def loadModel(seqLength, nVocab):
    model = Sequential()
    model.add(LSTM(64, input_shape=(seqLength, nVocab), return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(nVocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    
    return model
