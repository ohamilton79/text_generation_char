import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Masking
from keras.optimizers import Adam

def loadModel(seqLength, nVocab, maskValue):
    model = Sequential()
    model.add(Masking(mask_value=maskValue, input_shape=(seqLength, nVocab)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(128))
    model.add(Dense(nVocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))
    
    return model
