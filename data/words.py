from time import time

import tensorflow as tf
import string

import numpy as np
import pandas as pd
from keras.callbacks import TensorBoard
from keras.layers import Dense, RNN, SimpleRNN, GRU
from keras.layers import LSTM
from keras.layers.core import Activation, Dropout
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

df = pd.read_csv('/home/piotr/Workspace/Repositories/cs230-code-examples/tensorflow/nlp/data/yuh/IN.TXT')
df.name = df.name.str.lower()
df['gender'] = pd.factorize(df['gender'])[0]
vocab = {c: i for i, c in enumerate(string.ascii_lowercase)}
vocab['PAD'] = 26


def word2array(word, vocab, length):
    return np.array([char2vec(char, vocab) for char in word] + [char2vec('PAD', vocab)] * (length - len(word)))


def char2vec(char, vocab):
    vec = np.zeros(len(vocab))
    vec[vocab[char]] = 1
    return vec


def wordlist2array(word_list, vocab, length):
    return np.array([word2array(word, vocab, length) for word in word_list])


def get_labels(genders):
    return np.array([[1, 0] if gender == 0 else [0, 1] for gender in genders])


word_list = df['name'].tolist()
genders = get_labels(df['gender'].tolist())
data = wordlist2array(word_list, vocab, 11)
print(data.shape)
X_train, X_test, y_train, y_test = train_test_split(data, genders, test_size=0.2, random_state=42)


model = Sequential()
model.add(SimpleRNN(100, return_sequences=False, input_shape=(11, len(vocab))))
# model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=False))
# model.add(Dropout(0.2))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 500
tensorboard = TensorBoard(log_dir="logs/lstm3")

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=10, validation_data=(X_test, y_test), callbacks=[tensorboard])
result = model.predict(X_test)
print(result)
print(y_test)
print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(result, axis=1)))
# X = X_test
# Y = y_test
# np.savez('test.npz', name1=X, name2=Y)
# # np.savez('mat.npz', name1=X_test, name2=y_test)
