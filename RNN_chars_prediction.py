﻿import numpy as np
import re
from keras.layers import Dense, SimpleRNN, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer

with open('texts/train_data_true.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')
    text = re.sub(r'[^А-я ]', '', text)

num_characters = 34
tokenizer = Tokenizer(num_words=num_characters, char_level=True)
tokenizer.fit_on_texts([text])

print(tokenizer.word_index)

inp_chars = 6
data = tokenizer.texts_to_matrix(text)
n = data.shape[0] - inp_chars

X = np.array([data[i:i + inp_chars, :] for i in range(n)])
Y = data[inp_chars:]

print(data.shape)

model = Sequential()
model.add(Input((inp_chars, num_characters)))
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(num_characters, activation='softmax'))
model.summary()

model.compile(
    loss='categorical_crossentropy',
    metrics=['accuracy'],
    optimizer='adam'
)

history = model.fit(X, Y, batch_size=32, epochs=100)


def buildPhrase(inp_str, str_len=50):
    for i in range(str_len):
        x = []

        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j]))

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters)

        pred = model.predict(inp)
        d = tokenizer.index_word[pred.argmax(axis=1)[0]]

        inp_str += d

    return inp_str


res = buildPhrase("утренн")

print(res)
