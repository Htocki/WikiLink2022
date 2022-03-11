# -*- coding: utf-8 -*-
## Задание
# Реализовать и обучить нейронную сеть, которая определяет к какому
# оператору (A1, MTC, life, Beltelecom) относится указанный номер.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

x_text = (
  '+375-29-123-12-12 '
  '+375-29-293-44-71 '
  '+375-29-333-33-33 '
  '+375-29-532-55-53 '
  '+375-25-342-55-22 '
  '+375-17-123-87-76 '
  '+375-25-434-23-67 '
  '+375-25-234-54-13 '
  '+375-29-154-45-33 '
  '+375-29-876-32-43')

y_train = np.array([1, 2, 1, 2, 3, 4, 3, 3, 1, 2])
operators_count = 4

max_words_count = np.size(y_train)
input_words = max_words_count

tokenizer = Tokenizer(
  num_words=max_words_count, filters='', split=' ', char_level=False)
tokenizer.fit_on_texts([x_text])
x_numbers = tokenizer.texts_to_sequences([x_text])
x_ohe_matrix = to_categorical(x_numbers[0], num_classes=max_words_count)
x_train = np.array([x_ohe_matrix[i, :] for i in range(x_ohe_matrix.shape[0])])

model = Sequential()
model.add(Input(input_words, max_words_count))
model.add(SimpleRNN(units=128, activation='tanh'))
model.add(Dense(units=operators_count, activation='softmax'))

model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))

log = model.fit(x_train, y_train, epochs=500, steps_per_epoch=1, verbose=False)

number = ''
while number != 'quit':
  input("Введите номер в форме +XXX-XX-XXX-XX-XX:")
  result = round((model.predict(["+375-29-533-01-10"])))
  if    result == 1:  print('Оператор: A1')
  elif  result == 2:  print('Оператор: MTS')
  elif  result == 3:  print('Оператор: life:)')
  elif  result == 4:  print('Оператор: beltelecom')
  else: print('Ошибка')
