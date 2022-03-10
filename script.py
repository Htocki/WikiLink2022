# -*- coding: utf-8 -*-
## Задание
# Реализовать и обучить нейронную сеть, которая определяет к какому
# оператору (A1, MTC, life, Beltelecom) относится указанный номер.

import re, sys
import numpy as np

from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def ReadTextFrom(file_name):
  with open(file_name, 'r', encoding='utf-8') as file:
    text = file.read()
    text = text.replace('\ufeff', '') # Удаление спецсимвола кодировки
    text = re.sub('\n', ' ', text) # Замена всех переходов на новую
                                   # строку пробелами
  return text

def CheckCommandLineArguments():
  if len(sys.argv) < 2:
    print('Ошибка: Некорректное число аргументов командной строки')
    print('Синтаксис: python script.py train.txt validation.txt')
    exit(1)
  return sys.argv[1], sys.argv[2]

# Разбивает входной текст на слова по пробелам и разделяет их на две 
# категории (номера телефонов, именования операторов). Преобразует данные из
# категорий в два тензора соотвественно входной и выходной.
def ConvertToTensor(text, max_words_count, input_words):
  tokenizer = Tokenizer(
  # Максимальное количество слов (символов), которое вернет Tokenizer
  # (если элементов будет больше, то останутся наиболее повторяющиеся)
  num_words=max_words_count,
  # Настройка фильтра
  filters='', # Фильтрация отсутствует
  # Определение символа-разделителя
  split=' ', # Текст будет разделен на слова по пробелам
  # Определение типа разбивки
  char_level=False) # Разбивка по словам
  # Обновление словаря на основании указанного текста
  tokenizer.fit_on_texts([text])
  # Преобразование текста в последовательность целых чисел. Каждому
  # уникальному слову соответствует число
  data = tokenizer.texts_to_sequences([text])
  # Конвертирует получившеюся последовательность чисел в бинарную матрицу
  res = to_categorical(data[0], num_classes=max_words_count)
  # Разделение
  n = res.shape[0] - input_words
  input_tensor = np.array([res[i:i + input_words, :] for i in range(n)])
  output_tensor = res[input_words:]
  return input_tensor, output_tensor

## Обрабока аргументов командной строки
train_file_name, validation_file_name = CheckCommandLineArguments()

## Определение обучающей выборки
text = ReadTextFrom(train_file_name)
# Лимит сохраненных в словаре класса Tokenizer слов. Слова подбираются по
# частоте их появления в тексте
max_words_count = 250 # В исходном тексте чуть менее 250 слов
# Что это?
input_words = 1 # Почему 1?
# Необходимо разбить на две функции !
input_tensor, output_tensor = ConvertToTensor(text, max_words_count, input_words)

## Определение структуры нейронной сети
# Используется модель многослойного перцептрона с последовательно
# расположенными слоями
model = Sequential()
# Добавление входного слоя
model.add(Input((input_words, max_words_count)))
# Добавление простого рекуррентного слоя
model.add(SimpleRNN(units=128, activation='tanh'))
# Добавление вероятностного слоя
model.add(Dense(max_words_count, activation='softmax'))
model.summary()

## Компиляция нейронной сети
# При вызове compile() автоматически задаются начальные весовые кэффициенты
model.compile(
  # Функция потерь
  loss='categorical_crossentropy',
  # Алгоритм оптимизации алгоритма градиентного спуска
  optimizer='adam', # Шаг сходимости 0.001 (по умолчанию)
  # Метрики для мониторинга обучения
  metrics=['accuracy']) # Вычисляет как часто прогнозы соответствуют меткам

## Обучение нейронной сети
history = model.fit(
  # Входные данные
  input_tensor,
  # Ожидаемые данные
  output_tensor,
  # Количество эпох
  epochs=50,
  # Режим детализации
  verbose=1) # Отображается индикатор выполнения

## Передача нейронной сети данных на вход (с выводом результата)
#print(model.predict(['''значение''']))

## Возможно понадобятся
# Отображение всех весовых коэффициентов нейронной сети
#print(model.get_weights())
# Построение графика функции потерь
#ptl.plot(history.history['loss'])
#plt.grid(True)
#plt.show()
