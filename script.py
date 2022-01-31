# -*- coding: utf-8 -*-
## Задание
# Реализовать и обучить нейронную сеть, которая определяет к какому
# оператору (AOne, MTC, life, Beltelecom) относится указанный номер.

import re, sys
import numpy as np

from tensorflow.keras.layers import Dense, Input, SimpleRNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

def GetTextFromFile(name):
  with open(name, 'r', encoding='utf-8') as file:
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

def ConvertToTensor(text):
  return 0;

def GetOperatorName(number_as_text, tokenizer):
  return 0;

## Обрабока аргументов командной строки
train, validation = CheckCommandLineArguments()

## Определение обучающей выборки
# Загрузка данных из файла
text = GetTextFromFile(train)
# Определение токенизатора
max_words_count = 1000
tokenizer = Tokenizer(
  # Максимальное количество слов (символов), которое вернет Tokenizer
  # (если элементов будет больше, то останутся наиболее повторяющиеся)
  num_words=max_words_count,
  # Настройка фильтра
  filters='', # Фильтрация отсутствует
  # Определение символа-разделителя
  split=' ', # Текст будет разделен на слова по пробелам
  # Тип разбивки: разбивка по словам
  char_level=False) # Разбивка по словам
# Формирование токенов на основе частотности их появления в тексте
tokenizer.fit_on_texts([text]) # зачем квадратные кавычки?

### Мутный кусок: начало 
data = tokenizer.texts_to_sequences([text])
res = to_categorical(data[0], num_classes=max_words_count)
input_words=1
n = res.shape[0] - input_words
x_train = np.array([res[i:i + input_words, :] for i in range(n)])
y_train = res[input_words:]
### Мутный кусок: конец

## Определение структуры нейронной сети
# Определение модели многослойного перцептрона с последовательно
# расположенными слоями
model = Sequential()
# Добавление входного слоя
model.add(Input((input_words, max_words_count)))
# Добавление простого рекуррентного слоя
model.add(SimpleRNN(128, activation='tanh'))
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
  # Входные значения обучающей выборки
  x_train,
  # Ожидаемые выходные значения
  y_train,
  # Что это?
  batch_size=32,
  # Число эпох(?)
  epochs=50,
  # Вывод информации (метрик?) о процессе обучения нейросети
  verbose=True)

## Передача нейронной сети данных на вход (с выводом результата)
#print(model.predict(['''значение''']))

## Возможно понадобятся
# Отображение всех весовых коэффициентов нейронной сети
#print(model.get_weights())
# Построение графика функции потерь
#ptl.plot(history.history['loss'])
#plt.grid(True)
#plt.show()
