## Задание
# Реализовать и обучить нейронную сеть, которая определяет к какому
# оператору (AOne, MTC, life, Beltelecom) относится указанный номер.

import re, sys
from keras.preprocessing.text import Tokenizer
#from keras.models import Sequential
#from keras.layers import Dense

def GetInformationFromFile(name):
  with open(name, 'r', encoding='utf-8') as file:
    text = file.read()
    words = re.sub(r'[^0-9+\n]', '', text)
    words = words.split()
    labels = re.sub(r'[^a-zA-Z\n]', '', text)
    labels = labels.split()
  return words, labels

def CheckCommandLineArguments():
  if len(sys.argv) < 2:
    print('Ошибка: Некорректное число аргументов командной строки')
    print('Синтаксис: python script.py train.txt validation.txt')
    exit(1)
  return sys.argv[1], sys.argv[2]

## Обрабока аргументов командной строки
train, validation = CheckCommandLineArguments()

## Определение обучающей выборки
# Загрузка данных из файлов
t_words, t_labels = GetInformationFromFile(train)
v_words, v_labels = GetInformationFromFile(validation)
# Создание токенизатора и его настройка на учет только num_words наиболее
# часто используемых слов
tokenizer = Tokenizer(num_words=len(t_words))
# Создание индекса всех слов
tokenizer.fit_on_texts(t_words)

## Определение структуры нейронной сети
# Определение модели многослойного перцептрона с последовательно
# расположенными слоями
#model = Sequential()
# Добавление нейронного слоя полносвязного типа
#model.add(Dence(
  # Число нейронных элементов на слое
#  units='''значение''',
  # Число входов
#  input_shape('''значение''',),
  # Функция активации нейронного элемента
#  activation='''значение'''

## Компиляция нейронной сети
# При вызове compile() автоматически задаются начальные весовые кэффициенты
#model.compile(
  # Функция потерь
#  loss='''значение''',
  # Способ оптимизации алгоритма градиентного спуска
  # 0.1 - шаг сходимости алгоритма (по умолчанию: 0.001)
#  optimizer=keras.optimizers.Adam(0.1))

## Обучение нейронной сети
#history = model.fit(
  # Входные значения обучающей выборки
#  '''значение''',
  # Ожидаемые выходные значения
#  '''значение''',
  # Число эпох
#  epochs='''значение''',
  # Вывод служебной информации о процессе обучения нейросети
#  verbose=False)

## Передача нейронной сети данных на вход (с выводом результата)
#print(model.predict(['''значение''']))

## Возможно понадобятся
# Отображение всех весовых коэффициентов нейронной сети
#print(model.get_weights())
# Построение графика функции потерь
#ptl.plot(history.history['loss'])
#plt.grid(True)
#plt.show()
