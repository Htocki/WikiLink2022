import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

dataset_dir = os.path.join("./dataset", '')
train_dir = os.path.join(dataset_dir, 'train')

batch_size = 32
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
  train_dir,
  batch_size=batch_size,
  validation_split=0.2,
  subset='training',
  seed=seed)

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
  train_dir,
  batch_size=batch_size,
  validation_split=0.2,
  subset='validation',
  seed=seed)

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
  train_dir,
  batch_size=batch_size)

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase, ',', ' ')

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
  standardize=custom_standardization,
  max_tokens=max_features,
  output_mode='int',
  output_sequence_length=sequence_length)

train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

embedding_dim = 16

model = tf.keras.Sequential([
  layers.Embedding(max_features + 1, embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(1)])

model.summary()

model.compile(
  loss=losses.BinaryCrossentropy(from_logits=True),
  optimizer='sgd',
  metrics=[
    tf.keras.metrics.AUC(from_logits=True),
    #tf.keras.metrics.FalseNegatives(),
    #tf.keras.metrics.FalsePositives(),
    #tf.keras.metrics.TrueNegatives(),
    #tf.keras.metrics.TruePositives()
])

epochs = 500
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)

loss, auc = model.evaluate(test_ds)

print('Export model loss: ', loss)
print('Export model auc: ', auc)

# Графики
history_dict = history.history
history_dict.keys()

loss = history_dict['loss']
val_loss = history_dict['val_loss']
auc = history_dict['auc']
val_auc = history_dict['val_auc']
epochs = range(1, epochs + 1)

# График потерь
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(axis = 'both')
plt.legend()
plt.show()

# График AUC
plt.plot(epochs, auc, 'r', label='Training auc')
plt.plot(epochs, val_auc, 'b', label='Validation auc')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.grid(axis = 'both')
plt.legend()
plt.show()

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
  loss=losses.BinaryCrossentropy(from_logits=False),
  optimizer="sgd",
  metrics=tf.keras.metrics.AUC())

loss, auc = export_model.evaluate(raw_test_ds)
print('')
print('Export model loss: ', loss)
print('Export model auc: ', auc)
print('')

print('Class name 0: ', raw_test_ds.class_names[0])
print('Class name 1: ', raw_test_ds.class_names[1])

import sys

def get_line_count(file_name):
  i = 0
  file = open(file_name, 'r')
  for line in file:
    i += 1
  file.close()
  return i

if len(sys.argv) == 3:
  i = 0
  count = get_line_count(sys.argv[1])

  input_f = open(sys.argv[1], 'r')
  output_f = open(sys.argv[2], 'w+')

  for line in input_f:
    print('[', sys.argv[1], '] Line: ', i, '/', count)
    i += 1
    output_f.write('[String] ' +  line)
    output_f.write('[Result] ' +  str(export_model.predict([line])[0]))
    output_f.write('\n\n')
  input_f.close()
  output_f.close()
