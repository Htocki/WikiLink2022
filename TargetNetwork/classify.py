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

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
  standardize='lower_and_strip_punctuation',
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
  layers.Dense(4)])

model.summary()

model.compile(loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'])

epochs = 10
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs)

loss, accuracy = model.evaluate(test_ds)

print("Model loss: ", loss)
print("Model accuracy: ", accuracy)

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
  loss=losses.SparseCategoricalCrossentropy(from_logits=False),
  optimizer="adam",
  metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print('Export model loss', loss)
print('Export model accaracy: ', accuracy)

#examples = [
  # normal
#  '',
  # anomaly
#  '',
  # normal
#  '',
  # normal
#  '',
  # anomaly
#  '']

#print('Result:\n', export_model.predict(examples))
print('Class name 0: ', raw_test_ds.class_names[0])
print('Class name 1: ', raw_test_ds.class_names[1])
