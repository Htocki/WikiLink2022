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
  return tf.strings.regex_replace(lowercase,
                                  ',',
                                  ' ')

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

model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

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

acc = history_dict['binary_accuracy']
val_acc = history_dict['val_binary_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation('sigmoid')
])

export_model.compile(
  loss=losses.BinaryCrossentropy(from_logits=False),
  optimizer="adam",
  metrics=['accuracy']
)

# Test it with `raw_test_ds`, which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print('Export model loss', loss)
print('Export model accaracy: ', accuracy)

examples = [ 
  # normal
  '16:53:55,172.16.0.253,s410,00:00:BE:12:5C:E6,http://tv.tvsat.by:30000/BT1_HD/,3232235626,24,3232235521,45158287,0,0,1,30127,1315112,0,16738,,7778802,0,0,1649426037,1649426037,46,0,0,2,0,0,294',
  # anomaly
  '16:52:02,172.28.30.15,s410,00:00:BE:16:0A:76,http://tv.tvsat.by:30000/NTV_B/,3232235623,24,3232235521,581375211,0,0,3197,69045512,2408888987,0,34813421,,6427425,80373959,1,1649417829,1649425922,379426,0,0,202326,1,0,319',
  # normal
  '16:53:28,172.23.85.17,s410,00:00:BE:11:D0:73,nil,3232235627,24,3232235521,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil',
  # normal
  '16:53:27,172.16.0.253,s615,00:00:BE:37:0A:1C,nil,3232235620,24,3232235521,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil,nil',
  # anomaly
  '16:52:16,172.23.146.9,s530,00:00:BE:34:C4:48,http://tv.kottv.by:30002/NatGeoWildHD/,3232235621,24,3232235521,3611056204,0,0,22182,619307491,2594463565,0,163384904,,6331415,82632547,26,1649423584,1649425936,98194,1,1,58799,37889,37889,606']

print('Result:\n', export_model.predict(examples))
print('Class name 0: ', raw_test_ds.class_names[0])
print('Class name 1: ', raw_test_ds.class_names[1])
