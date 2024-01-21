########## Deep learning quantization using Tensorflow and Pytorch. All code is compatible with GPU environment set in Google colab.
# Reference will be added later.
########## Quantization aware training. Tensorflow 
import numpy as np
from tensorflow.keras import datasets, layers, models, losses, Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization, Input
import time
from tensorflow import keras
import tensorflow as tf

###############" Loading MNist"
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
assert x_train.shape == (50000, 32, 32, 3)
assert x_test.shape == (10000, 32, 32, 3)
assert y_train.shape == (50000, 1)
assert y_test.shape == (10000, 1)
input_shape=(3,32,32) #channels_first format
model=keras.applications.VGG16(weights=None,input_shape= input_shape,classes=10,include_top =True)
model.summary()
#!pip install -q tensorflow-model-optimization
import tensorflow_model_optimization as tfmot
input_shape=x_train.shape[1:4]
steps_per_epoch=20
model=keras.applications.VGG16(weights=None,input_shape= input_shape,classes=10,include_top =True)
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
model.fit(
  x_train,
  y_train,
  epochs=20,steps_per_epoch=steps_per_epoch, batch_size=2)
def apply_quantization_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Dense):
    return tfmot.quantization.keras.quantize_annotate_layer(layer)
  return layer
# Use `tf.keras.models.clone_model` to apply `apply_quantization_to_dense`
# to the layers of the model.
base_model=model
annotated_model = tf.keras.models.clone_model(
    base_model,
    clone_function=apply_quantization_to_dense,
)
# `quantize_apply` actually makes the model quantization aware.
quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
#quant_aware_model.summary()
## or annotated_model = tf.keras.Sequential([ tfmot.quantization.keras.quantize_annotate_layer(tf.keras.layers.Dense(20, input_shape=input_shape)),tf.keras.layers.Flatten()])
'''Typically you train the model here.'''
quant_aware_model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])
quant_aware_model.fit(
  x_train,
  y_train,
  epochs=1,steps_per_epoch=steps_per_epoch, batch_size=2) #,validation_data=(x_test, y_test))
####### Save or checkpoint the model.
# `quantize_scope` is needed for deserializing HDF5 models.
#_, keras_model_file = tempfile.mkstemp('.h5')
#quant_aware_model.save(keras_model_file)
#with tfmot.quantization.keras.quantize_scope():
#  loaded_model = tf.keras.models.load_model(keras_model_file)
####### Deployment of model quantization post training
#converter = tf.lite.TFLiteConverter.from_keras_model(quant_aware_model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#quantized_tflite_model = converter.convert()