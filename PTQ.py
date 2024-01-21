########## Deep learning quantization using Tensorflow and Pytorch. All code is compatible with GPU environment set in Google colab.
# Reference will be added later.
########## Post-training quantization. Tensorflow 
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
tt=[]
input_shape=(3,32,32) #channels_first format
model=keras.applications.VGG16(weights=None,input_shape= input_shape,classes=10,include_top =True)
#def single_inf(model,x):
for i in[5,10,50,100]:
  st=time.time()
  out =model.predict(x_test[:i],verbose = 0)#np.expand_dims(x_test[i],axis=0))#
  t=time.time()-st
  #tt+=[t] #inference latency mean
  print(f'{t:.4f}', f'{t/i:.4f}', f'{i/t:.4f}', '\n')#,np.mean(tt))#
  #print(f'{i/t:.4f}')
#GPU results. T4
#Latency:  [0.10116028785705566, 0.06974124908447266, 0.05391836166381836, 0.05707049369812012, 0.05506753921508789]  0.06739158630371093
#pipelined latency[5,10,50,100]: 0.0149,0.0072,0.0022,0.0009 to compare with latency. Indication of parallelism benefits. DECREASING
#throughput (1/batch latency) [5,10,50,100]: 67.0876 ,139.3109,450.0317 ,1129.6997
#CPU results:
#Latency: [0.3026766777038574, 0.06519889831542969, 0.06671833992004395, 0.06192755699157715, 0.07696533203125] 0.11469736099243164
#pipelined latency[5,10,50,100]:0.0186,0.0125,0.0085,0.0131. NO MONOTONE VARIATION
#throughput (1/batch latency) [5,10,50,100]: 47.7759,78.4903,122.0628,127.4688,
converter = tf.lite.TFLiteConverter.from_saved_model('my_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_quant_model)
  
model1=interpreter = tf.lite.Interpreter(model_path="./model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
tt=[]
for i in range(200):
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
  interpreter.set_tensor(input_details[0]['index'], input_data)
  st=time.time()
  interpreter.invoke()
  t=time.time()-st
  tt+=[t]
print(np.mean(tt))
#default 0.019141
converter = tf.compat.v1.lite.TFLiteConverter.from_keras_model_file("my_model/saved_model.pb")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def fake_representative_data_gen():
  for _ in range(2):
    fake_image = np.expand_dims(x_test[0],axis=0).astype(np.float32)#np.random.random((1,32,32,3)).astype(np.float32)
    yield [fake_image]
converter.representative_dataset = fake_representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model_quant = converter.convert()
with open('model3.tflite', 'wb') as f:
  f.write(tflite_model_quant)

