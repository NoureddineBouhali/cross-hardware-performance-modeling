# cross-hardware-performance-modeling
The investigation of different signature set selection methodologies on the Nas-Bench-201. The signature set represents the hardware features in a cross hardware.
# Deep learning model quantization using Tensorflow and Pytorch
An example is presented using both deep learning frameworks applied to predefined models from either keras or torchvision model zoos.
## Deep learning quantization
Quantization is a deep learning compression method that impacts memory size and foot print of model inference using smaller floating point/integer bit widths. Models are generally represented using single precision FP32, and quantized to FP6 for GPUs and Int8 in CPUs. Accuracy degradation is expecte from information loss using smaller precision format for *weights and/or activations* of deep learning models.
### Tflite deployment
Tensorflow's framework for embedded systems *"tflite"* hosts libraries for model convpression using native "converter"/"interpreter" classes in python.
### Pytorch deployment
Pytorch offers compatibility with NVIDIA's TensorRT for model compression.
## Post training quantization
Models are compressed after full training on datasets using post training quntization. 
## Quantization aware training
Models are training with inner _quantization/dequantization_ operations throughtout training epochs to adapt to information loss from lower precision formats of weights and/or activations.


