########## Deep learning quantization using Tensorflow and Pytorch. All code is compatible with GPU environment set in Google colab.
# Reference will be added later.
########## Post-training quantization. Pytorch
#requirements for Colab dev APi
'''!pip install torch-summary
!pip install torch
!pip install -q torchvision
!pip install tensorrt
!export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.10/dist-packages/tensorrt
!pip install torch_tensorrt'''
from torchvision.io import read_image
from torchvision.models import vgg16
from torchsummary import summary
import torch, torchvision
import torchvision.transforms as transforms
import torch_tensorrt as torchtrt
import time
import numpy as np
testing_dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    ),
)

model = vgg16(num_classes= 1000,init_weights=False)
#summary(model, input_size=(3,32,32))

testing_dataloader = torch.utils.data.DataLoader(
    testing_dataset, batch_size=1, shuffle=False, num_workers=1)
calibrator = torchtrt.ptq.DataLoaderCalibrator(
    testing_dataloader,
    cache_file="./calibration.cache",
    use_cache=False,
    algo_type=torchtrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
    device=torch.device("cuda:0"),)
#convert model to eval mode
model.eval()
trt_mod = torchtrt.compile(model, inputs=[torchtrt.Input((1, 3, 32, 32))],
                        enabled_precisions={torch.float, torch.half, torch.int8},
                        calibrator=calibrator,
                        device={"device_type": torchtrt.DeviceType.GPU,
                              "gpu_id": 0,
                              "dla_core": 0,
                              "allow_gpu_fallback": False,
                              "disable_tf32": False })
torch.jit.save(trt_mod, "trt_torchscript_module.ts")
torch.save(model, 'original_model.pt')
torch.jit.save(trt_mod, "quantizedmodel.pt")
tt=[]
for i in range(5):
  st=time.time()
  result = trt_mod(testing_dataset[i][0])
  t=time.time()
  t=t-st
  tt+=[t]
print(tt, np.mean(tt))
size=32
tt=[]
for i in range(5):
  image=torch.rand((1,3,size,size))
  image=image.cuda()
  model=model.to(torch.device("cuda:0"))
  st=time.time()
  out=model(image)
  t=time.time()-st
  tt+=[t]
print(tt, np.mean(tt))