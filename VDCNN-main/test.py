import tensorflow as tf

# 检查TensorFlow版本
print("TensorFlow version:", tf.__version__)

# 检查是否有GPU可用
if tf.test.gpu_device_name():
    print("Default GPU device:", tf.test.gpu_device_name())
else:
    print("No GPU available")

# 显示GPU的详细信息
from tensorflow.python.client import device_lib

local_device_protos = device_lib.list_local_devices()
gpu_devices = [x for x in local_device_protos if x.device_type == 'GPU']
for device in gpu_devices:
    print(device)