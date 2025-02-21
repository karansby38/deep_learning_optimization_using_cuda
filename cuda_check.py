import torch
print(torch.cuda.device_count())  # Number of available GPUs
print(torch.cuda.get_device_name(0))  # Name of GPU 0
if torch.cuda.device_count() > 1:
    print(torch.cuda.get_device_name(1))  # Name of GPU 1
