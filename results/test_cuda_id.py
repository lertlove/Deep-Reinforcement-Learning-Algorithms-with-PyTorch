import torch
import os

cuda_device = int(os.getenv('CUDA_DEVICE'))
torch.cuda.set_device(cuda_device) 
print(torch.cuda.current_device())


