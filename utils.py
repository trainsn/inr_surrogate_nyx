import numpy as np
import torch

def ReadScalarBinary(filename):
    data = np.fromfile(filename, dtype=np.float32)
    data = np.log10(data)
    return data

def pytorch_device_config():
    #  configuring device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on the GPU')
    else:
        device = torch.device('cpu')
        print('Running on the CPU')
    return device

def ReadScalarSubdominBinary(filename, startidx:int, numItems:int):
    data = np.fromfile(filename, count= numItems, offset=startidx*4, dtype=np.float32)
    data = np.log10(data)
    return data

if __name__ == '__main__':
    fname = '/fs/ess/PAS0027/nyx_vdl/512/train/0000_0.14903_0.02182_0.83355.bin'
    all_data = ReadScalarBinary(fname)
    sub_data = ReadScalarSubdominBinary(fname, 5, 5)
    print(all_data[:12])
    print(sub_data)
