import os
import math
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
# from MyDataset import getVariable
# from My_test import test_RMSE
import time
from torch import nn
from torch.nn import functional as F
from models.afnonet import AFNONet
from torch.nn.parallel import DataParallel
from hrrr_dataset import HRRRDataset

from tools import save_tools
from loss.l2 import RelativeL2Loss
import datetime
from models.afno_attention_parallel import AFNOAttnParallelNet
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
#from eval import eval
import h5py



batch_size = 16
# batch_size = 4
# 获取数据
# 右下
father_path = "/nfs/samba/数据聚变/气象数据/hrrr_440_24var_rb"

# set dataset path
TRAIN_FILE_PATH =   father_path+"/train"
VALID_FILE_PATH =   father_path+"/valid"
DATA_MEAN_PATH =    father_path+"/stat_rt/mean_crop.npy"
DATA_STD_PATH =     father_path+"/stat_rt/std_crop.npy"
DATA_TIME_MEAN_PATH =    father_path+"/stat_rt/time_mean_crop.npy"


# set training hyper-parameters
input_keys = ("input",)
output_keys = ("output",)
IMG_H, IMG_W = 440, 408  # for HRRR dataset croped data
EPOCHS = 30 
# FourCastNet HRRR Crop use 24 atmospheric variable，their index in the dataset is from 0 to 23.
# The variable name is 'z50', 'z500', 'z850', 'z1000', 't50', 't500', 't850', 'z1000',
# 's50', 's500', 's850', 's1000', 'u50', 'u500', 'u850', 'u1000', 'v50', 'v500', 'v850', 'v1000',
# 'mslp', 'u10', 'v10', 't2m'.
VARS_CHANNEL = list(range(24))
# set output directory
OUTPUT_DIR = "./output/rb/pretrain" 
# initialize logger


print("右下")
input_timestamps = 1
label_timestamps = 1
dataset = HRRRDataset(VALID_FILE_PATH, input_keys, output_keys,None ,input_timestamps,label_timestamps,VARS_CHANNEL, training=True, stride=1)
# dataloader_ = DataLoader(dataset, batch_size=batch_size, shuffle=True)

