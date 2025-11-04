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
# 并行设置
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
local_rank = int(os.environ['LOCAL_RANK'])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')


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
dataset = HRRRDataset(TRAIN_FILE_PATH, input_keys, output_keys,None ,input_timestamps,label_timestamps,VARS_CHANNEL, training=True, stride=1)
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
dataloader_ = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
# dataloader_ = DataLoader(dataset, batch_size=batch_size, shuffle=True)

print(len(dataset)//(batch_size*8))


#训练
# 测试能否使用 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)




# model = AFNONet()
model = AFNOAttnParallelNet(
        input_keys,
        output_keys,
        img_size=(IMG_H, IMG_W),
        in_channels=len(VARS_CHANNEL)+1,
        # in_channels=len(VARS_CHANNEL),
        out_channels=len(VARS_CHANNEL),
        num_timestamps=label_timestamps,
        attn_channel_ratio=0.25
    )






model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


# pretrain_model  = torch.load('./output/rt/pretrain/modelww_44.pth')
# model.load_state_dict(pretrain_model)




# 定义损失函数和优化器
criterion = RelativeL2Loss()

optimizer = torch.optim.Adam(model.parameters(), lr= 5e-4)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
	optimizer, 
	T_max = EPOCHS , 
	eta_min=0, 
	last_epoch=- 1, 
	verbose=False)


last_epoch = 12
optimizer_path = OUTPUT_DIR + '/optimizer_' + str(last_epoch) + '.pth'
model_path = OUTPUT_DIR + '/modelww_' + str(last_epoch) + '.pth'
scheduler_path = OUTPUT_DIR + '/scheduler_' + str(last_epoch) + '.pth'

# 加载模型权重
model.load_state_dict(torch.load(model_path))

# 加载优化器状态
optimizer.load_state_dict(torch.load(optimizer_path))

# 加载学习率调度器状态
scheduler.load_state_dict(torch.load(scheduler_path))

last_epoch = last_epoch  # 设置为你中断时的最后一个 epoch
start_epoch = last_epoch + 1  # 从下一个 epoch 开始
print('load finish!')


for epoch in range(start_epoch, EPOCHS):

    # 每一个iteration
    for i, (input_item, label_item, weight_item,input_time,label_time, geo) in enumerate(dataloader_):
        
        start_time = time.time()

  

        input_item = input_item.reshape(-1,24,IMG_H, IMG_W).to(device=device)
        label_item = label_item
        label_list = [label_item[:, i, ...].to(device=device) for i in range(label_item.size(1))]
       
        input_time = [input_time[:, i, :].to(device = device) for i in range(input_time.size(1))]

        geo = geo.to(device=device)
     
       
        out = model(input_item, input_time, geo)
        
      

      
        loss=criterion(out, label_list)
      
      
       

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        end_time = time.time()
        
        if(i%10==0) and dist.get_rank() == 0:
  
            print(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),"  epoch: ",epoch,"iter: ",i,"迭代时间：", (end_time - start_time) , "秒","loss:",loss.item(),"lr: ",optimizer.param_groups[0]['lr'] )
            log_file_path = OUTPUT_DIR+'/train.log'
            with open(log_file_path, 'a') as log_file:
                # 您要记录的文本
                log_text = "{}, [train]  epoch: {} iter: {} 迭代时间： {} 秒 loss: {} lr: {}\n".format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),epoch, i, (end_time - start_time), loss.item(), optimizer.param_groups[0]['lr'])
                
                # 将文本写入日志文件
                log_file.write(log_text)
                
                
                torch.save(model.state_dict(), OUTPUT_DIR+'/modelww_'+str(epoch)+'.pth')
                
    scheduler.step()
    # eval(model, device, local_rank, epoch, label_timestamps,dist)

    
        


 







    





