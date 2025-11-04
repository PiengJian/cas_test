import torch
import numpy as np
from tools import save_tools
def RMSE(out,label,RMSE_list):
    # 48 1 24 H W -> 48 24 H W
    out = torch.stack(out, dim=0).reshape(-1,24,440,408)
    label = torch.stack(label, dim=0).reshape(-1,24,440,408)
    # out = out[:,:,48:-48,48:-48]
    # label = label[:,:,48:-48,48:-48]
    out = out[:,:,64:-64,64:-64]
    label = label[:,:,64:-64,64:-64]

    # save_tools.save_np(out,'out')
    # save_tools.save_np(label,'label')
    mse_values = torch.mean((out - label)**2, dim=(2, 3))
    rmse_values = torch.sqrt(mse_values)

    # 将RMSE张量转换为NumPy数组
    rmse_values = rmse_values.cpu().numpy()

    # rmse_values现在是一个48x24的NumPy数组，每一列代表一个变量在不同时间的RMSE值
    # 如果你想要一个24x48的数组，只需转置rmse_values
    rmse_values = rmse_values.T
    RMSE_list += rmse_values
    
    return RMSE_list

def ACC(out,label,ACC_list,mean,data_mean,data_std):
    out = torch.stack(out, dim=0).reshape(-1,24,440,408)
    label = torch.stack(label, dim=0).reshape(-1,24,440,408)
    out = out[:,:,64:-64,64:-64]
    label = label[:,:,64:-64,64:-64]

    # save_tools.save_np(out,'out')
    # save_tools.save_np(label,'label')
    out = out.cpu().numpy()
    label =label.cpu().numpy()
    # mean = np.load("/ssd1/hrrr_data/stat/time_mean_crop.npy").reshape(1,24,440,408).astype(np.float32) 
    # data_mean = np.load("/ssd1/hrrr_data/stat/mean_crop.npy").reshape(1,-1, 1, 1).astype(np.float32)
    # data_std = np.load("/ssd1/hrrr_data/stat/std_crop.npy").reshape(1,-1, 1, 1).astype(np.float32)
    mean = (mean-data_mean)/data_std
    mean = mean[:,:,64:-64,64:-64]
    
    ACC_values = np.sum((out - mean)*(label-mean), axis=(-2, -1)) / np.sqrt(   
                 np.sum( (out-mean)**2, axis=(-2, -1) ) 
               * np.sum((label-mean)**2, axis=(-2, -1))   
               )
    # rmse_values = torch.sqrt(mse_values)

    # 将RMSE张量转换为NumPy数组
    # ACC_values = ACC_values.cpu().numpy()

    # rmse_values现在是一个48x24的NumPy数组，每一列代表一个变量的RMSE值
    # 如果你想要一个24x48的数组，只需转置rmse_values
    ACC_values = ACC_values.T
    ACC_list += ACC_values
    
    return ACC_list





