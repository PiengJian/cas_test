from metric.RMSE import RMSE, ACC
import torch
from torch.utils.data import DataLoader
import numpy as np
from hrrr_dataset_marge import HRRRDataset as testHRRRDataset
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
print(device)

father_path = "/nfs/samba/数据聚变/气象数据/hrrr_440_30var"

# set dataset path
TRAIN_FILE_PATH =   father_path+"/train"
VALID_FILE_PATH =   father_path+"/test"
DATA_MEAN_PATH =    father_path+"/traindata_stat/var_mean.npy"
DATA_STD_PATH =     father_path+"/traindata_stat/var_std.npy"
DATA_TIME_MEAN_PATH =    father_path+"/traindata_stat/var_pos_mean.npy"

input_keys = ("input",)
output_keys = ("output",)
VARS_CHANNEL = list(range(30))
test_batch_size = 1
test_input_timestamps = 1
test_label_timestamps = 48
test_dataset = testHRRRDataset(VALID_FILE_PATH, test_input_timestamps,test_label_timestamps,VARS_CHANNEL, stride=24)
test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size,num_workers=4)

mean = np.load(father_path + "/traindata_stat/var_pos_mean.npy")[:,-224:,:224].reshape(1, 30, 224, 224).astype(np.float32) 
data_mean = np.load(father_path + "/traindata_stat/var_mean.npy").reshape(1, -1, 1, 1).astype(np.float32)
data_std = np.load(father_path + "/traindata_stat/var_std.npy").reshape(1, -1, 1, 1).astype(np.float32) 
std = np.load(father_path + "/traindata_stat/var_std.npy").reshape(-1, 1).astype(np.float32)   
marge_mean = torch.from_numpy(data_mean).float().to(device=device)
marge_std = torch.from_numpy(data_std).float().to(device=device)
OUTPUT_DIR = '/nfs/samba/数据聚变/气象数据/pengjian/result/cas_former_30var/nwp'


RMSE_list= [[0 for i in range(48)] for i in range(30)]
ACC_list= [[0 for i in range(48)] for i in range(30)]

for i,(input_item, label_item,input_time,label_time, merge) in enumerate(test_dataloader):
    # merge = torch.from_numpy(merge)
    # label_item = torch.from_numpy(label_item)
    merge = merge.to(device)
    merge = (merge-marge_mean)/marge_std
    label_list = [label_item[:, i, ...].to(device=device) for i in range(label_item.size(1))]
    merge_list = [merge[:, i, ...].to(device=device) for i in range(merge.size(1))]
    if i == 0:
        print(type(merge),type(label_item))
        print(label_list[0].shape, merge_list[0].shape,label_item.size(1),merge.size(1))
    RMSE_list = RMSE(merge_list, label_list, RMSE_list)
                # ACC_list = ACC(out_list[1:], target, RMSE_list)
    ACC_list = ACC(merge_list, label_list, ACC_list,mean,data_mean,data_std)
    rmse = (RMSE_list/(i+1))*std
    acc = ACC_list/(i+1)
    np.save(OUTPUT_DIR + "/nwp_rmse.npy", rmse)
    np.save(OUTPUT_DIR + "/nwp_acc.npy", acc)