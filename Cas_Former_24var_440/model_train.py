import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim.lr_scheduler as lr_scheduler
import json
import os
# import xarray as xr
import yaml
import time
from torch.cuda.amp import autocast, GradScaler
from hrrr_dataset import geovar
from hrrr_dataset import HRRRDataset as trainDataset

# 预训练
# from hrrr_dataset import HRRRDataset as testDataset
# 微调
from hrrr_dataset_marge import HRRRDataset as testDataset
from metric.RMSE import RMSE, ACC
from loss.l2 import RelativeL2Loss,RelativeL2LossForC
LOSS_FUNCTIONS = {'RelativeL2Loss':RelativeL2Loss,'RelativeL2LossForC':RelativeL2LossForC}

class ModelTrain():
    def __init__(self, model, config,local_rank):
        self.device = torch.device('cuda:' + str(local_rank))
        # with open(config_path, 'r') as f:
        #     self.config = yaml.safe_load(f)
        self.config = config
        self.model = model
        self.merge = self.config['model']['merge']
        self.start_epoch = self.config['model']['start_epoch']
        self.use_ddp = self.config['model']['use_ddp']
        if self.use_ddp:
            # dist.init_process_group(backend='nccl')
            # self.local_rank = dist.get_rank()
            self.local_rank = local_rank
            print(f'use ddp, local_rank:{self.local_rank}')
            # torch.cuda.set_device(self.local_rank)
            # self.device = torch.device("cuda",self.local_rank)
            
        self.use_mixtrain = self.config['model']['use_mixtrain']
        self.scaler = GradScaler() if self.use_mixtrain else None
        self.only_test = self.config['model']['only_test']
        self.IMG_H, self.IMG_W = self.config['model']['IMG_H'], self.config['model']['IMG_W']
        self.load_weights = self.config['model']['load_weights']
        if self.load_weights:
            self.model_weights_save_path = self.config['model']['model_weights']
            self.load_model_weights()
        else:
            self.model.to(self.device)
            if self.use_ddp: 
                self.model = DDP(self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank, 
                    find_unused_parameters=True)
                print("model without load weight with ddp")
            else:
                print("model without load weight without ddp")
                            
        self.model_loss_save_dir = self.config['model']['model_loss']
        self.train_batch_file = os.path.join(self.model_loss_save_dir, "train_batch.log")
        self.test_batch_file = os.path.join(self.model_loss_save_dir, "test_batch.log")
        self.epoch_file = os.path.join(self.model_loss_save_dir, "epoch_loss.json")

        self.train_batch_size = self.config['train']['train_batch_size']
        self.train_input_timestamps = self.config['train']['train_input_timestamps']
        self.train_label_timestamps = self.config['train']['train_label_timestamps']
        self.test_batch_size = self.config['test']['test_batch_size']
        self.test_input_timestamps = self.config['test']['test_input_timestamps']
        self.test_label_timestamps = self.config['test']['test_label_timestamps']
        self.variable_weights = self.config['train']['variable_weights']
        self.time_weights = self.config['train']['time_weights']
        self.train_data_path = self.config['train']['train_data_path']
        self.test_data_path = self.config['test']['test_data_path']
        self.loss_name = self.config['train']['loss']
        if self.loss_name in LOSS_FUNCTIONS:
            self.criterion = LOSS_FUNCTIONS[self.loss_name]()
        optimizer_config = self.config['train']['optimizer']
        self.optimizer = getattr(torch.optim, optimizer_config['name'])(
            self.model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'])
            # momentum=optimizer_config['momentum'])
        if self.start_epoch != 0:    
            for group in self.optimizer.param_groups:
                if 'initial_lr' not in group:
                    group['initial_lr'] = group['lr']
        
        # 学习率调度器
        scheduler_config = self.config['train'].get('scheduler', None)
        if scheduler_config:
            self.scheduler = getattr(lr_scheduler, scheduler_config['name'])(
                self.optimizer,
                T_max = scheduler_config['T_max'],
                eta_min=float(scheduler_config['eta_min']),
                last_epoch=scheduler_config['last_epoch'],
                verbose=False
            )
        else:
            self.scheduler = None
            print('No Scheduler')
        
        self.geo_path = self.config['model']['geo']
        self.geo = geovar(self.geo_path)
        # self.geo = self.geo.to(self.device)
        # 初始化JSON文件结构
        if not os.path.exists(self.epoch_file):
            with open(self.epoch_file, "w") as f:
                json.dump({"train": [], "test": []}, f)
        
        # # 创建目录（仅主进程执行）
        # if not self.use_ddp or (self.use_ddp and dist.get_rank() == 0):
        #     self._create_directories()
    # def _create_directories(self):
    #     """创建模型权重和损失文件的存储目录"""
    #     os.makedirs(
    #         os.path.dirname(self.model_weights_save_path),
    #         exist_ok=True
    #     )
    def load_model_weights(self):
        weights = torch.load(self.model_weights_save_path, map_location=torch.device("cuda:0"))
        self.model.load_state_dict(weights['state_dict'])
        # self.optimizer.load_state_dict(weights['optimizer'])
        # self.scheduler.load_state_dict(weights['scheduler'])
        self.model.to(self.device)
        if self.use_ddp: 
            self.model = DDP(self.model,
                    device_ids=[self.local_rank],
                    output_device=self.local_rank, 
                    find_unused_parameters=True)
            print("model has load weight with ddp")
        else:
            print("model has load weight without ddp")
                
                

    def dataset(self, train_data_path, test_data_path=None):
        """
    Args:
        train_data_path (str): 训练数据根目录路径
        test_data_path (str, optional): 测试数据根目录路径. Defaults to None.
    """
        # 创建训练数据集 - 24个变量在30个变量中的位置映射
        # 位置: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 21, 22, 20
        VARS_CHANNEL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 21, 22, 20]
        train_loader = None
        if not self.only_test:
            train_dataset = trainDataset(train_data_path, 
                                        self.train_input_timestamps,
                                        self.train_label_timestamps,
                                        VARS_CHANNEL,
                                        stride=1)
            # 创建训练数据加载器（支持分布式训练）
            if self.use_ddp:
                self.train_sampler = DistributedSampler(train_dataset, shuffle=True,rank=self.local_rank)
                train_loader = DataLoader(
                    train_dataset,
                    sampler=self.train_sampler,
                    batch_size=self.train_batch_size,
                    num_workers=4)
            else:
                train_loader = DataLoader(
                    train_dataset, 
                    batch_size=self.train_batch_size, 
                    shuffle=True,
                    num_workers=4)
        
        # 创建测试数据加载器（如果存在测试数据）
        test_loader = None
        if test_data_path:
            test_dataset = testDataset(
                test_data_path, 
                self.test_input_timestamps,
                self.test_label_timestamps,
                VARS_CHANNEL, 
                stride=24)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=4)
            # if self.use_ddp:
            #     test_sampler = DistributedSampler(test_dataset, shuffle=False,num_replicas=self.world_size,rank=self.local_rank)  # 测试集不需要shuffle
            #     test_loader = DataLoader(
            #         test_dataset,
            #         sampler=test_sampler,
            #         batch_size=self.test_batch_size,
            #         num_workers=4,
            #         pin_memory=True
            #     )
            # else:
            #     test_loader = DataLoader(
            #         test_dataset,
            #         batch_size=self.test_batch_size,
            #         shuffle=False,
            #         num_workers=4,
            #         pin_memory=True
            #     )
        self.train_loader = train_loader
        self.test_loader = test_loader
        
        return train_loader, test_loader  # 仅返回训练和测试加载器


    def train_one_epoch(self,epoch):
        self.model.module.num_timestamps = self.train_label_timestamps
        self.model.train()
        start_time = time.time()
        total_loss = 0.0
        num_samples = len(self.train_loader.dataset)
        batches_per_epoch = len(self.train_loader)
        # 主进程打印样本数量
        if self.use_ddp:
            if dist.get_rank() == 0:
                print(f"ddp Sample of train: {num_samples}")
        else:
            print(f"Sample of train: {num_samples}")
        geo= np.repeat(self.geo[np.newaxis, ...], self.train_batch_size, axis=0)
        geo = torch.from_numpy(geo).to(torch.float32)
        geo = geo.contiguous().reshape((self.train_batch_size, 1 , self.IMG_H, self.IMG_W))
    
        for batch_idx, (input_item, label_item,input_time,label_time) in enumerate(self.train_loader):
            input_item = input_item.reshape(-1,24,self.IMG_H, self.IMG_W).to(self.device)
            label_list = [label_item[:, i, ...].to(self.device) for i in range(label_item.size(1))]
            input_time = [input_time[:, i, :].to(self.device) for i in range(input_time.size(1))]
            # geo = geo.cuda(non_blocking=True)
            if input_item.shape[0] != self.train_batch_size:
                geo= np.repeat(self.geo[np.newaxis, ...], input_item.shape[0], axis=0)
                geo = torch.from_numpy(geo).to(torch.float32)
                geo = geo.contiguous().reshape((input_item.shape[0], 1 , self.IMG_H, self.IMG_W))
                
        
            if self.use_mixtrain:
                with autocast():
                    pred = self.model(input_item, input_time, geo,merge=None, marge_mean=None,marge_std=None,Wm=None,Wn=None)
                    if self.loss_name == 'RelativeL2Loss':
                        if batch_idx == 0:
                            print(f'predshape{len(pred)},{pred[0].shape}')
                            print(f'labelshape{len(label_list)},{label_list[0].shape}')
                        loss = self.criterion(pred, label_list)    
                    else:
                        loss = self.criterion(pred, label_list,self.variable_weights,self.time_weights)  
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred = self.model(input_item, input_time, geo,merge=None, marge_mean=None,marge_std=None,Wm=None,Wn=None)
                if self.loss_name == 'RelativeL2Loss':
                    loss = self.criterion(pred, label_list)    
                else:
                    loss = self.criterion(pred, label_list,self.variable_weights,self.time_weights) 
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                with open(self.train_batch_file, "a") as f:
                    log_info = f'train: epoch {epoch}, iter:{batch_idx}/{batches_per_epoch}, loss: {loss.item():.4f}, lr: {current_lr}\n'
                    f.write(log_info)
            
        save_path_dir = self.model_loss_save_dir+'/weight'
        os.makedirs(os.path.dirname(save_path_dir), exist_ok=True)
        torch.save({'epoch': epoch, 
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),}, save_path_dir+'/model_'+str(epoch)+'.pth')
        
        if self.scheduler:
            self.scheduler.step()
        
        end_time = time.time()
        avg_loss = total_loss / batches_per_epoch
        elapsed_time = end_time - start_time
        
        # 记录epoch结果到JSON
        self._record_epoch_result(
            phase="train",
            epoch=epoch,
            loss=avg_loss,
            time_diff=elapsed_time
        )
        
        return avg_loss


    def test_one_epoch(self,merge,epoch):
        
        self.model.num_timestamps = self.test_label_timestamps
  
        batches_per_epoch = len(self.test_loader)
        total_loss = 0.0
        
        # 加载30个变量的统计信息
        mean_all = np.load("/nfs/samba/数据聚变/气象数据/hrrr_440_30var/traindata_stat/var_pos_mean.npy").reshape(1, 30, self.IMG_H, self.IMG_W).astype(np.float32) 
        data_mean_all = np.load("/nfs/samba/数据聚变/气象数据/hrrr_440_30var/traindata_stat/var_mean.npy").reshape(1, -1, 1, 1).astype(np.float32)
        data_std_all = np.load("/nfs/samba/数据聚变/气象数据/hrrr_440_30var/traindata_stat/var_std.npy").reshape(1, -1, 1, 1).astype(np.float32) 
        std_all = np.load("/nfs/samba/数据聚变/气象数据/hrrr_440_30var/traindata_stat/var_std.npy").reshape(-1, 1).astype(np.float32)
        
        # 选择24个变量对应的统计信息
        # 位置: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 21, 22, 20
        vars_channel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 26, 21, 22, 20]
        mean = mean_all[:, vars_channel, :, :]
        data_mean = data_mean_all[:, vars_channel, :, :]
        data_std = data_std_all[:, vars_channel, :, :]
        std = std_all[vars_channel]  
        geo= np.repeat(self.geo[np.newaxis, ...], self.test_batch_size, axis=0)
        geo = torch.from_numpy(geo).to(torch.float32)
        geo = geo.contiguous().reshape((self.test_batch_size, 1 , self.IMG_H, self.IMG_W)).to(self.device)

        RMSE_list= [[0 for i in range(self.test_label_timestamps)] for i in range(24)]
        ACC_list= [[0 for i in range(self.test_label_timestamps)] for i in range(24)]

        if merge:
            Wm=torch.from_numpy(np.load('/home/pengjian/CAS_Former_440/CASmwp64.npy')).float().to(self.device)
            Wn=torch.from_numpy(np.load('/home/pengjian/CAS_Former_440/nwp67.npy')).float().to(self.device)
            marge_mean = torch.from_numpy(data_mean).float().to(self.device)
            marge_std = torch.from_numpy(data_std).float().to(self.device)
            # Wm=torch.from_numpy(np.load('/home/jovyan/boundary_weight/CASmwp48.npy')).float().to(device)
            # Wn=torch.from_numpy(np.load('/home/jovyan/boundary_weight/CASnwp48.npy')).float().to(device)
            # marge_mean = torch.from_numpy(data_mean).float().to(device)
            # marge_std = torch.from_numpy(data_std).float().to(device)

        self.model.eval()
        
        with torch.no_grad():
            if merge:
                for i,(input_item, label_item,input_time,label_time, merge) in enumerate(self.test_loader):
                # for i,(input_item, label_item, weight_item,input_time,label_time, geo, marge) in enumerate(tqdm(test_loader)):
                # i,(input_item, label_item, weight_item,input_time,label_time, geo, merge) =0,next(iter(test_loader))
                
                    start_time = time.time()
                    
                    input_item = input_item.reshape(-1,24,self.IMG_H, self.IMG_W).to(self.device)
                    # input_item = input_item.reshape(-1,24,self.IMG_H, self.IMG_W).to(device)
                    # label_list = [label_item[:, i, ...].to(device=device) for i in range(label_item.size(1))]
                    # input_time = [input_time[:, i, :].to(device = device) for i in range(input_time.size(1))]
                    label_list = [label_item[:, i, ...].to(self.device) for i in range(label_item.size(1))]
                    input_time = [input_time[:, i, :].to(self.device) for i in range(input_time.size(1))]
                    if input_item.shape[0] != self.test_batch_size:
                        geo= np.repeat(self.geo[np.newaxis, ...], input_item.shape[0], axis=0)
                        geo = torch.from_numpy(geo).to(torch.float32)
                        geo = geo.contiguous().reshape((input_item.shape[0], 1 , self.IMG_H, self.IMG_W))
                    # geo = geo.to(device=device)
                    # geo = geo.cuda(non_blocking=True)
                    # marge = marge.to(device=device)
                    merge = merge.to(self.device)
                    
                    out = self.model(input_item, input_time, geo,  merge, marge_mean,marge_std,Wm,Wn)
                    if i == 0:
                        print(f'predshape{len(out)},{out[0].shape}')
                        print(f'labelshape{len(label_list)},{label_list[0].shape}')
                    loss = self.criterion(out, label_list)
                    total_loss += loss.item()
                    
                    RMSE_list = RMSE(out, label_list, RMSE_list)
                    # ACC_list = ACC(out_list[1:], target, RMSE_list)
                    ACC_list = ACC(out, label_list, ACC_list,mean,data_mean,data_std)
                    # ACC_list = ACC(out, target, ACC_list)
                    end_time = time.time()
                    
                    if i%10==0:
                        with open(self.test_batch_file, 'a') as log_file:
                            # 您要记录的文本
                            log_info = f'inference: iter:{i}/{batches_per_epoch},迭代时间： {end_time - start_time:.2f} 秒 , loss: {loss.item():.4f}\n'
                            log_file.write(log_info)

                    rmse = (RMSE_list/(i+1))*std
                    acc = ACC_list/(i+1)
                    np.save(self.model_loss_save_dir +"/rmse"+ "/rmse"+str(epoch)+".npy", rmse)
                    np.save(self.model_loss_save_dir +"/acc"+ "/acc"+str(epoch)+".npy", acc)
                
            else:
                for i, (input_item, label_item,input_time,label_time) in enumerate(self.test_loader):
                # for i, (input_item, label_item, weight_item,input_time,label_time, geo) in enumerate(tqdm(test_loader)):
                # i,(input_item, label_item, weight_item,input_time,label_time, geo) =10,next(iter(test_loader))
                    start_time = time.time()
                    
                    input_item = input_item.reshape(-1,24,self.IMG_H, self.IMG_W).to(self.device)
                    # label_list = [label_item[:, i, ...].to(device=device) for i in range(label_item.size(1))]
                    # input_time = [input_time[:, i, :].to(device = device) for i in range(input_time.size(1))]
                    label_list = [label_item[:, i, ...].to(self.device) for i in range(label_item.size(1))]
                    input_time = [input_time[:, i, :].to(self.device) for i in range(input_time.size(1))]
                    if input_item.shape[0] != self.test_batch_size:
                        geo= np.repeat(self.geo[np.newaxis, ...], input_item.shape[0], axis=0)
                        geo = torch.from_numpy(geo).to(torch.float32)
                        geo = geo.contiguous().reshape((input_item.shape[0], 1 , self.IMG_H, self.IMG_W))
                    # geo = geo.to(device=device)
                    # geo = geo.cuda(non_blocking=True)
                    
                    out = self.model(input_item, input_time, geo)
                    if i == 0:
                        print(f'predshape{len(out)},{out[0].shape}')
                        print(f'labelshape{len(label_list)},{label_list[0].shape}')
                    loss = self.criterion(out, label_list)
                    total_loss += loss.item()
                    
                    RMSE_list = RMSE(out, label_list, RMSE_list)
                    # ACC_list = ACC(out_list[1:], target, RMSE_list)
                    ACC_list = ACC(out, label_list, ACC_list,mean,data_mean,data_std)
                    # ACC_list = ACC(out, target, ACC_list)
                    end_time = time.time()
                    
                    if i%10==0:
                        with open(self.test_batch_file, 'a') as log_file:
                            log_info = f'inference: iter:{i}/{batches_per_epoch},迭代时间： {end_time - start_time:.2f} 秒 , loss: {loss.item():.4f}\n'
                            log_file.write(log_info)

                    rmse = (RMSE_list/(i+1))*std
                    acc = ACC_list/(i+1)
                    np.save(self.model_loss_save_dir +"/rmse"+ "/rmse"+str(epoch)+".npy", rmse)
                    np.save(self.model_loss_save_dir +"/acc"+ "/acc"+str(epoch)+".npy", acc)
              
            
        avg_loss = total_loss / batches_per_epoch
        
        # 记录epoch结果到JSON
        self._record_epoch_result(
            phase="test",
            epoch=epoch,
            loss=avg_loss,
            time_diff=1
        )
        
        return avg_loss


    def train(self, epochs):
        _, _ = self.dataset(train_data_path=self.train_data_path,test_data_path=self.test_data_path)
        test_loss       = 0

        for i in range(self.start_epoch, epochs):
            if self.use_ddp: 
                if dist.get_rank() == 0:
                    print(f"ddpEpoch {i}\n-------------------------------")
            else:
                print(f"Epoch {i}\n-------------------------------")
            if not self.only_test:
                # 为了让每张卡在每个周期中得到的数据是随机的
                self.train_sampler.set_epoch(i)
                train_loss = self.train_one_epoch(epoch = i)
            test_loss = self.test_one_epoch(merge = self.merge, epoch = i)
            # self.scheduler.step(test_loss)
        


    def _record_epoch_result(self, phase, epoch, loss, time_diff):
        """记录epoch结果到JSON文件"""
        if self.use_ddp and dist.get_rank() != 0:
            return
        # 读取现有数据
        with open(self.epoch_file, "r") as f:
            data = json.load(f)
        # 创建新记录
        record = {
            "epoch": epoch,
            "loss": round(loss, 4),
            "time(s)": round(time_diff, 2),
        }
        # 更新对应阶段的记录
        data[phase].append(record)
        # 写回文件
        with open(self.epoch_file, "w") as f:
            json.dump(data, f, indent=2)
    
    
    
    
    
    
    
    
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.model_weights_save_path)
        print(f"Saved PyTorch Model State to {self.model_weights_save_path}, \
                loss from {self.loss_state:>.7f} to {self.val_loss:>.7f}")  
    
    def _index_cal(self, data_loader):
        num_batches = len(data_loader)
        if self.use_ddp:
            if dist.get_rank()==0: print("batch : ", num_batches)
        self.model.eval()
        data_loss = 0
        with torch.no_grad():
            for X, y in data_loader:
                X, y = X.to(self.device), y.to(self.device)
                pred = self.model(X)
                data_loss += self.criterion(pred, y).item()
    
        data_loss /= num_batches
        return data_loss
    
    @staticmethod
    def save_json_file(data_dic, save_path):

        with open(save_path, 'w') as f:
            json.dump(data_dic, f, indent = 4, sort_keys = True)
            f.close()
        print(f"Saved json file to {save_path}!")
        
    @staticmethod
    def create_directory(directory):

        if not os.path.exists(directory):
            os.makedirs(directory)
            print(directory, " have created!")
        else:
            print(directory, " already exists!")
