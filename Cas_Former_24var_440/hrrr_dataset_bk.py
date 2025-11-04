# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Dict
from typing import Optional
from typing import Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils.timefeatures import time_features
import pandas as pd

class HRRRDataset(Dataset):
    """Class for HRRR dataset.

    Args:
        file_path (str): Data set path.
        input_keys (Tuple[str, ...]): Input keys, such as ("input",).
        label_keys (Tuple[str, ...]): Output keys, such as ("output",).
        precip_file_path (Optional[str]): Precipitation data set path. Defaults to None.
        weight_dict (Optional[Dict[str, float]]): Weight dictionary. Defaults to None.
        vars_channel (Optional[Tuple[int, ...]]): The variable channel index in ERA5 dataset. Defaults to None.
        num_label_timestamps (int, optional): Number of timestamp of label. Defaults to 1.
        transforms (Optional[vision.Compose]): Compose object contains sample wise
            transform(s). Defaults to None.
        training (bool, optional): Whether in train mode. Defaults to True.
        stride (int, optional): Stride of sampling data. Defaults to 1.

    Examples:
        >>> import ppsci
        >>> dataset = ppsci.data.dataset.ERA5Dataset(
        ...     "file_path": "/path/to/ERA5Dataset",
        ...     "input_keys": ("input",),
        ...     "label_keys": ("output",),
        ... )  # doctest: +SKIP
    """

    def __init__(
        self,
        file_path: str,
        input_keys: Tuple[str, ...],
        label_keys: Tuple[str, ...],
        weight_dict: Optional[Dict[str, float]] = None,
        num_input_timestamps: int = 1,
        num_label_timestamps: int = 1,
        vars_channel: Optional[Tuple[int, ...]] = None,
        # transforms: Optional[vision.Compose] = None,
        training: bool = True,
        stride: int = 1,
    ):
        super().__init__()
        self.file_path = file_path
        self.father_path = os.path.dirname(file_path)
        self.input_keys = input_keys
        self.label_keys = label_keys

        self.weight_dict = {key: 1.0 for key in self.label_keys}
        if weight_dict is not None:
            self.weight_dict.update(weight_dict)

        self.vars_channel = list(range(69)) if vars_channel is None else vars_channel
        self.num_label_timestamps = num_label_timestamps
        self.num_input_timestamps = num_input_timestamps
        # self.transforms = transforms
        self.training = training
        self.stride = stride

        self.files = self.read_data(file_path)
    
        self.num_days = len(self.files)
        self.num_samples_per_day = self.files[0][0].shape[0]
        self.num_samples = self.num_days * self.num_samples_per_day
    
    def transforms(self,input_item, label_item):
        data_mean = np.load(  self.father_path+"/stat/mean_crop.npy").reshape(-1, 1, 1).astype(np.float32)
        
        data_std = np.load(  self.father_path+"/stat/std_crop.npy").reshape(-1, 1, 1).astype(np.float32)
        input_item = (input_item - data_mean)/data_std
        label_item = (label_item - data_mean)/data_std
        
        return input_item, label_item



    def set_time(self,time_stamp):
        # 将日期时间字符串解析为日期时间对象
        # time_stamp = [pd.to_datetime(date_str, format='%Y/%m/%d/%H') for date_str in time_stamp]
        time_stamp = [pd.to_datetime(date_str, format='%Y/%m/%d/%H') for date_str in time_stamp]
        time_stamp = pd.DataFrame({'date': time_stamp})

        time_feature = time_features(time_stamp, timeenc=1, freq='h').astype(np.float32)
        time_feature = torch.from_numpy(time_feature.squeeze())

        return time_feature

    def read_data(self, path: str, var="fields"):
        if path.endswith(".h5"):
            paths = [path]
        else:
            paths = []
            for root, dirs, files in os.walk(path):
                for file in files:
                    paths.append(os.path.join(root, file))
        paths.sort()
        files = []
        for path_ in paths:
            _file = h5py.File(path_, "r")
            files.append([_file[var],path_[-13:-3]])
        print(files[0][-1])
        print(files[1][-1])
        print(files[2][-1])
        return files

    def __len__(self):
        return self.num_samples // self.stride

    def __getitem__(self, global_idx):

        global_idx *= self.stride

        if global_idx >= self.num_samples - self.num_label_timestamps-7:
            return self.__getitem__(np.random.randint(self.__len__()))
        
        input_item_list = []
        input_time_list = []
        label_item_list = []
        label_time_list = []
        
        for i in range(self.num_input_timestamps):
        # input_time ={}
            input_day_idx = (global_idx+i) // self.num_samples_per_day
            input_hour_idx = (global_idx+i) % self.num_samples_per_day

            input_file = self.files[input_day_idx][0]
            # check fake data
            if len(input_file.shape) == 1:
                # print("Warning: fake data detected, please check your data")
                return self.__getitem__(np.random.randint(self.__len__()))
            # input_item = {self.input_keys[0]: input_file[input_hour_idx, self.vars_channel]}    
            # input_time[self.input_keys[0]] = self.files[input_day_idx][1]+"/"+str(input_hour_idx)

            input_item = input_file[input_hour_idx, self.vars_channel]


            input_time = [self.files[input_day_idx][1]+"/"+str(input_hour_idx)]
            input_time = self.set_time(input_time)
            input_item_list.append(input_item)
            input_time_list.append(input_time)
        for i in range(self.num_label_timestamps):
            label_day_idx = (global_idx  +1+ i) // self.num_samples_per_day
            label_hour_idx = (global_idx  +1+ i) % self.num_samples_per_day
            label_file = self.files[label_day_idx][0]
            if len(label_file.shape) == 1:
                # print("Warning: fake data detected, please check your data")
                return self.__getitem__(np.random.randint(self.__len__()))

            
            
            label_item = label_file[label_hour_idx, self.vars_channel]


            label_time = [self.files[label_day_idx][1]+"/"+str(label_hour_idx)]
            label_time = self.set_time(label_time)
            label_item_list.append(label_item)
            label_time_list.append(label_time)
            input_time_list.append(label_time)
        
        input_item_list = np.stack(input_item_list, axis=0)
        input_time_list = np.stack(input_time_list, axis=0)
        label_item_list = np.stack(label_item_list, axis=0)
        label_time_list = np.stack(label_time_list, axis=0)


        weight_shape = [1] * len(next(iter(label_item)).shape)
        weight_item = {
            key: np.full(weight_shape, value, 'float32')
            for key, value in self.weight_dict.items()
        }

        
        # input_item, label_item = self.transforms(input_item, label_item)

        # return input_item, label_item, weight_item,input_time,label_time
        
        input_item_list, label_item_list = self.transforms(input_item_list, label_item_list)
        with h5py.File( self.father_path+"/geo.h5", 'r') as h5_file:
            geo = h5_file['fields'][:].astype(np.float32)

        return input_item_list, label_item_list, weight_item,input_time_list,label_time_list, geo
