import os
import h5py
import numpy as np
import datetime
from datetime import date, timedelta
from tqdm import tqdm

# 配置路径
forecast_root = "/nfs/samba/数据聚变/气象数据/hefang/hrrr_nwp_east_h5/2024"
truth_root = "/nfs/samba/数据聚变/气象数据/hrrr_440_30var/test/2024"
stat_root = "/nfs/samba/数据聚变/气象数据/hrrr_440_30var/traindata_stat"
output_dir = "/home/pengjian/CAS_Former_440/yinglong_nwp_results/metric"

# 预测24变量 -> 真值30变量中的通道索引
truth_idx_for_pred24 = np.array([
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    26, 21, 22, 20  # mslp, u10, v10, t2m
], dtype=int)

def crop_hw(x, pad=64):
    return x[..., pad:-pad, pad:-pad] if pad > 0 else x

def day_file(root, d):
    return os.path.join(root, f"{d:%m}", f"{d:%d}.h5")

# 预加载统计量
mean = np.load(os.path.join(stat_root, "var_pos_mean.npy")).reshape(1, 30, 440, 408).astype(np.float32)
data_mean = np.load(os.path.join(stat_root, "var_mean.npy")).reshape(1, -1, 1, 1).astype(np.float32)
data_std = np.load(os.path.join(stat_root, "var_std.npy")).reshape(1, -1, 1, 1).astype(np.float32)


# 子集到24变量并预计算
mean_24 = mean[:, truth_idx_for_pred24, ...]
data_mean_24 = data_mean[:, truth_idx_for_pred24, ...]
data_std_24 = data_std[:, truth_idx_for_pred24, ...]

# 预计算标准化后的气候场
mean_norm = (mean_24 - data_mean_24) / data_std_24
mean_norm_c = crop_hw(mean_norm, pad=64)

# 年度汇总 - 修改为与RMSE.py一致的计算方式
# 先对每个样本计算RMSE和ACC，再求平均
rmse_list = []
acc_list = []

# 假设valid_days已经提前获得
valid_days = [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2), datetime.date(2024, 1, 3), datetime.date(2024, 1, 4), datetime.date(2024, 1, 5), datetime.date(2024, 1, 6), datetime.date(2024, 1, 7), datetime.date(2024, 1, 8), datetime.date(2024, 1, 9), datetime.date(2024, 1, 10), datetime.date(2024, 1, 11), datetime.date(2024, 1, 12), datetime.date(2024, 1, 13), datetime.date(2024, 1, 14), datetime.date(2024, 1, 15), datetime.date(2024, 1, 16), datetime.date(2024, 1, 17), datetime.date(2024, 1, 18), datetime.date(2024, 1, 19), datetime.date(2024, 1, 20), datetime.date(2024, 1, 21), datetime.date(2024, 1, 22), datetime.date(2024, 1, 23), datetime.date(2024, 1, 24), datetime.date(2024, 1, 25), datetime.date(2024, 1, 26), datetime.date(2024, 1, 27), datetime.date(2024, 1, 28), datetime.date(2024, 1, 29), datetime.date(2024, 1, 30), datetime.date(2024, 1, 31), datetime.date(2024, 2, 1), datetime.date(2024, 2, 2), datetime.date(2024, 2, 3), datetime.date(2024, 2, 4), datetime.date(2024, 2, 5), datetime.date(2024, 2, 6), datetime.date(2024, 2, 7), datetime.date(2024, 2, 8), datetime.date(2024, 2, 9), datetime.date(2024, 2, 10), datetime.date(2024, 2, 11), datetime.date(2024, 2, 12), datetime.date(2024, 2, 13), datetime.date(2024, 2, 14), datetime.date(2024, 2, 15), datetime.date(2024, 2, 16), datetime.date(2024, 2, 17), datetime.date(2024, 2, 18), datetime.date(2024, 2, 19), datetime.date(2024, 2, 20), datetime.date(2024, 2, 21), datetime.date(2024, 2, 22), datetime.date(2024, 2, 23), datetime.date(2024, 2, 24), datetime.date(2024, 2, 25), datetime.date(2024, 2, 26), datetime.date(2024, 2, 27), datetime.date(2024, 2, 28), datetime.date(2024, 2, 29), datetime.date(2024, 3, 1), datetime.date(2024, 3, 2), datetime.date(2024, 3, 3), datetime.date(2024, 3, 4), datetime.date(2024, 3, 5), datetime.date(2024, 3, 6), datetime.date(2024, 3, 7), datetime.date(2024, 3, 8), datetime.date(2024, 3, 9), datetime.date(2024, 3, 10), datetime.date(2024, 3, 11), datetime.date(2024, 3, 12), datetime.date(2024, 3, 13), datetime.date(2024, 3, 14), datetime.date(2024, 3, 15), datetime.date(2024, 3, 16), datetime.date(2024, 3, 17), datetime.date(2024, 3, 18), datetime.date(2024, 3, 19), datetime.date(2024, 3, 20), datetime.date(2024, 3, 22), datetime.date(2024, 3, 23), datetime.date(2024, 3, 24), datetime.date(2024, 3, 25), datetime.date(2024, 3, 26), datetime.date(2024, 3, 27), datetime.date(2024, 3, 28), datetime.date(2024, 3, 29), datetime.date(2024, 3, 30), datetime.date(2024, 3, 31), datetime.date(2024, 4, 1), datetime.date(2024, 4, 2), datetime.date(2024, 4, 3), datetime.date(2024, 4, 4), datetime.date(2024, 4, 5), datetime.date(2024, 4, 6), datetime.date(2024, 4, 7), datetime.date(2024, 4, 8), datetime.date(2024, 4, 9), datetime.date(2024, 4, 10), datetime.date(2024, 4, 11), datetime.date(2024, 4, 12), datetime.date(2024, 4, 13), datetime.date(2024, 4, 14), datetime.date(2024, 4, 15), datetime.date(2024, 4, 16), datetime.date(2024, 4, 17), datetime.date(2024, 4, 18), datetime.date(2024, 4, 19), datetime.date(2024, 4, 20), datetime.date(2024, 4, 21), datetime.date(2024, 4, 22), datetime.date(2024, 4, 23), datetime.date(2024, 4, 24), datetime.date(2024, 4, 25), datetime.date(2024, 4, 26), datetime.date(2024, 4, 27), datetime.date(2024, 4, 28), datetime.date(2024, 4, 29), datetime.date(2024, 4, 30), datetime.date(2024, 5, 1), datetime.date(2024, 5, 2), datetime.date(2024, 5, 3), datetime.date(2024, 5, 4), datetime.date(2024, 5, 5), datetime.date(2024, 5, 6), datetime.date(2024, 5, 7), datetime.date(2024, 5, 8), datetime.date(2024, 5, 9), datetime.date(2024, 5, 10), datetime.date(2024, 5, 11), datetime.date(2024, 5, 12), datetime.date(2024, 5, 13), datetime.date(2024, 5, 14), datetime.date(2024, 5, 15), datetime.date(2024, 5, 16), datetime.date(2024, 5, 17), datetime.date(2024, 5, 18), datetime.date(2024, 5, 19), datetime.date(2024, 5, 23), datetime.date(2024, 5, 24), datetime.date(2024, 5, 25), datetime.date(2024, 5, 26), datetime.date(2024, 5, 27), datetime.date(2024, 5, 28), datetime.date(2024, 5, 29), datetime.date(2024, 5, 30), datetime.date(2024, 5, 31), datetime.date(2024, 6, 1), datetime.date(2024, 6, 2), datetime.date(2024, 6, 3), datetime.date(2024, 6, 4), datetime.date(2024, 6, 5), datetime.date(2024, 6, 6), datetime.date(2024, 6, 7), datetime.date(2024, 6, 8), datetime.date(2024, 6, 9), datetime.date(2024, 6, 10), datetime.date(2024, 6, 11), datetime.date(2024, 6, 12), datetime.date(2024, 6, 13), datetime.date(2024, 6, 14), datetime.date(2024, 6, 15), datetime.date(2024, 6, 16), datetime.date(2024, 6, 17), datetime.date(2024, 6, 18), datetime.date(2024, 6, 19), datetime.date(2024, 6, 20), datetime.date(2024, 6, 21), datetime.date(2024, 6, 22), datetime.date(2024, 6, 23), datetime.date(2024, 6, 24), datetime.date(2024, 6, 25), datetime.date(2024, 6, 26), datetime.date(2024, 6, 27), datetime.date(2024, 6, 28), datetime.date(2024, 6, 29), datetime.date(2024, 6, 30), datetime.date(2024, 7, 1), datetime.date(2024, 7, 2), datetime.date(2024, 7, 3), datetime.date(2024, 7, 4), datetime.date(2024, 7, 5), datetime.date(2024, 7, 6), datetime.date(2024, 7, 7), datetime.date(2024, 7, 8), datetime.date(2024, 7, 9), datetime.date(2024, 7, 10), datetime.date(2024, 7, 11), datetime.date(2024, 7, 12), datetime.date(2024, 7, 13), datetime.date(2024, 7, 14), datetime.date(2024, 7, 15), datetime.date(2024, 7, 16), datetime.date(2024, 7, 17), datetime.date(2024, 7, 18), datetime.date(2024, 7, 19), datetime.date(2024, 7, 20), datetime.date(2024, 7, 21), datetime.date(2024, 7, 22), datetime.date(2024, 7, 23), datetime.date(2024, 7, 24), datetime.date(2024, 7, 25), datetime.date(2024, 7, 26), datetime.date(2024, 7, 27), datetime.date(2024, 7, 28), datetime.date(2024, 7, 29), datetime.date(2024, 7, 30), datetime.date(2024, 7, 31), datetime.date(2024, 8, 1), datetime.date(2024, 8, 2), datetime.date(2024, 8, 3), datetime.date(2024, 8, 4), datetime.date(2024, 8, 5), datetime.date(2024, 8, 6), datetime.date(2024, 8, 7), datetime.date(2024, 8, 8), datetime.date(2024, 8, 9), datetime.date(2024, 8, 10), datetime.date(2024, 8, 11), datetime.date(2024, 8, 12), datetime.date(2024, 8, 13), datetime.date(2024, 8, 14), datetime.date(2024, 8, 15), datetime.date(2024, 8, 16), datetime.date(2024, 8, 17), datetime.date(2024, 8, 18), datetime.date(2024, 8, 19), datetime.date(2024, 8, 20), datetime.date(2024, 8, 21), datetime.date(2024, 8, 22), datetime.date(2024, 8, 23), datetime.date(2024, 8, 24), datetime.date(2024, 8, 25), datetime.date(2024, 8, 26), datetime.date(2024, 8, 27), datetime.date(2024, 8, 28), datetime.date(2024, 8, 29), datetime.date(2024, 8, 30), datetime.date(2024, 8, 31), datetime.date(2024, 9, 1), datetime.date(2024, 9, 2), datetime.date(2024, 9, 3), datetime.date(2024, 9, 4), datetime.date(2024, 9, 5), datetime.date(2024, 9, 6), datetime.date(2024, 9, 7), datetime.date(2024, 9, 8), datetime.date(2024, 9, 9), datetime.date(2024, 9, 10), datetime.date(2024, 9, 11), datetime.date(2024, 9, 12), datetime.date(2024, 9, 13), datetime.date(2024, 9, 14), datetime.date(2024, 9, 15), datetime.date(2024, 9, 16), datetime.date(2024, 9, 17), datetime.date(2024, 9, 18), datetime.date(2024, 9, 19), datetime.date(2024, 9, 20), datetime.date(2024, 9, 21), datetime.date(2024, 9, 22), datetime.date(2024, 9, 23), datetime.date(2024, 9, 24), datetime.date(2024, 9, 25), datetime.date(2024, 9, 26), datetime.date(2024, 9, 27), datetime.date(2024, 9, 28), datetime.date(2024, 9, 29), datetime.date(2024, 9, 30), datetime.date(2024, 10, 1), datetime.date(2024, 10, 2), datetime.date(2024, 10, 3), datetime.date(2024, 10, 4), datetime.date(2024, 10, 5), datetime.date(2024, 10, 6), datetime.date(2024, 10, 7), datetime.date(2024, 10, 8), datetime.date(2024, 10, 9), datetime.date(2024, 10, 10), datetime.date(2024, 10, 11), datetime.date(2024, 10, 12), datetime.date(2024, 10, 13), datetime.date(2024, 10, 14), datetime.date(2024, 10, 15), datetime.date(2024, 10, 16), datetime.date(2024, 10, 17), datetime.date(2024, 10, 18), datetime.date(2024, 10, 19), datetime.date(2024, 10, 20), datetime.date(2024, 10, 21), datetime.date(2024, 10, 22), datetime.date(2024, 10, 23), datetime.date(2024, 10, 24), datetime.date(2024, 10, 25), datetime.date(2024, 10, 26), datetime.date(2024, 10, 27), datetime.date(2024, 10, 28), datetime.date(2024, 10, 29), datetime.date(2024, 10, 30), datetime.date(2024, 10, 31), datetime.date(2024, 11, 1), datetime.date(2024, 11, 2), datetime.date(2024, 11, 3), datetime.date(2024, 11, 4), datetime.date(2024, 11, 5), datetime.date(2024, 11, 6), datetime.date(2024, 11, 7), datetime.date(2024, 11, 8), datetime.date(2024, 11, 9), datetime.date(2024, 11, 10), datetime.date(2024, 11, 11), datetime.date(2024, 11, 12), datetime.date(2024, 11, 13), datetime.date(2024, 11, 14), datetime.date(2024, 11, 15), datetime.date(2024, 11, 16), datetime.date(2024, 11, 17), datetime.date(2024, 11, 18), datetime.date(2024, 11, 19), datetime.date(2024, 11, 20), datetime.date(2024, 11, 21), datetime.date(2024, 11, 22), datetime.date(2024, 11, 23), datetime.date(2024, 11, 24), datetime.date(2024, 11, 25), datetime.date(2024, 11, 26), datetime.date(2024, 11, 27), datetime.date(2024, 11, 28), datetime.date(2024, 11, 29), datetime.date(2024, 11, 30), datetime.date(2024, 12, 1), datetime.date(2024, 12, 2), datetime.date(2024, 12, 3), datetime.date(2024, 12, 4), datetime.date(2024, 12, 5), datetime.date(2024, 12, 6), datetime.date(2024, 12, 7), datetime.date(2024, 12, 8), datetime.date(2024, 12, 9), datetime.date(2024, 12, 10), datetime.date(2024, 12, 11), datetime.date(2024, 12, 12), datetime.date(2024, 12, 13), datetime.date(2024, 12, 14), datetime.date(2024, 12, 15), datetime.date(2024, 12, 16), datetime.date(2024, 12, 17), datetime.date(2024, 12, 18), datetime.date(2024, 12, 19), datetime.date(2024, 12, 20), datetime.date(2024, 12, 21), datetime.date(2024, 12, 22), datetime.date(2024, 12, 23), datetime.date(2024, 12, 24), datetime.date(2024, 12, 25), datetime.date(2024, 12, 26), datetime.date(2024, 12, 27), datetime.date(2024, 12, 28), datetime.date(2024, 12, 29)]  
n_days_used = len(valid_days)

print(f"有效天数: {n_days_used}")

# 主循环
for d in tqdm(valid_days, desc="Computing yearly averages"):
    # 读取预测数据
    with h5py.File(day_file(forecast_root, d), "r") as hf:
        fcst = hf["fields"][:].astype(np.float32)
    
    # 读取真值数据并拼接
    with h5py.File(day_file(truth_root, d), "r") as hf:
        truth_today = hf["fields"][:].astype(np.float32)
    with h5py.File(day_file(truth_root, d + timedelta(days=1)), "r") as hf:
        truth_next = hf["fields"][:].astype(np.float32)
    with h5py.File(day_file(truth_root, d + timedelta(days=2)), "r") as hf:
        truth_day_after_next = hf["fields"][:].astype(np.float32)
    
    # 正确的label对齐方式：d1[1:] + d2 + d3[0]
    truth48_30 = np.concatenate([
        truth_today[1:],      # 今天的1-23时
        truth_next,           # 明天的0-23时
        truth_day_after_next[0:1]  # 后天的0时
    ], axis=0)
    
    truth_24 = truth48_30[:, truth_idx_for_pred24, ...]
    
    # 裁剪
    fcst_c = crop_hw(fcst, pad=64)
    truth_c = crop_hw(truth_24, pad=64)
    
    # RMSE计算 - 与RMSE.py一致：先对每个样本计算RMSE
    mse_values = np.mean((fcst_c - truth_c) ** 2, axis=(-2, -1))  # 对每个样本计算MSE
    rmse_values = np.sqrt(mse_values)  # 对每个样本计算RMSE
    rmse_list.append(rmse_values)
    
    # ACC计算 - 与RMSE.py一致：先对每个样本计算ACC
    fcst_norm = (fcst_c - data_mean_24) / data_std_24
    truth_norm = (truth_c - data_mean_24) / data_std_24
    
    # 对每个样本计算ACC
    acc_values = np.sum((fcst_norm - mean_norm_c) * (truth_norm - mean_norm_c), axis=(-2, -1)) / \
                 np.sqrt(np.sum((fcst_norm - mean_norm_c) ** 2, axis=(-2, -1)) * 
                         np.sum((truth_norm - mean_norm_c) ** 2, axis=(-2, -1)))
    acc_list.append(acc_values)

# 计算年平均 - 与RMSE.py一致：对所有样本的RMSE和ACC求平均
mean_rmse_phy = np.mean(rmse_list, axis=0).astype(np.float32)
mean_acc = np.mean(acc_list, axis=0).astype(np.float32)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 保存结果
np.save(os.path.join(output_dir, "rmse_nwp.npy"), mean_rmse_phy)
np.save(os.path.join(output_dir, "acc_nwp.npy"), mean_acc)

print(f"\n完成。有效天数 = {n_days_used}")
print(f"年平均RMSE形状: {mean_rmse_phy.shape}")
print(f"年平均ACC形状: {mean_acc.shape}")