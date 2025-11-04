
# import os
# os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
# import h5py

# # # with h5py.File('/nfs/samba/数据聚变/气象数据/hrrr_440_24var_rb/train/2017/10/18.h5',"r") as f:
# # #     print(f['fields'][0, 0])


# path_ = '/nfs/samba/数据聚变/气象数据/hrrr_440_30var/train/2017/08/30.h5'
# # path_ = '/nfs/samba/数据聚变/气象数据/hrrr_440_30var/nwp/2024/09/10.h5'

# _file = h5py.File(path_, "r")
# out = _file["fields"][:]
# print(out[23][26])
import numpy as np
d = np.load("/nfs/samba/数据聚变/气象数据/hrrr_440_30var/traindata_stat/var_pos_mean.npy")
print(d[26])





# import pygrib
# import h5py
# import os
# # import glob
# import numpy as np
# from datetime import datetime, timedelta

# variables = {
#     '198':[0,'meanSea'], #meanSea
# } 

# def grib2h5(year, month, day):
#     yyyy = str(year).zfill(4)
#     mm   = str(month).zfill(2)
#     dd   = str(day).zfill(2)
#     ymd  = f"{yyyy}{mm}{dd}"

#     for hour in range(0,1):
#         hh = f"{hour:02d}"
#         file_path = (
#             f"/nfs/samba/数据聚变/气象数据/hrrr_grib/"
#             f"{yyyy}/{mm}/{dd}/"
#             f"hrrr_{ymd}_hrrr_t{hh}z_wrfprsf00.grib2"
#         )
#         gri = pygrib.open(file_path.encode('utf-8'))

#         for message in gri:
#             # 获取变量名称和层次值
#             variable_name = message.parameterName
#             level = message.level
#             level_type = message.typeOfLevel
#             # print("var_name:",variable_name, "level:",level, "level_type:",level_type)
#             # print(message.values.shape)
            
#             # 这里可以优化，直接去取就好了，不需要遍历
#             if variable_name in variables and level in variables[variable_name] and level_type in variables[variable_name]:
#                 print("var_name:",variable_name, "level:",level, "level_type:",level_type)
#                 print(message.values.shape)
#                 data = np.float32(message.values)[252:252+440,969:969+408]
#                 print(data)
#         gri.close()
       
# start_date = datetime(2017, 1, 1, 00)  # 设置开始日期
# current_date = start_date
# grib2h5(current_date.year, current_date.month, current_date.day)




# grib2h5(2023,1,1)
  
        # a = [0, 5, 10, 19, 1, 6, 11, 15, 2, 7, 12, 16, 3, 8, 13, 17, 4, 9, 14, 18, 20,21, 22, 23, 24,25,26,27,28,29]
        # n_outdata = outdata[:, a, :, :]
        
    # output_file_path = '/nfs/samba/数据聚变/气象数据/hrrr_440_30var/nwp/'+yyyy+'/'+mm+'/'
    
    # if not os.path.exists(output_file_path):
    #     os.makedirs(output_file_path)
    
    # output_file_path = output_file_path + dd+'.h5'import h5py

    
    

    # with h5py.File(output_file_path, 'w') as h5_file:
    #     # 创建一个名为'data'的数据集，将数据写入其中
    #     h5_file.create_dataset('fields', data = n_outdata) 
