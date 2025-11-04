import os
import numpy as np
import h5py
import torch
def saveLog(i,epoch,end_time,start_time,loss,optimizer,OUTPUT_DIR):
     if(i%10==0):
            print("epoch: ",epoch,"iter: ",i,"迭代时间：", (end_time - start_time) , "秒","loss:",loss.item(),"lr: ",optimizer.param_groups[0]['lr'] )
            log_file_path = OUTPUT_DIR+'/train.log'
            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            with open(log_file_path, 'a') as log_file:
                # 您要记录的文本
                log_text = "epoch: {} iter: {} 迭代时间： {} 秒 loss: {} lr: {}\n".format(epoch, i, (end_time - start_time), loss.item(), optimizer.param_groups[0]['lr'])
                
                # 将文本写入日志文件
                log_file.write(log_text)

  
def save_np(result,path):
       

        # 指定要保存的txt文件路径
        # file_path = './output/hrrr_finetune/test/inf_output_data.h5'
        file_path = './output/'+path+'.h5'

    

        # 将Paddle张量转换为NumPy数组
        numpy_array = result.cpu().numpy()

        # 保存NumPy数组到H5文件
        h5_file = h5py.File(file_path, 'w')
        h5_file.create_dataset('fields', data=numpy_array)
        h5_file.close()