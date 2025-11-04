import torch
from models.afno_attention_parallel import AFNOAttnParallelNet
import torch.distributed as dist
from model_train import ModelTrain
import argparse
import sys
import os
import logging
import yaml
logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

if __name__ == "__main__":
    print('begin train')
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", help="local device id on current node",
    #                     type=int)
    # args = parser.parse_args()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,3,4,6,7'
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')
    n_gpus = 2
    dist.init_process_group("nccl", world_size=n_gpus, rank=local_rank)
    # torch.cuda.set_device(local_rank)
    # world_size = dist.get_world_size()
    if torch.cuda.is_available():
        logging.warning("Cuda is available!")
        if torch.cuda.device_count() > 1:
            logging.warning(f"Find {torch.cuda.device_count()} GPUs!")
        else:
            logging.warning("Too few GPU!")
    else:
        logging.warning("Cuda is not available! Exit!") 
    
    model = AFNOAttnParallelNet(
        img_size=(440, 408),
        in_channels=25,  # 24个气象变量 + 1个地理常量
        out_channels=24,  # 输出24个变量
        num_timestamps=1,
        attn_channel_ratio=0.20
    )
    config = '/home/pengjian/Cas_Former_24var_440/config.yaml'
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    Train = ModelTrain(model,config,local_rank)
    Train.train(40)
    dist.destroy_process_group()  # 释放资源
    
    