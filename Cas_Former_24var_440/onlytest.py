import torch
from models.afno_attention_parallel import AFNOAttnParallelNet
import torch.distributed as dist
from model_train import ModelTrain
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
    # 选择使用的gpu
    local_rank = 5
    model = AFNOAttnParallelNet(
        img_size=(440, 408),
        in_channels=25,  # 24个气象变量 + 1个地理常量
        out_channels=24,  # 输出24个变量
        num_timestamps=1,
        attn_channel_ratio=0.20
    )
    config = '/home/pengjian/CAS_Former_440/config_onlytest.yaml'
    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    Train = ModelTrain(model,config,local_rank)
    Train.train(1)
    