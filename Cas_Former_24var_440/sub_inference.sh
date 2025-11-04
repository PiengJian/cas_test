/home/gty/.conda/envs/py39_torch/bin/torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12366 inference.py
#/home/gty/.conda/envs/py39_torch/bin/torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12366 inference_analysis.py
#/home/gty/.conda/envs/py39_torch/bin/torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12336 inference_2branch.py
#/home/gty/.conda/envs/py39_torch/bin/torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12366 inference_real.py
#torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=12366 inference_marge.py
