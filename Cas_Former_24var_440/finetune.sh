export OMP_NUM_THREADS=4
export NCCL_IB_DISABLE=1
# 指定可见 GPU 列表
export CUDA_VISIBLE_DEVICES=0,7

# 启动分布式训练
torchrun \
  --nproc_per_node=2 \
  --master_port=12355 \
  Cas_Former_24var_440/finetune.py