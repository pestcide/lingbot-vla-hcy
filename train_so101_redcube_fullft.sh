#!/bin/bash
# SO-101 red_cube 数据集全量微调脚本(基于 lingbot-vla-4b 基模)
# 数据集: /data1/hcy/lerobot-dataset/so101_ubuntu_bimanual_red_cube (100 ep / 63085 帧, 双臂 12 维)
# 显存方案: 3 卡 4090 48GB (GPU 0,1,2) FSDP 全分片 —— 单卡全参必 OOM(4B x Adam fp32 32GB)
# 用法:
#   全流程(自动算 norm + 训练): bash train_so101_redcube_fullft.sh
#   跳过 norm 重新训练:          bash train_so101_redcube_fullft.sh --skip_norm
#   命令行覆盖:                  bash train_so101_redcube_fullft.sh --train.max_steps 8000
set -e

export TOKENIZERS_PARALLELISM=false
export PATH=/data1/hcy/miniconda3/envs/lingbotvla/bin:$PATH
# 关键:强制使用当前项目的 lingbotvla,避免误导入 site-packages 里的其他版本
export PYTHONPATH=/data1/hcy/Projects/lingbot-vla-hcy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 62500 被残留 pt_elastic 进程占用,换默认端口(可被环境变量覆盖)
export MASTER_PORT=${MASTER_PORT:-62700}

CONFIG=./configs/vla/so101_red_cube_fullft.yaml
NORM_FILE=assets/norm_stats/so101_red_cube.json
GPUS=${GPUS:-0,1,2}   # 当前空闲卡:GPU 0/1/2(48GB x3)

# 1) 计算新数据集的归一化统计(仅首次;失败可 --skip_norm 跳过)
if [ "$1" != "--skip_norm" ] && [ ! -f "$NORM_FILE" ]; then
  echo ">>> 计算 norm stats: $NORM_FILE"
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 python scripts/compute_norm.py "$CONFIG"
fi

# 2) 3 卡 FSDP 全量微调
CUDA_VISIBLE_DEVICES=$GPUS \
bash train.sh tasks/vla/train_lingbotvla.py "$CONFIG" "$@"
