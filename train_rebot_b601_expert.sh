#!/bin/bash
# REBOT B601 单臂 red_cube 数据集动作专家后训练脚本(基于 lingbot-vla-4b 基模,仅训练 expert)
# 数据集: /home/hcy/data/lerobot-dataset/rebot_v060_v1_red_cube (40 ep / 21702 帧, 单臂 7 维, 2 RGB 相机)
# 方案: 单卡 4090 48GB (GPU 4), micro 8 x accum 16 = batch 128, 峰值 ~35GB(so101 已验证)
# 用法:
#   正式训练:          bash train_rebot_b601_expert.sh
#   发烟测试:          bash train_rebot_b601_expert.sh --train.max_steps 20 --train.save_steps 10
#   命令行覆盖:        bash train_rebot_b601_expert.sh --train.max_steps 3000
set -e

export TOKENIZERS_PARALLELISM=false
export PATH=/data1/hcy/miniconda3/envs/lingbotvla/bin:$PATH
# 关键:强制使用当前项目的 lingbotvla,避免误导入 site-packages 里的其他版本
export PYTHONPATH=/data1/hcy/Projects/lingbot-vla-hcy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 62500/62700/62800 被其他训练占用,用 62900(可被环境变量覆盖)
export MASTER_PORT=${MASTER_PORT:-62900}

CONFIG=./configs/vla/rebot_b601_expert.yaml
NORM_FILE=assets/norm_stats/rebot_b601.json

# 1) 计算新数据集的归一化统计(仅首次;失败可 --skip_norm 跳过)
if [ "$1" != "--skip_norm" ] && [ ! -f "$NORM_FILE" ]; then
  echo ">>> 计算 norm stats: $NORM_FILE"
  LOCAL_RANK=0 RANK=0 WORLD_SIZE=1 python scripts/compute_norm.py "$CONFIG"
fi

# 2) 单卡 expert-only 训练(GPU 4)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
bash train.sh tasks/vla/train_lingbotvla.py "$CONFIG" "$@"
