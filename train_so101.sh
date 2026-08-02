#!/bin/bash
# SO-101 双臂后训练启动脚本(expert-only 方案,单卡 GPU 4)
# 用法:
#   正式训练: bash train_so101.sh
#   发烟测试: bash train_so101.sh --train.max_steps 100 --train.save_steps 100 --train.output_dir output/so101_bi_smoke_test
set -e

export TOKENIZERS_PARALLELISM=false
export PATH=/data1/hcy/miniconda3/envs/lingbotvla/bin:$PATH
# 关键:强制使用当前项目的 lingbotvla,避免误导入 site-packages 里的其他版本
export PYTHONPATH=/data1/hcy/Projects/lingbot-vla-hcy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/so101_bi_load20000h.yaml "$@"
