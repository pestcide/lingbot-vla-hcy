#!/bin/bash
# SO-101 双臂后训练开环测试(expert-only 方案,用 global_step_3000 检查点)
# 用法:
#   默认(测轨迹 0-4):     bash eval_so101_openloop.sh
#   指定轨迹:             bash eval_so101_openloop.sh --traj_ids 0 5 10
#   指定检查点/数据/输出:  bash eval_so101_openloop.sh --model_path output/checkpoints/global_step_2000/hf_ckpt --data_path /path/to/val --save_plot_path ./open_loop_test/v2
set -e

export TOKENIZERS_PARALLELISM=false
export PATH=/data1/hcy/miniconda3/envs/lingbotvla/bin:$PATH
# 关键:强制使用当前项目的 lingbotvla,避免误导入 site-packages 里的其他版本
export PYTHONPATH=/data1/hcy/Projects/lingbot-vla-hcy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 本地 Qwen2.5-VL-3B-Instruct snapshot,避免从 HF 下载
export QWEN25_PATH=/data1/hcy/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

# 默认参数(可被命令行覆盖)
MODEL_PATH=${MODEL_PATH:-output/checkpoints/global_step_3000/hf_ckpt}
ROBO_NAME=${ROBO_NAME:-so101_bi}
NORM_PATH=${NORM_PATH:-assets/norm_stats/so101_bi.json}
DATA_PATH=${DATA_PATH:-/data1/hcy/lerobot-dataset/so101_bi_150}
TRAJ_IDS=${TRAJ_IDS:-"0 1 2 3 4"}
USE_LENGTH=${USE_LENGTH:-50}
NUM_DENOISING_STEP=${NUM_DENOISING_STEP:-10}
SAVE_PLOT_PATH=${SAVE_PLOT_PATH:-./open_loop_test/global_step_3000}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
python scripts/open_loop_eval.py \
  --model_path "$MODEL_PATH" \
  --robo_name "$ROBO_NAME" \
  --norm_path "$NORM_PATH" \
  --data_path "$DATA_PATH" \
  --traj_ids $TRAJ_IDS \
  --use_length "$USE_LENGTH" \
  --num_denoising_step "$NUM_DENOISING_STEP" \
  --save_plot_path "$SAVE_PLOT_PATH"
