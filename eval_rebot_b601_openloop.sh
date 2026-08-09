#!/bin/bash
# REBOT B601 单臂 expert-only 后训练开环测试(用 global_step_10000 最终检查点)
# 用法:
#   默认(测轨迹 0-4):   bash eval_rebot_b601_openloop.sh
#   指定轨迹:            bash eval_rebot_b601_openloop.sh --traj_ids 0 10 20
#   指定检查点/数据/输出: bash eval_rebot_b601_openloop.sh --model_path output/rebot_b601_expert/checkpoints/global_step_3000/hf_ckpt --save_plot_path ./open_loop_test/v2
set -e

export TOKENIZERS_PARALLELISM=false
export PATH=/data1/hcy/miniconda3/envs/lingbotvla/bin:$PATH
# 关键:强制使用当前项目的 lingbotvla,避免误导入 site-packages 里的其他版本
export PYTHONPATH=/data1/hcy/Projects/lingbot-vla-hcy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 本地 Qwen2.5-VL-3B-Instruct snapshot,避免从 HF 下载
export QWEN25_PATH=/data1/hcy/cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3

# 默认参数(可被命令行覆盖)
MODEL_PATH=${MODEL_PATH:-output/rebot_b601_expert/checkpoints/global_step_10000/hf_ckpt}
ROBO_NAME=${ROBO_NAME:-rebot_b601}
NORM_PATH=${NORM_PATH:-assets/norm_stats/rebot_b601.json}
DATA_PATH=${DATA_PATH:-/home/hcy/data/lerobot-dataset/rebot_v060_v1_red_cube}
TRAJ_IDS=${TRAJ_IDS:-"0 1 2 3 4"}
USE_LENGTH=${USE_LENGTH:-50}
NUM_DENOISING_STEP=${NUM_DENOISING_STEP:-10}
SAVE_PLOT_PATH=${SAVE_PLOT_PATH:-./open_loop_test/rebot_b601_global_step_10000}

# 命令行传入的 --traj_ids / --save_plot_path 优先;未指定才用上面的环境变量默认值
EXTRA_ARGS=()
if [[ "$*" != *"--traj_ids"* ]]; then EXTRA_ARGS+=(--traj_ids $TRAJ_IDS); fi
if [[ "$*" != *"--save_plot_path"* ]]; then EXTRA_ARGS+=(--save_plot_path "$SAVE_PLOT_PATH"); fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
python scripts/open_loop_eval.py \
  --model_path "$MODEL_PATH" \
  --robo_name "$ROBO_NAME" \
  --norm_path "$NORM_PATH" \
  --data_path "$DATA_PATH" \
  --use_length "$USE_LENGTH" \
  --num_denoising_step "$NUM_DENOISING_STEP" \
  "${EXTRA_ARGS[@]}" "$@"
