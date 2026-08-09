#!/bin/bash
# SO-101 双臂后训练续训脚本:从最新检查点(global_step_9400)继续训练,加长训练轮次
# 背景:第一次续训(1e-5 cosine)已收敛到 loss 0.081,开环测试形状正确但精度有余量
#       → 从 9400 继续低 LR 打磨(cosine 按新 max_steps 重算,续训起步 LR≈3.1e-6)
# 用法:
#   默认续训(从 OUTPUT_DIR 最新检查点继续):  bash train_so101_cont.sh
#   自定义: CKPT=xxx LR=5e-6 MAX_STEPS=20000 bash train_so101_cont.sh
#   命令行覆盖(优先级最高):                  bash train_so101_cont.sh --train.max_steps 20000
set -e

export TOKENIZERS_PARALLELISM=false
export PATH=/data1/hcy/miniconda3/envs/lingbotvla/bin:$PATH
# 关键:强制使用当前项目的 lingbotvla,避免误导入 site-packages 里的其他版本
export PYTHONPATH=/data1/hcy/Projects/lingbot-vla-hcy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 默认参数(可被环境变量覆盖)
# 注意:默认【不】传 load_checkpoint_path → 走 yaml 的 enable_resume: true,
# 自动扫描 OUTPUT_DIR/checkpoints/ 里最新的 global_step_* 继续(如 global_step_9400)。
# 若必须从指定检查点续,才设置 CKPT=xxx(CKPT 优先于自动扫描)。
LR=${LR:-1e-5}                                        # 原 5e-5 → 1e-5(5 倍降)
LR_DECAY_STYLE=${LR_DECAY_STYLE:-cosine}              # 可选 cosine(自动退火到 lr_min=1e-7)
OUTPUT_DIR=${OUTPUT_DIR:-output/so101_bi_expert_cont} # 续训输出目录(已含 9400 检查点,自动扫描续)
MAX_STEPS=${MAX_STEPS:-15000}                         # 加长训练:9400 → 15000(续训 5600 步 ≈ 28h)

LOAD_CKPT_ARG=""
if [ -n "${CKPT:-}" ]; then
  LOAD_CKPT_ARG="--train.load_checkpoint_path $CKPT"
fi

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4} \
bash train.sh tasks/vla/train_lingbotvla.py ./configs/vla/so101_bi_load20000h.yaml \
  --train.output_dir "$OUTPUT_DIR" \
  $LOAD_CKPT_ARG \
  --train.lr "$LR" \
  --train.lr_decay_style "$LR_DECAY_STYLE" \
  --train.max_steps "$MAX_STEPS" \
  "$@"
