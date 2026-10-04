#!/usr/bin/env bash
# M4 n5 单模态对照组 ACT 训练（12 条，每变体 1 风格）
# 用法: STEPS=40000 bash policy/act_lerobot/train_transfer_uni.sh
set -e
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
PY=/home/lifd/anaconda3/envs/lerobot-env/bin/python

STEPS=${STEPS:-40000}
OUT=${OUT:-checkpoints/act_m4_uni}
BS=${BS:-8}

RESUME_ARGS=""
LATEST_CFG=$(ls $OUT/checkpoints/[0-9]*/pretrained_model/train_config.json 2>/dev/null | sort | tail -1)
if [ -n "$LATEST_CFG" ]; then
  RESUME_ARGS="--resume=true --config_path=$LATEST_CFG"
  echo "resume from $LATEST_CFG"
fi

$PY -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/bimanual_transfer_uni_train \
  --dataset.root=data/bimanual_transfer_uni_train \
  --policy.type=act \
  --policy.chunk_size=32 \
  --policy.n_action_steps=16 \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --batch_size=$BS \
  --num_workers=4 \
  --steps=$STEPS \
  --eval_freq=0 \
  --save_freq=5000 \
  --log_freq=50 \
  --output_dir=$OUT \
  --job_name=act_m4_uni \
  --seed=1000 \
  $RESUME_ARGS
