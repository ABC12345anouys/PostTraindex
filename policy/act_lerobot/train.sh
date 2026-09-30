#!/usr/bin/env bash
# M2 ACT 训练（lerobot-env / mujoco 3.7.0 物理，与录数和闭环评测一致）
# 用法:
#   STEPS=5000 bash policy/act_lerobot/train.sh                 # 新跑/续跑到 5000 步
#   STEPS=20000 bash policy/act_lerobot/train.sh                # 已有 checkpoint 自动 resume
#   STEPS=20000 CKPT=checkpoints/act_m2/checkpoints/000050 bash policy/act_lerobot/train.sh
set -e
cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1          # 离线：本地数据集，勿查 HF hub
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
PY=/home/lifd/anaconda3/envs/lerobot-env/bin/python

STEPS=${STEPS:-20000}
OUT=${OUT:-checkpoints/act_m2}
BS=${BS:-8}

RESUME_ARGS=""
LATEST_CFG=$(ls $OUT/checkpoints/[0-9]*/pretrained_model/train_config.json 2>/dev/null | sort | tail -1)
if [ -n "$LATEST_CFG" ]; then
  RESUME_ARGS="--resume=true --config_path=$LATEST_CFG"
  echo "resume from $LATEST_CFG"
fi

$PY -m lerobot.scripts.lerobot_train \
  --dataset.repo_id=local/bimanual_pen_train \
  --dataset.root=data/bimanual_pen_train \
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
  --job_name=act_m2 \
  --seed=1000 \
  $RESUME_ARGS
