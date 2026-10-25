#!/bin/bash
# GRPO (off-policy) baseline: lr=1e-6, mini_batch=8, n=8, batch=32
export VLLM_USE_V1=1
set -e

CONFIG_NAME="baseline_grpo"
DATA_PATH="datasets/sciknoweval/chemistry/"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
MINI_BATCH_SIZE=8
LR=1e-6
TOTAL_EPOCHS=7
MODEL_PATH=Qwen3/Qwen3-8B"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export REPO_ROOT
LOG_DIR="$REPO_ROOT/experiments_logs/sepo/chemistry"

export WANDB_MODE=disabled
export VERL_FILE_LOGGER_ROOT="$LOG_DIR"

EXP_NAME="grpo-che-offpolicy-lr${LR}-mb${MINI_BATCH_SIZE}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=SEPO-chemistry \
trainer.save_freq=0 \
trainer.save_best=False \
trainer.save_latest=False \
trainer.total_epochs=$TOTAL_EPOCHS \
actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.rollout.val_kwargs.n=16 \
actor_rollout_ref.rollout.enforce_eager=True \
custom_reward_function.path=$REPO_ROOT/verl/utils/reward_score/feedback/__init__.py"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${EXP_NAME}.log"

echo "Starting: $EXP_NAME"
echo "Log: $LOG_FILE"
bash "training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS 2>&1 | tee "$LOG_FILE"
