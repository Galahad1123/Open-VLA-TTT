#!/usr/bin/env bash
# Default is a short wiring check. Formal training requires explicit --train.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
VLM_DIR=/home/hello/.cache/huggingface/hub/models--OpenGVLab--InternVL3-1B-hf/snapshots/014c0583a0d4bedf29fbe2dbff4f865eb998e171
case "${1:---smoke}" in
  --smoke) RUN=robomme-evo1-groot-preflight-20261008; STEPS1=3; STEPS2=3; SAVE1=3; SAVE2=3; WARMUP=1; LOG=1 ;;
  --train) RUN=robomme-evo1-groot-scratch-seed1000; STEPS1=5000; STEPS2=80000; SAVE1=2500; SAVE2=10000; WARMUP=1000; LOG=50 ;;
  *) echo "Usage: bash $0 [--smoke|--train]" >&2; exit 2 ;;
esac
S1="outputs/${RUN}-stage1"
S2="outputs/${RUN}-stage2"
# Never overwrite an existing run or silently resume.
[[ ! -e "$S1" && ! -e "$S2" ]] || { echo "Output already exists; inspect it before a new run" >&2; exit 1; }
[[ -d "$VLM_DIR" && -f /home/hello/lizk/datasets/robomme/meta/info.json ]]
mkdir -p outputs/logs
TRAIN_COMMON=(
  --dataset.repo_id=lerobot/robomme
  --dataset.root=/home/hello/lizk/datasets/robomme
  --dataset.revision=92796abbc67db4ea6447e5031d04bf47aa3770bb
  --dataset.video_backend=pyav --dataset.return_uint8=true
  --dataset.image_transforms.enable=false --dataset.use_imagenet_stats=false --dataset.eval_split=0.0
  --policy.vlm_model_name="$VLM_DIR" --policy.apply_training_stage_defaults=true
  --policy.device=cuda --policy.use_amp=true --policy.dtype=bfloat16 --policy.use_flash_attn=false
  --policy.enable_gradient_checkpointing=true --policy.gradient_checkpointing_use_reentrant=false
  --policy.image_resolution='[448,448]' --policy.chunk_size=20 --policy.n_action_steps=10
  --policy.max_state_dim=24 --policy.max_action_dim=24 --policy.task_field=task
  --policy.postprocess_action_dim=8 --policy.gripper_index=7 --policy.binarize_gripper=false
  --policy.dropout=0.2 --policy.optimizer_lr=1e-5 --policy.optimizer_weight_decay=1e-3
  --policy.optimizer_grad_clip_norm=1.0 --policy.scheduler_warmup_steps="$WARMUP"
  --policy.push_to_hub=false --use_policy_training_preset=true --resume=false
  --save_checkpoint=true --save_checkpoint_to_hub=false --log_freq="$LOG" --env_eval_freq=0
  --num_workers=4 --prefetch_factor=2 --persistent_workers=true
  --seed=1000 --cudnn_deterministic=true --wandb.enable=false
)
# Explicit local factory routing; the standard LIBERO entry point is unchanged.
.venv/bin/vla-ttt-train-robomme \
  "${TRAIN_COMMON[@]}" --policy.type=evo1_groot --policy.training_stage=stage1 \
  --policy.vlm_dtype=bfloat16 --batch_size=32 --steps="$STEPS1" --save_freq="$SAVE1" \
  --output_dir="$S1" 2>&1 | tee "outputs/logs/${RUN}-stage1.log"
printf -v CKPT1 '%06d' "$STEPS1"
.venv/bin/vla-ttt-train-robomme \
  "${TRAIN_COMMON[@]}" --policy.path="$S1/checkpoints/$CKPT1/pretrained_model" \
  --policy.training_stage=stage2 --policy.vlm_dtype=float32 --batch_size=16 \
  --steps="$STEPS2" --save_freq="$SAVE2" --output_dir="$S2" \
  2>&1 | tee "outputs/logs/${RUN}-stage2.log"
