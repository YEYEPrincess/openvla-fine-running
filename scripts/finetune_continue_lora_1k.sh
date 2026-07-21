#!/bin/bash

cd /root/autodl-tmp/openvla_project/openvla-oft

unset PYTHONPATH
export PYTHONPATH=/root/autodl-tmp/openvla_project/openvla-oft/LIBERO:$PYTHONPATH

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export WANDB_MODE=offline
export TMPDIR=/root/autodl-tmp/tmp

mkdir -p /root/autodl-tmp/tmp
mkdir -p /root/autodl-tmp/openvla_project/runs

torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \
  --vla_path /root/autodl-tmp/models/openvla-7b-oft-finetuned-libero-goal \
  --data_root_dir /root/autodl-tmp/openvla_project/modified_libero_rlds \
  --dataset_name libero_goal_no_noops \
  --run_root_dir /root/autodl-tmp/openvla_project/runs/libero_goal_continue_lora_r32_1k \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 2 \
  --use_proprio True \
  --batch_size 1 \
  --learning_rate 5e-5 \
  --num_steps_before_decay 1000 \
  --max_steps 1000 \
  --save_freq 1000 \
  --save_latest_checkpoint_only True \
  --image_aug True \
  --lora_rank 32 \
  --wandb_project openvla-libero \
  --run_id_note continue_goal_ckpt_lora_r32_1k_lr5e-5
