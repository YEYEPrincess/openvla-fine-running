#!/bin/bash

cd /root/autodl-tmp/openvla_project/openvla-oft

unset PYTHONPATH
export PYTHONPATH=/root/autodl-tmp/openvla_project/openvla-oft/LIBERO:$PYTHONPATH

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /root/autodl-tmp/openvla_project/runs/libero_goal_continue_lora_r32_1k/openvla-7b-oft-finetuned-libero-goal+libero_goal_no_noops+b1+lr-5e-05+lora-r32+dropout-0.0--image_aug--continue_goal_ckpt_lora_r32_1k_lr5e-5 \
  --task_suite_name libero_goal \
  --num_trials_per_task 5 \
  --center_crop True
