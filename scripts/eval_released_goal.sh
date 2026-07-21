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
  --pretrained_checkpoint /root/autodl-tmp/models/openvla-7b-oft-finetuned-libero-goal \
  --task_suite_name libero_goal \
  --num_trials_per_task 1 \
  --center_crop True
