# OpenVLA-OFT Fine-Tuning and Evaluation on LIBERO

This project reproduces pretrained Vision-Language-Action (VLA) evaluation and explores LoRA behavior-cloning fine-tuning with [OpenVLA-OFT](https://github.com/moojink/openvla-oft) on [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO).

The central finding is that **lower supervised action-regression loss does not necessarily produce higher closed-loop task success**. Fine-tuning from OpenVLA-7B reduced training loss but produced no successful LIBERO-Goal rollouts in the tested budgets. Continued fine-tuning from a task-adapted checkpoint retained some behavior, but degraded substantially relative to the released policy.

## Project status

| Objective | Status |
|---|---|
| Pretrained VLA inference | Completed |
| LoRA fine-tuning on LIBERO demonstrations | Completed |
| Seen-task evaluation | Completed |
| Unseen-object generalization | Partially explored; no controlled cross-suite study |
| LoRA vs full/partial fine-tuning | LoRA variants compared; full/partial FT not run |
| Success rate | Completed |
| Per-step inference latency | Not completed |

## Method

The policy maps RGB observations, a language instruction, and proprioception to chunks of continuous robot actions. Evaluation is closed-loop: observe, predict, execute, receive a new observation, and repeat until success or timeout.

Fine-tuning uses offline behavior cloning on the RLDS dataset libero_goal_no_noops with L1 action regression. LoRA rank 32 updates low-rank adapters rather than all 7.6B parameters.

## Environment

- Python 3.10
- PyTorch 2.2.0 + CUDA 12.1
- TensorFlow 2.15.0
- NumPy 1.26.4
- Transformers 4.40.1 (OpenVLA-OFT fork)
- LIBERO, robosuite, and MuJoCo
- Weights & Biases 0.16.6 in offline mode

Typical settings include PYTHONPATH pointing to LIBERO, MUJOCO_GL=egl, PYOPENGL_PLATFORM=egl, TOKENIZERS_PARALLELISM=false, OMP_NUM_THREADS=1, and WANDB_MODE=offline.

The scripts retain paths from the original AutoDL setup. Update local paths before running them elsewhere. Model weights, datasets, checkpoints, W&B runs, and full rollout collections are intentionally excluded.

## Results

### Released checkpoint evaluation

| Checkpoint | Suite | Trials/task | Episodes | Successes | Success rate |
|---|---|---:|---:|---:|---:|
| Released OpenVLA-OFT Spatial | LIBERO-Spatial | 50 | 500 | 489 | **97.8%** |
| Released OpenVLA-OFT Object | LIBERO-Object | 50 | 500 | 485 | **97.0%** |
| Released OpenVLA-OFT Goal | LIBERO-Goal | 50 | 500 | 489 | **97.8%** |

The released Goal checkpoint also achieved 10/10 in a sanity check. These are matched suite evaluations and should not be interpreted as controlled cross-suite unseen-object generalization.

### LoRA from OpenVLA-7B

| Initialization | Steps | Initial LR | Image aug. | Final loss | Evaluation |
|---|---:|---:|---:|---:|---:|
| OpenVLA-7B | 1,000 | 5e-4 | Yes | 0.3867 | **0/10 (0%)** |
| OpenVLA-7B | 5,000 | 5e-4 | Yes | 0.2539 | **0/10 (0%)** |

Despite lower offline action loss, neither model produced a successful closed-loop rollout.

### Continued LoRA fine-tuning

| Initialization | Steps | Initial LR | Image aug. | Final loss | Evaluation |
|---|---:|---:|---:|---:|---:|
| Released Goal | 1,000 | 5e-5 | Yes | 0.0786 | **4/10 (40%)** |
| Released Goal | 1,000 | 5e-5 | Yes | 0.0786 | **16/50 (32%)** |
| Released Goal | 500 | 1e-5 | No | 0.1270 | **0/10 (0%)** |

Starting from a task-adapted checkpoint retained more executable behavior than starting from the base model, but still caused severe policy degradation. Raw results are in [results/success_rates.csv](results/success_rates.csv).

## Failure analysis

Observed failures included inaccurate approach or grasp poses, unstable gripper timing, accumulated action-chunk errors, and failure to recover after entering states absent from expert demonstrations.

LIBERO-Object failures were concentrated on particular objects:

| Object | Failures / 50 | Success rate |
|---|---:|---:|
| BBQ sauce | 4 | 92% |
| Salad dressing | 3 | 94% |
| Chocolate pudding | 3 | 94% |
| Milk | 2 | 96% |
| Cream cheese | 1 | 98% |
| Butter | 1 | 98% |
| Tomato sauce | 1 | 98% |

## Selected rollout videos

Three qualitative failure examples belong in videos_selected:

- [Wine bottle on cabinet](videos_selected/put_the_wine_bottle_on_top_of_the_cabinet_failure.mp4)
- [Push plate toward stove](videos_selected/push_the_plate_to_the_front_of_the_stove_failure.mp4)
- [Turn on stove](videos_selected/turn_on_the_stove_failure.mp4)

The complete rollout collection is excluded because it is large.

## Reproduction scripts

| Script | Purpose |
|---|---|
| [eval_released_goal.sh](scripts/eval_released_goal.sh) | Released Goal sanity check |
| [finetune_base_lora_5k.sh](scripts/finetune_base_lora_5k.sh) | LoRA rank-32 training from OpenVLA-7B |
| [finetune_continue_lora_1k.sh](scripts/finetune_continue_lora_1k.sh) | Continued LoRA training from released Goal |
| [eval_continue_lora_1k.sh](scripts/eval_continue_lora_1k.sh) | Evaluation of the continued checkpoint |

## Key findings

1. **Closed-loop evaluation is essential.** Low demonstration loss did not guarantee interactive success.
2. **Initialization matters.** A task-adapted checkpoint retained more behavior over the tested budgets.
3. **Fine-tuning can degrade a strong policy.**
4. **Robot-policy errors compound over time.** Small errors move the policy into unfamiliar states.

## Limitations

- Full fine-tuning was not run due to single-GPU compute and storage constraints.
- Partial fine-tuning was not implemented as a separate baseline.
- Systematic cross-suite unseen-object generalization was not completed.
- Per-step inference latency was not measured; episode time includes simulation and encoding.
- The hyperparameter sweep was small and used only LIBERO-Goal demonstrations.
- Evaluation was simulation-only.

## Future work

- Measure synchronized per-step latency (mean, median, p95, and throughput).
- Compare LoRA ranks 8, 16, 32, and 64.
- Add partial fine-tuning baselines.
- Evaluate cross-suite transfer with compatible action normalization.
- Use validation rollouts, replay, and regularization to reduce degradation.
- Explore DAgger to address expert-to-policy state distribution shift.

## Conclusion

This project reproduced OpenVLA-OFT inference, evaluated released policies over 1,500 formal rollouts, and completed multiple LoRA experiments. The results show that supervised behavior-cloning loss may improve while closed-loop manipulation remains poor or degrades. Robust VLA adaptation therefore requires interactive evaluation and explicit treatment of compounding error and distribution shift.

## Acknowledgements

Built on OpenVLA-OFT, OpenVLA, LIBERO, robosuite, Hugging Face, and Weights & Biases.