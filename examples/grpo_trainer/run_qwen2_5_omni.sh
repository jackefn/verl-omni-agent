#!/bin/bash
# 文件名: test_qwen_omni_3b_gpro_multimodal.sh
set -x
export CUDA_VISIBLE_DEVICES=4,5,6,7
OMNI_TRAIN_PATH=/mnt/hpfs/xiangc/mxy/verl-agent/video_data/train.parquet
OMNI_TEST_PATH=/mnt/hpfs/xiangc/mxy/verl-agent/video_data/test.parquet
train_files="['$OMNI_TRAIN_PATH']"
test_files="['$OMNI_TEST_PATH']"
OMNI_MODEL_PATH="/mnt/hpfs/xiangc/llms/Qwen2.5-Omni-3B"
export C_INCLUDE_PATH=/usr/local/cuda/include:$C_INCLUDE_PATH

export WANDB_MODE=offline
ray start --head --port=6391
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=4 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.video_key=videos \
    data.audio_key=audios \
    env.env_name=video_retrieval \
    env.rollout.n=1 \
    data.use_audio_in_video=True \
    actor_rollout_ref.model.path=$OMNI_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='verl_grpo_qwen_omni_test' \
    trainer.experiment_name='qwen25_omni_3b_multimodal' \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 $@