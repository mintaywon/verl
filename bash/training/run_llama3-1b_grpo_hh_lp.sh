#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=grpo                                                                                                                                
#SBATCH --output=/home/taywonmin/slurm-logs/test-%j.log  # log                                                                                                   
#SBATCH --error=/home/taywonmin/slurm-logs/test-%j.log   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:4   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행   
#SBATCH --exclude=node7

cd $HOME/verl

unset ROCR_VISIBLE_DEVICES
export HYDRA_FULL_ERROR=1

eval "$(/opt/miniconda3/condabin/conda shell.bash hook)"
conda activate verl

set -x

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.01 \
    data.return_raw_chat=True \
    data.return_raw_input_ids=True \
    data.train_files=$HOME/data/helpfulness_hh_rlhf/rl/train.parquet \
    data.val_files=$HOME/data/helpfulness_hh_rlhf/rl/train.parquet \
    data.train_batch_size=128 \
    data.prompt_key=prompt \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-1B-Instruct \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    +actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.65 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.model.enable_gradient_checkpointing=True \
    critic.model.enable_activation_offload=True \
    reward_model.enable=True \
    reward_model.model.path=Skywork/Skywork-Reward-V2-Llama-3.1-8B \
    reward_model.model.trust_remote_code=True \
    reward_model.micro_batch_size_per_gpu=1 \
    +reward_model.length_penalty_factor=0.01 \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='grpo_reward_hacking' \
    trainer.experiment_name='llama3_1b_grpo_hh_lp' \
    trainer.default_local_dir=$HOME/verl/logs \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.validation_data_dir=$HOME/data/helpfulness_hh_rlhf/rl \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=2

# reward_model.model.input_tokenizer=Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \