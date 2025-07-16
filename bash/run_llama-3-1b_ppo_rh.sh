#!/bin/bash                                                                                                                                                  
#SBATCH --job-name=grpo                                                                                                                                
#SBATCH --output=/home/taywonmin/slurm-logs/test-%j.log  # log                                                                                                   
#SBATCH --error=/home/taywonmin/slurm-logs/test-%j.log   # log                                                                                                   
#SBATCH --nodes=1            # 노드 1개 사용                                                                                                                 
#SBATCH --gres=gpu:a6000:4   # GPU 1개 사용                                                                                                                  
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수                                                                                                             
#SBATCH --mem-per-gpu=32G     # GPU당 mem 사용량                                                                                                              
#SBATCH --time=72:00:00      # 최대 48시간 실행   


cd $HOME/verl

unset ROCR_VISIBLE_DEVICES

set -x

python3 -m verl.trainer.main_ppo --config-path=./config --config-name='ppo_trainer'\
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.01 \
    data.return_raw_chat=True \
    data.return_raw_input_ids=True \
    data.train_files=$HOME/data/helpsteer2/rl/train.parquet \
    data.val_files=$HOME/data/helpsteer2/rl/test.parquet \
    data.train_batch_size=128 \
    data.prompt_key=prompt \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=meta-llama/Llama-3.2-1B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    critic.optim.lr=1e-5 \
    critic.model.path=meta-llama/Llama-3.2-1B-Instruct \
    critic.model.enable_gradient_checkpointing=False \
    critic.ppo_micro_batch_size_per_gpu=1 \
    reward_model.enable=True \
    reward_model.model.path=/root/rm/Llama-3.2-1B-rm \
    reward_model.micro_batch_size_per_gpu=16 \
    +reward_model.param_offload=False \
    algorithm.use_kl_in_reward=False \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='ppo_reward_hacking' \
    trainer.experiment_name='llama_3_1b_ppo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=10 \
    +reward_model.enable_true_reward_model=True \
    +reward_model.true_reward_model.model.path=Skywork/Skywork-Reward-Llama-3.1-8B-v0.2 \
    +reward_model.true_reward_model.micro_batch_size_per_gpu=1 \
    +reward_model.true_reward_model.param_offload=False \
    +reward_model.true_reward_model.model.fsdp_config.min_num_params=0 \
    +reward_model.true_reward_model.model.fsdp_config.param_offload=False \
    +reward_model.true_reward_model.model.fsdp_config.fsdp_size=-1 \
    +reward_model.true_reward_model.micro_batch_size=1 \
    +reward_model.true_reward_model.model.input_tokenizer=null \
    +reward_model.true_reward_model.strategy=fsdp \
    +reward_model.true_reward_model.model.fsdp_config.forward_prefetch=False \
    +reward_model.true_reward_model.use_dynamic_bsz=True \
    +reward_model.true_reward_model.forward_max_token_len_per_gpu=4096 \
    $@