LOCAL_DIR=/home/taywonmin/rhbench/verl/logs/llama-3.2-1b/global_step_50/actor
TARGET_DIR=/home/taywonmin/rhbench/verl/logs/llama-3.2-1b/global_step_50/actor/merged

python -m verl.model_merger merge --backend fsdp --local_dir $LOCAL_DIR --target_dir $TARGET_DIR