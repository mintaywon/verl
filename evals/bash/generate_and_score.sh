#!/bin/bash
#SBATCH --job-name=generate_response
#SBATCH --output=/home/taywonmin/slurm-logs/test-%j.log  # log
#SBATCH --error=/home/taywonmin/slurm-logs/test-%j.log   # log
#SBATCH --nodes=1            # 노드 1개 사용
#SBATCH --gres=gpu:1         # GPU 1개 사용
#SBATCH --cpus-per-gpu=8     # GPU당 CPU 사용 수
#SBATCH --mem-per-gpu=32G    # GPU당 mem 사용량
#SBATCH --time=72:00:00      # 최대 72시간 실행
#SBATCH --exclude=node5,node7

set -e  # Exit immediately if any command fails

cd "$HOME/rhbench" || exit 1

# Default values
MODEL_NAME="llama-3.2-1b_grpo_lp_n0.01"
DATASET_NAME="Taywon/HH_full_parsed"
SPLIT="test"
MAX_LENGTH=2048 # 2048 is the max length of the generated response
BATCH_SIZE_RESPONSE=64 # 64 is the batch size for the generated response
BATCH_SIZE_EVAL=16 # 16 is the batch size for the evaluation
N=2

for STEP in $(seq 50 50 850); do
  if [[ "$STEP" == "0" ]]; then
    MODEL_PATH="meta-llama/Llama-3.2-1B"
    MERGED_MODEL_PATH="meta-llama/Llama-3.2-1B"
  else
    MODEL_PATH="/home/taywonmin/rhbench/verl/logs/$MODEL_NAME/global_step_$STEP/actor"
    MERGED_MODEL_PATH="/home/taywonmin/rhbench/verl/logs/$MODEL_NAME/global_step_$STEP/actor/merged"
  fi
  OUTPUT_PATH="/home/taywonmin/rhbench/evals/results/$MODEL_NAME/${MODEL_NAME}_hh_step_${STEP}.json"
  SCORES_PATH="/home/taywonmin/rhbench/evals/results/$MODEL_NAME/${MODEL_NAME}_hh_step_${STEP}_scores.json"

  # Step 1: Model merging
  if [[ "$STEP" != "0" ]]; then
    python -m verl.model_merger merge --backend fsdp --local_dir "$MODEL_PATH" --target_dir "$MERGED_MODEL_PATH"
    echo "Merged model saved to $MERGED_MODEL_PATH"
  fi

  # Step 2: Generate responses (will only run if Step 1 succeeds)
  python evals/generate_response.py \
      --model_path "$MERGED_MODEL_PATH" \
      --dataset_name "$DATASET_NAME" \
      --split "$SPLIT" \
      --max_length "$MAX_LENGTH" \
      --batch_size "$BATCH_SIZE_RESPONSE" \
      --n "$N" \
      --output_path "$OUTPUT_PATH"

  echo "Responses saved to $OUTPUT_PATH"

  # Step 3: Evaluate (will only run if Step 2 succeeds)
  python evals/evaluate_generated.py \
      --input_path "$OUTPUT_PATH" \
      --output_path "$SCORES_PATH" \
      --batch_size "$BATCH_SIZE_EVAL" \
      --max_length "$MAX_LENGTH" \
      --reward_model_path "Skywork/Skywork-Reward-V2-Llama-3.1-8B"

  echo "Scores saved to $SCORES_PATH"
done