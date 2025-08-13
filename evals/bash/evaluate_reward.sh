#!/bin/bash

model_list=(
    "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    "Skywork/Skywork-Reward-V2-Llama-3.2-1B"
    "Skywork/Skywork-Reward-V2-Llama-3.2-3B"
    "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    "Skywork/Skywork-Reward-V2-Qwen3-1.7B"
    "Skywork/Skywork-Reward-V2-Qwen3-4B"
    "Skywork/Skywork-Reward-V2-Qwen3-8B"
)

dataset="Taywon/HH_full_parsed"
device="cuda:0"
batch_size=16
max_length=2048
split="test"

echo "Starting evaluation of ${#model_list[@]} models on dataset: $dataset"
echo "=========================================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Loop through each model
for model in "${model_list[@]}"; do
    echo "Evaluating model: $model"
    echo "Started at: $(date)"
    
    # Extract model name for logging (remove path)
    model_name=$(basename "$model")
    log_file="logs/evaluation_${model_name}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run the evaluation script
    python src/evaluate.py \
        --model_name "$model" \
        --dataset_name "$dataset" \
        --device "$device" \
        --batch_size "$batch_size" \
        --max_length "$max_length" \
        --split "$split" \
        2>&1 | tee "$log_file"
    
    # Check if the evaluation was successful
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed evaluation for $model"
        echo "Log saved to: $log_file"
    else
        echo "✗ Failed to evaluate $model"
        echo "Check log file: $log_file"
    fi
    
    echo "Finished at: $(date)"
    echo "----------------------------------------------------------"
    echo
done

echo "All evaluations completed!"
echo "Results and logs are saved in the current directory and logs/ folder"

