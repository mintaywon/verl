import json
import torch
import argparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="evals/results/llama-3.2-1b_hh.json", help="Path to the generated responses")
    parser.add_argument("--output_path", type=str, default="evals/results/llama-3.2-1b_hh_scores.json", help="Path to the output scores")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum length of the input")
    parser.add_argument("--reward_model_path", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B", help="Path to the reward model")
    return parser.parse_args()

args = parse_args()

# Load prompts, outputs, and results from input file
with open(args.input_path, "r") as f:
    results = json.load(f)
    prompts = results["prompt"]
    outputs = results["response"]

# Load reward model
rm = AutoModelForSequenceClassification.from_pretrained(
    args.reward_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    num_labels=1,
)
rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
if rm_tokenizer.pad_token is None:
    rm_tokenizer.pad_token = rm_tokenizer.eos_token

scores = []
token_lengths = []

# Process the generated prompts and responses in batches
batch_size = args.batch_size
with torch.no_grad():
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc="Scoring responses"):
        batch_end = min(batch_start + batch_size, len(prompts))
        batch_prompts = prompts[batch_start:batch_end]
        batch_outputs = outputs[batch_start:batch_end]

        # Remove system prompt if present in the prompt
        batch_full_texts = []
        for p, o in zip(batch_prompts, batch_outputs):
            # Split the prompt using the special delimiter
            split_prompt = p.split("<|eot_id|><|start_header_id|>user<|end_header_id|>")
            if len(split_prompt) > 1:
                prompt_no_system = split_prompt[1]
            else:
                prompt_no_system = p
            full_text = prompt_no_system + o
            batch_full_texts.append(full_text)

        # Tokenize the batch of full texts
        inputs = rm_tokenizer(
            batch_full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length
        ).to(rm.device)

        # Get reward scores for the batch
        batch_scores = rm(**inputs).logits.squeeze(-1).cpu().tolist()
        if isinstance(batch_scores, float):  # Handle single example in batch
            batch_scores = [batch_scores]
        scores.extend(batch_scores)

        # Get token lengths (prompt + response) for the batch
        batch_token_lengths = inputs["attention_mask"].sum(dim=1).cpu().tolist()
        token_lengths.extend(batch_token_lengths)

# Update results with scores and token lengths
results.update({
    "scores": scores,
    "token_lengths": token_lengths
})

# Save updated results to the same output path
with open(args.output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results with scores to {args.output_path}")