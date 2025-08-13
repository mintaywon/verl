from vllm import LLM, SamplingParams
import argparse
import datasets
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import torch
from tqdm import tqdm
def parse_args():
    parser = argparse.ArgumentParser(description="Generate response script arguments")
    parser.add_argument("--model_path", type=str, default="/home/taywonmin/rhbench/verl/logs/llama-3.2-1b/global_step_50/actor/merged", help="path to model checkpoint")
    parser.add_argument("--dataset_name", type=str, default="Taywon/HH_full_parsed", help="Name of the dataset to use")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for evaluation")
    parser.add_argument("--n", type=int, default=2, help="Number of responses to generate per prompt")
    parser.add_argument("--output_path", type=str, default="/home/taywonmin/rhbench/evals/results/llama-3.2-1b_hh.json", help="Path to save results")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    # load dataset
    dataset = datasets.load_dataset(args.dataset_name, split=args.split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    def process_fn(example, idx):
        # Parse chosen (list of dicts) into prompt string as described
        chosen = example.pop("chosen")
        prompt_parts = []
        for turn in chosen[:-1]:  # Exclude the last assistant response
            if turn["role"] == "user":
                prompt_parts.append(f"Human: {turn['content']}")
            elif turn["role"] == "assistant":
                prompt_parts.append(f"Assistant: {turn['content']}")
        # Add the final user turn if present, and append "Assistant:" for the prompt
        if chosen and chosen[-1]["role"] == "user":
            prompt_parts.append(f"Human: {chosen[-1]['content']}")
            prompt_parts.append("Assistant:")
        else:
            prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        
        return  {"prompt": [{"role": "user", "content": prompt}]}
    
    def format_dataset(example):
        formatted_prompt = tokenizer.apply_chat_template(
            example["prompt"], 
            tokenize=False, 
            add_generation_prompt=False,
        )

            # These two lines remove the potential duplicate bos token
        if tokenizer.bos_token is not None and formatted_prompt.startswith(tokenizer.bos_token):
            formatted_prompt = formatted_prompt[len(tokenizer.bos_token):]

        return {"prompt": formatted_prompt}
    
    dataset = dataset.map(process_fn, with_indices=True)
    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(["rejected"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    sampling_params = SamplingParams(temperature=1.0, max_tokens=args.max_length, n=args.n)
    llm = LLM(model=args.model_path)
    outputs = []
    prompts = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        output_batch = llm.generate(batch["prompt"], sampling_params=sampling_params)
        for output in output_batch:
            prompt = output.prompt
            prompt = prompt.split("<|eot_id|><|start_header_id|>user<|end_header_id|>")
            if len(prompt) > 1:
                prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>" + prompt[1]
            else:
                prompt = "<|eot_id|><|start_header_id|>user<|end_header_id|>" + prompt[0]
            prompts.extend([prompt] * args.n)
            # Fix: handle multiple outputs per prompt (n > 1)
            for out in output.outputs:
                outputs.append(out.text)

    results = {
        "prompt": prompts,
        "response": outputs
    }

    # save
    import os
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    
