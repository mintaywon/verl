import datasets
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluation script arguments")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for evaluation")
    parser.add_argument("--dataset_name", type=str, default="Taywon/HH_full_parsed", help="Name of the dataset to use")
    parser.add_argument("--model_name", type=str, default="Skywork/Skywork-Reward-V2-Llama-3.1-8B", help="Model name or path")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = args.device
    dataset_name = args.dataset_name
    model_name = args.model_name
    split = args.split
    max_length = args.max_length
    batch_size = args.batch_size
    dataset = datasets.load_dataset(dataset_name, split=split)
    # Let's first parse the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def format_dataset(example):
        formatted_chosen = tokenizer.apply_chat_template(example["chosen"], tokenize=False)
        formatted_rejected = tokenizer.apply_chat_template(example["rejected"], tokenize=False)

            # These two lines remove the potential duplicate bos token
        if tokenizer.bos_token is not None and formatted_chosen.startswith(tokenizer.bos_token):
            formatted_chosen = formatted_chosen[len(tokenizer.bos_token):]
        if tokenizer.bos_token is not None and formatted_rejected.startswith(tokenizer.bos_token):
            formatted_rejected = formatted_rejected[len(tokenizer.bos_token):]

        return {"chosen": formatted_chosen, "rejected": formatted_rejected}

    def tokenize_dataset(example):
        tokenized_chosen = tokenizer(example["chosen"], padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        tokenized_rejected = tokenizer(example["rejected"], padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
        
        # Squeeze the batch dimension since we're processing one example at a time
        chosen_squeezed = {k: v.squeeze(0) for k, v in tokenized_chosen.items()}
        rejected_squeezed = {k: v.squeeze(0) for k, v in tokenized_rejected.items()}
        
        return {"chosen": chosen_squeezed, "rejected": rejected_squeezed}

    dataset = dataset.map(format_dataset)
    dataset = dataset.remove_columns(["prompt"])
    dataset = dataset.map(tokenize_dataset)
    # Set format to torch for efficiency
    dataset.set_format(type='torch')

    # dataset = dataset.select(range(32))
    # dataloader
    def collate_fn(batch):
        chosen_batch = {}
        rejected_batch = {}
        
        # Get all the keys from the first example
        chosen_keys = batch[0]["chosen"].keys()
        rejected_keys = batch[0]["rejected"].keys()
        
        # Stack tensors for chosen and move to device (they're already tensors)
        for key in chosen_keys:
            chosen_batch[key] = torch.stack([example["chosen"][key] for example in batch]).to(device)
        
        # Stack tensors for rejected and move to device (they're already tensors)
        for key in rejected_keys:
            rejected_batch[key] = torch.stack([example["rejected"][key] for example in batch]).to(device)
        
        return {
            "chosen": chosen_batch,
            "rejected": rejected_batch
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # load model
    rm = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
        num_labels=1,
    )

    chosen_scores = []
    rejected_scores = []
    token_length_chosen = []
    token_length_rejected = []


    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            chosen_scores_batch = rm(**batch["chosen"]).logits.squeeze(-1).cpu().tolist()
            rejected_scores_batch = rm(**batch["rejected"]).logits.squeeze(-1).cpu().tolist()

            # store the chosen_scores, rejected_scores, and token_length_chosen, token_length_rejected
            chosen_scores.append(chosen_scores_batch)
            rejected_scores.append(rejected_scores_batch)
            token_length_chosen.append(batch["chosen"]["attention_mask"].sum(dim=1).cpu().tolist())
            token_length_rejected.append(batch["rejected"]["attention_mask"].sum(dim=1).cpu().tolist())

            # print(f"chosen_scores: {chosen_scores}")
            # print(f"rejected_scores: {rejected_scores}")
            # print(f"token_length_chosen: {token_length_chosen}")
            # print(f"token_length_rejected: {token_length_rejected}")
    # Flatten the lists so that each is a single long list
    chosen_scores = sum(chosen_scores, [])
    rejected_scores = sum(rejected_scores, [])
    token_length_chosen = sum(token_length_chosen, [])
    token_length_rejected = sum(token_length_rejected, [])


    # Combine the results into a list of dicts, one per example
    results = {
        "chosen_scores": chosen_scores,
        "rejected_scores": rejected_scores,
        "token_length_chosen": token_length_chosen,
        "token_length_rejected": token_length_rejected
    }

    # Save to a JSON file, with model name and dataset in the filename
    output_filename = f"evaluation_results_{model_name.split('/')[-1]}_{dataset_name.split('/')[-1]}.json"
    with open(output_filename, "w") as f:
        json.dump(results, f, indent=2)






