# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
- Preprocess data and split the training set into 75% for training RM and 25% for validting RM.
- All the training data is used to train SFT and RL.
- Both chosen and rejected is used to train SFT
"""

import argparse
import os

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from verl.utils.fs import copy, makedirs


def generate_sft_dataset(target_hdfs_path_dir, local_dir="~/data/helpsteer2/sft"):
    dataset = load_dataset("Dahoas/full-hh-rlhf")
    output = {"prompt": [], "response": []}
    for data in tqdm(dataset["train"]):
        # add chosen
        output["prompt"].append(data["prompt"])
        output["response"].append(data["chosen"])

        # add rejection
        output["prompt"].append(data["prompt"])
        output["response"].append(data["rejected"])

    df = pd.DataFrame(output)

    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    local_path = os.path.join(local_dir, "train.parquet")

    df.to_parquet(path=local_path)

    if target_hdfs_path_dir is not None:
        hdfs_dir = target_hdfs_path_dir + "/" + "train.parquet"
        makedirs(hdfs_dir)

        copy(local_path, hdfs_dir)


def generate_rm_dataset(target_hdfs_path_dir, local_dir="~/data/helpsteer2/rm"):
    train_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train[:75%]")
    test_dataset = load_dataset("Dahoas/full-hh-rlhf", split="train[-25%:]")

    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    for dataset, name in zip([train_dataset, test_dataset], ["train", "test"]):
        output = {"prompt": [], "chosen": [], "rejected": []}
        for data in tqdm(dataset):
            # add chosen
            output["prompt"].append(data["prompt"])
            output["chosen"].append(data["chosen"])
            output["rejected"].append(data["rejected"])

        df = pd.DataFrame(output)

        local_path = os.path.join(local_dir, name + ".parquet")

        df.to_parquet(path=local_path)

        if target_hdfs_path_dir is not None:
            hdfs_dir = target_hdfs_path_dir + "/" + name + ".parquet"
            makedirs(hdfs_dir)

            copy(local_path, hdfs_dir)


def generate_rl_dataset(target_hdfs_path_dir, local_dir="~/data/helpsteer2/rl"):
    dataset = load_dataset("nvidia/HelpSteer2")
    train_dataset = dataset["train"]
    test_dataset = dataset["validation"]

    data_source = "nvidia/HelpSteer2"

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop("prompt")
            response = example.pop("response")

            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": prompt}],
                "ability": "alignment",
                "reward_model": {
                    "style": "model",
                    "ground_truth": response,  # should not be used
                },
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    
    
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)
    
    # First identify duplicated prompts for train dataset
    seen_prompts = set()
    duplicate_indices_train = []
    for idx, example in enumerate(train_dataset):
        prompt = example["prompt"][0]["content"]
        if prompt in seen_prompts:
            duplicate_indices_train.append(idx)
        else:
            seen_prompts.add(prompt)

    print(f"Number of duplicate prompts: {len(duplicate_indices_train)} among {len(train_dataset)} train examples")
    
    # First identify duplicated prompts for test dataset
    seen_prompts = set()
    duplicate_indices_test = []
    for idx, example in enumerate(test_dataset):
        prompt = example["prompt"][0]["content"]
        if prompt in seen_prompts:
            duplicate_indices_test.append(idx)
        else:
            seen_prompts.add(prompt)

    print(f"Number of duplicate prompts: {len(duplicate_indices_test)} among {len(test_dataset)} test examples")
    
    # Filter out duplicates
    train_dataset = train_dataset.filter(lambda example, idx: idx not in duplicate_indices_train, with_indices=True)
    test_dataset = test_dataset.filter(lambda example, idx: idx not in duplicate_indices_test, with_indices=True)

    local_dir = os.path.expanduser(local_dir)
    local_path_train = os.path.join(local_dir, "train.parquet")
    local_path_test = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(local_path_train)
    test_dataset.to_parquet(local_path_test)

    if target_hdfs_path_dir is not None:
        hdfs_dir = target_hdfs_path_dir + "/" + "train.parquet"
        makedirs(hdfs_dir)
        copy(local_path_train, hdfs_dir)
        
        hdfs_dir = target_hdfs_path_dir + "/" + "test.parquet"
        makedirs(hdfs_dir)
        copy(local_path_test, hdfs_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, choices=["sft", "rm", "rl"], required=True)
    parser.add_argument("--local_dir", type=str, default="~/data/helpsteer2/")
    parser.add_argument("--hdfs_dir", type=str, required=False, default=None)

    args = parser.parse_args()

    if args.split == "sft":
        generate_sft_dataset(args.hdfs_dir, os.path.join(args.local_dir, args.split))
    elif args.split == "rm":
        generate_rm_dataset(args.hdfs_dir, os.path.join(args.local_dir, args.split))
    elif args.split == "rl":
        generate_rl_dataset(args.hdfs_dir, os.path.join(args.local_dir, args.split))
    else:
        raise NotImplementedError
