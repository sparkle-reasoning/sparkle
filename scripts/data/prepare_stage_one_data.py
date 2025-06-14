"""Script to prepare stage one training and test datasets.

This script processes math problem datasets into a standardized format 
for training stage one models. It loads problems from specified datasets, 
adds instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
import json
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
import datasets

def make_map_fn(split: str, data_source: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        question = example.pop('problem')
        instruction = """A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step. The Assistant first thinks about the complete reasoning process in the mind enclosed within <think> </think> tags. Then the Assistant provides a clear, concise answer to the user within <answer> </answer> tags, with the final result enclosed in \\boxed{} notation.\n\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\nThe answer is \\boxed{...}.\n</answer>"""
        question = f"{instruction}\n\nUser: {question} Assistant:"
        answer = example.pop('answer')

        data = {
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn

def save_jsonl(data: List[Dict], output_path: str):
    """Save data as jsonl file."""
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for es training')
    parser.add_argument('--local_dir', default=os.path.join(os.getcwd(), 'data'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)
    
    train_dataset_name = "sparkle-reasoning/dsr40k"
    test_dataset_name = "sparkle-reasoning/aime2024"

    train_dataset = datasets.load_dataset(train_dataset_name)['train']
    test_dataset = datasets.load_dataset(test_dataset_name)['test']

    # Process training data
    train_data: List[Dict[str, Any]] = []
    train_data_source_suffix = train_dataset_name.split('/')[-1]
    train_data_source = f'sparkle_{train_data_source_suffix}'
    process_fn = make_map_fn('train', train_data_source)
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)
            
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, f'{train_data_source}.parquet'))
    save_jsonl(train_data, os.path.join(local_dir, f'{train_data_source}.jsonl'))

    # Process test data
    test_data: List[Dict[str, Any]] = []
    test_data_source_suffix = test_dataset_name.split('/')[-1]
    test_data_source = f'sparkle_{test_data_source_suffix}'
    process_fn = make_map_fn('test', test_data_source)
    for idx, example in enumerate(test_dataset):
            processed_example = process_fn(example, idx)
            if processed_example is not None:
                test_data.append(processed_example)

    test_df = pd.DataFrame(test_data)
    test_df.to_parquet(os.path.join(local_dir, f'{test_data_source}.parquet'))
    save_jsonl(test_data, os.path.join(local_dir, f'{test_data_source}.jsonl')) # manually check if the format of data is correct

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)