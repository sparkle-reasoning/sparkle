#!/usr/bin/env python
"""
FSDP to HuggingFace Checkpoint Converter

This script converts FSDP (Fully Sharded Data Parallel) checkpoints to HuggingFace format
and optionally uploads them to HuggingFace Hub.

Features:
- Automatically detects world_size from checkpoint files
- Handles sharded model consolidation
- Supports uploading to HuggingFace Hub
- Secure token handling via environment variables
- Comprehensive error handling and validation

Usage Examples:
    # Convert FSDP checkpoint to HuggingFace format
    python fsdp2hf.py --fsdp_path /path/to/fsdp/checkpoint --base_model Qwen/Qwen2.5-Math-7B --output_path /path/to/output
    
    # Convert and upload to HuggingFace Hub
    export HF_TOKEN=your_token_here
    python fsdp2hf.py --fsdp_path /path/to/fsdp/checkpoint --base_model Qwen/Qwen2.5-Math-7B --output_path /path/to/output --upload --repo_id username/model-name
    
    # Upload to private repository
    python fsdp2hf.py --fsdp_path /path/to/fsdp/checkpoint --base_model Qwen/Qwen2.5-Math-7B --output_path /path/to/output --upload --repo_id username/model-name --private

Security Note:
    For security, use the HF_TOKEN environment variable instead of passing tokens as arguments.
    Never commit scripts containing hardcoded tokens to version control.
"""

import argparse
import os
import re
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs


def load_sharded_model(fsdp_checkpoint_path):
    """
    Load and consolidate sharded FSDP model weights.
    
    Args:
        fsdp_checkpoint_path: Path to directory containing FSDP checkpoint shards
        
    Returns:
        Dict containing consolidated model state dict
    """
    state_dict = defaultdict(list)
    checkpoint_dir = Path(fsdp_checkpoint_path)

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {fsdp_checkpoint_path}")

    shard_files = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))
    if not shard_files:
        raise ValueError(f"No checkpoint files found in {fsdp_checkpoint_path}")

    pattern = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt")
    world_sizes = set()
    for file in shard_files:
        match = pattern.match(file.name)
        if match:
            world_sizes.add(int(match.group(1)))

    if len(world_sizes) != 1:
        raise ValueError(
            f"Inconsistent world_size found in checkpoint files: {world_sizes}"
        )

    world_size = world_sizes.pop()
    print(f"Found checkpoints with world_size = {world_size}")

    for rank in range(world_size):
        filepath = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not filepath.exists():
            raise ValueError(f"Missing shard file: {filepath}")

        print(f"Loading shard: {filepath}")
        try:
            shard_dict = torch.load(filepath, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"Failed to load shard {filepath}: {str(e)}")

        for key, value in shard_dict.items():
            if hasattr(value, "to_local"):
                value = value.to_local()
            state_dict[key].append(value)

    consolidated_state_dict = {}
    for key in state_dict:
        try:
            consolidated_state_dict[key] = torch.cat(state_dict[key], dim=0)
        except (RuntimeError, TypeError) as e:
            # Some parameters might not need concatenation
            consolidated_state_dict[key] = state_dict[key][0]
            print(f"Parameter '{key}' does not need concatenation, using first shard value")

    return consolidated_state_dict


def initialize_model_and_tokenizer(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
):
    local_path = copy_local_path_from_hdfs(model_path)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    actor_model_config = AutoConfig.from_pretrained(
        local_path, trust_remote_code=trust_remote_code
    )
    actor_module = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=local_path,
        torch_dtype=torch_dtype,
        config=actor_model_config,
        attn_implementation="flash_attention_2",
        trust_remote_code=trust_remote_code,
    )

    return tokenizer, actor_module


def convert_fsdp_to_hf_checkpoint(fsdp_checkpoint_path, hf_base_model_path, output_path):
    """
    Convert FSDP checkpoint to HuggingFace checkpoint format
    
    Args:
        fsdp_checkpoint_path: Path to FSDP checkpoint directory
        hf_base_model_path: Path to base HuggingFace model or model ID
        output_path: Path where to save the converted model
    """
    # Initialize model and tokenizer from base HF model
    tokenizer, model = initialize_model_and_tokenizer(hf_base_model_path)
    
    # Load the FSDP checkpoint
    state_dict = load_sharded_model(fsdp_checkpoint_path)
    
    # Load state dict into the model
    model.load_state_dict(state_dict)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Save the model and tokenizer in HuggingFace format
    print(f"Saving converted model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    
    print(f"Model and tokenizer successfully saved to {output_path}")
    
    return model, tokenizer

def upload_model_to_hf(
    local_model_path: str,
    hf_repo_id: str,
    hf_token: str = None,
    private: bool = False
):
    """
    Upload a local model checkpoint to Hugging Face Hub using push_to_hub.
    
    Args:
        local_model_path: Path to the local model checkpoint
        hf_repo_id: Hugging Face repository ID (e.g., "username/model-name")
        hf_token: Hugging Face API token (recommended to use HF_TOKEN environment variable)
        private: Whether to create a private repository
    """
    # Get token from environment variable if not provided
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HuggingFace token not provided. Set HF_TOKEN environment variable or pass --hf_token argument.")
    
    # Login to Hugging Face
    try:
        login(token=hf_token)
    except Exception as e:
        raise ValueError(f"Failed to login to HuggingFace: {str(e)}")
    
    try:
        print(f"Loading model from {local_model_path}")
        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            local_model_path,
            trust_remote_code=True
        )
        
        print(f"Uploading model to {hf_repo_id}")
        # Push model and tokenizer to hub
        model.push_to_hub(hf_repo_id, private=private)
        tokenizer.push_to_hub(hf_repo_id, private=private)
        
        print(f"Successfully uploaded model to https://huggingface.co/{hf_repo_id}")
        
    except Exception as e:
        print(f"Error uploading model: {str(e)}")
        raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert FSDP checkpoint to HuggingFace format")
    
    # Required arguments
    parser.add_argument("--fsdp_path", type=str, required=True,
                        help="Path to FSDP checkpoint directory")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Base HuggingFace model path or model ID (e.g., Qwen/Qwen2.5-Math-7B)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output path for converted HuggingFace model")
    
    # Optional arguments
    parser.add_argument("--upload", action="store_true",
                        help="Upload the converted model to HuggingFace Hub")
    parser.add_argument("--repo_id", type=str,
                        help="HuggingFace repository ID for upload (e.g., username/model-name)")
    parser.add_argument("--hf_token", type=str,
                        help="HuggingFace API token (alternatively set HF_TOKEN environment variable)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository on HuggingFace Hub")
    
    return parser.parse_args()


def main():
    """Main function to convert FSDP checkpoint to HuggingFace format."""
    args = parse_args()
    
    # Validate arguments
    if args.upload and not args.repo_id:
        raise ValueError("--repo_id is required when --upload is specified")
    
    # Check if paths exist
    if not os.path.exists(args.fsdp_path):
        raise FileNotFoundError(f"FSDP checkpoint path not found: {args.fsdp_path}")
    
    print(f"Converting FSDP checkpoint to HuggingFace format")
    print(f"FSDP path: {args.fsdp_path}")
    print(f"Base model: {args.base_model}")
    print(f"Output path: {args.output_path}")
    
    # Convert FSDP checkpoint to HuggingFace format
    try:
        model, tokenizer = convert_fsdp_to_hf_checkpoint(
            fsdp_checkpoint_path=args.fsdp_path,
            hf_base_model_path=args.base_model,
            output_path=args.output_path
        )
        print(f"✅ Successfully converted model to {args.output_path}")
        
        # Upload to HuggingFace Hub if requested
        if args.upload:
            print(f"Uploading model to HuggingFace Hub: {args.repo_id}")
            upload_model_to_hf(
                local_model_path=args.output_path,
                hf_repo_id=args.repo_id,
                hf_token=args.hf_token,
                private=args.private
            )
            print(f"✅ Successfully uploaded model to https://huggingface.co/{args.repo_id}")
            
    except Exception as e:
        print(f"❌ Error during conversion: {str(e)}")
        raise
    
if __name__ == "__main__":
    main()
    