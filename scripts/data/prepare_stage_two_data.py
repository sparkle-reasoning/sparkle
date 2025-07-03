#!/usr/bin/env python3
"""
Stage Two Data Preparation Script

This script prepares augmented training data for stage two training in Sparkle. The questions 
used in stage two are from the 'sparkle-reasoning/hardmath' dataset.

The script employs a simple yet effective augmentation strategy based on the assumption that 
models may fail and get stuck at certain reasoning steps, but could perform well if they 
overcome these obstacles. Instead of just exposing the question to the model during RL training, 
it now exposes partial reasoning steps as well. By providing these intermediate reasoning states, 
the model can explore more solution paths and potentially overcome reasoning bottlenecks where 
it might otherwise get stuck.

Key Features:
- Loads mathematical reasoning problems from the sparkle-reasoning/hardmath dataset
- Creates multiple progressive versions of each problem with varying amounts of thinking steps
- Supports two augmentation modes: 'first_half' (partial steps) and 'all' (complete steps)
- Outputs data in both Parquet and JSONL formats for flexible downstream usage

Ablation Results:
Our experiments show that using 'all' reasoning traces is more effective than 'first_half', 
providing better performance improvements. We recommend using --aug_version all for optimal results.

Usage:
    python prepare_stage_two_data.py --local_dir ./data --aug_version all
"""

import re
import os
import datasets

import argparse
from typing import List, Dict
import json

def split_thinking_process(text, steps):
    """
    Split a thinking process into a specified number of steps without modifying the format
    
    Args:
        text (str): The full thinking process text to split
        steps (int): Number of sections to split the text into
        
    Returns:
        list: A list of text segments representing the steps
    """
    # First try to split by paragraphs separated by blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If we have enough paragraphs, use them
    if len(paragraphs) >= steps:
        # If we have exactly the right number of paragraphs, use them as is
        if len(paragraphs) == steps:
            return paragraphs
            
        # If we have more paragraphs than needed steps, we need to consolidate
        # by joining adjacent paragraphs to form the requested number of steps
        consolidated_paragraphs = []
        # Calculate how many original paragraphs should go in each consolidated step
        paragraphs_per_step = len(paragraphs) / steps
        
        for i in range(steps):
            start_idx = int(i * paragraphs_per_step)
            # For the last step, include all remaining paragraphs
            if i == steps - 1:
                end_idx = len(paragraphs)
            else:
                end_idx = int((i + 1) * paragraphs_per_step)
            
            # Join the paragraphs for this step with newlines in between
            joined_paragraph = "\n\n".join(paragraphs[start_idx:end_idx])
            consolidated_paragraphs.append(joined_paragraph)
            
        return consolidated_paragraphs
    
    # If not enough paragraphs, split by sentences
    # Split text into sentences using regex
    sentence_regex = r'([^.!?]+[.!?]+)'
    sentences = re.findall(sentence_regex, text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Check if there's any remaining text that doesn't end with punctuation
    last_sentence_end = text.rfind(sentences[-1]) + len(sentences[-1]) if sentences else 0
    if last_sentence_end < len(text):
        remaining = text[last_sentence_end:].strip()
        if remaining:
            sentences.append(remaining)
    
    # If we don't have enough sentences, just return what we have
    if len(sentences) <= steps:
        return sentences
    
    # Determine how many sentences per step (approximately)
    sentences_per_step = max(1, (len(sentences) + steps - 1) // steps)  # Ceiling division
    
    # Create steps array to hold the grouped sentences
    steps_array = []
    
    # Group sentences into steps
    for i in range(steps):
        start_idx = i * sentences_per_step
        end_idx = min(start_idx + sentences_per_step, len(sentences))
        
        if start_idx < len(sentences):
            step_sentences = sentences[start_idx:end_idx]
            steps_array.append(' '.join(step_sentences))
        else:
            # If we've run out of sentences, break
            break
    
    return steps_array

def smart_split_thinking_process(text, target_steps, use_first_half=False):
    """
    Enhanced version that tries to split at logical breaks in the text
    without adding labels or changing the format
    
    Args:
        text (str): The full thinking process text to split
        target_steps (int): Number of sections to split the text into
        use_first_half (bool): If True, only return first half of the steps
        
    Returns:
        list: A list of text segments representing the steps
    """
    steps = target_steps // 2 if use_first_half else target_steps # one can easily change this to test more granular steps
    
    # First try to split by paragraphs separated by blank lines
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # If we have enough paragraphs, use them
    if len(paragraphs) >= steps:
        # If we have exactly the right number of paragraphs, use them as is
        if len(paragraphs) == steps:
            return paragraphs
        
        # If we have more paragraphs than needed steps, we need to consolidate
        # by joining adjacent paragraphs to form the requested number of steps
        consolidated_paragraphs = []
        # Calculate how many original paragraphs should go in each consolidated step
        paragraphs_per_step = len(paragraphs) / steps
        
        for i in range(steps):
            start_idx = int(i * paragraphs_per_step)
            # For the last step, include all remaining paragraphs
            if i == steps - 1:
                end_idx = len(paragraphs)
            else:
                end_idx = int((i + 1) * paragraphs_per_step)
            
            # Join the paragraphs for this step with newlines in between
            joined_paragraph = "\n\n".join(paragraphs[start_idx:end_idx])
            consolidated_paragraphs.append(joined_paragraph)
            
        return consolidated_paragraphs
    
    # Look specifically for numbered items (e.g., "1.", "2.", etc.) that likely indicate new sections
    number_pattern = r'(?:^|[.!?]\s+|\n\s*)(\d+\.\s+(?:\*\*)?)'
    
    # Find all matches for the pattern
    matches = list(re.finditer(number_pattern, text))
    
    # If we find enough numbered sections, use those as break points
    if len(matches) >= steps - 1:
        section_starts = [0]  # Always start with position 0
        
        # Add positions of numbered sections
        for match in matches:
            # Find the actual position of the number, not including prior punctuation
            number_pos = match.start()
            # Extract just the position of the digit (e.g., "1" in ". 1. **")
            digit_match = re.search(r'\d+', match.group())
            if digit_match:
                # Adjust to start position of the digit
                number_pos = number_pos + match.group().index(digit_match.group())
            section_starts.append(number_pos)
        
        # Sort the section starts (should already be sorted, but just to be safe)
        section_starts.sort()
        
        # If we have too many sections, we need to consolidate
        if len(section_starts) >= steps:
            consolidated_sections = []
            sections_per_step = len(section_starts) / steps
            
            for i in range(steps):
                start_idx = int(i * sections_per_step)
                
                # For the last step, include all remaining sections
                if i == steps - 1:
                    end_idx = len(section_starts)
                else:
                    end_idx = int((i + 1) * sections_per_step)
                
                # Find the text from the first section's start to the last section's start
                start_pos = section_starts[start_idx]
                
                # For the last consolidated section, go to the end of the text
                if end_idx >= len(section_starts):
                    end_pos = len(text)
                else:
                    # Otherwise go to the start of the next consolidated section
                    end_pos = section_starts[end_idx]
                
                section_text = text[start_pos:end_pos].strip()
                consolidated_sections.append(section_text)
            
            return consolidated_sections
            
        # Create sections based on the identified break points
        sections = []
        for i in range(len(section_starts) - 1):
            start_pos = section_starts[i]
            end_pos = section_starts[i + 1]
            sections.append(text[start_pos:end_pos].strip())
        
        return sections[:steps]
    
    # Check for bullet lists
    bullet_pattern = r'(?:^|\n)\s*[-*•]\s+\*\*'
    bullet_matches = re.findall(bullet_pattern, text)
    
    # If we found bullet points, handle them differently
    if len(bullet_matches) >= 2:
        # For bullet lists, try to split by sentences instead of breaking up the list
        return split_thinking_process(text, target_steps)[:steps]
    
    # If we get here, try generic section detection
    # Patterns for identifying section breaks in math/geometry problems
    patterns = [
        # Bold headers with colons
        r'(?:^|\s)(\*\*[^*]+\*\*)(?:\s*:)',                                   
        # Capital-started phrases with colons
        r'(?:^|\s)([A-Z][a-zA-Z\s]+:)(?:\s)',                                 
        # Common section starters in mathematical reasoning
        r'(?:^|\s)(Understanding|Determining|Calculating|Finding|Analysis|Diagram|Position|Area|Volume|Step|First|Next|Finally)(?:\s+(?:of|the))?(?:\s*:)'
    ]
    
    # Find all potential section breaks
    potential_breaks = []
    
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            potential_breaks.append(match.start())
    
    # Sort the breaks by position
    potential_breaks.sort()
    
    # If we still don't have enough section breaks, fall back to sentence splitting
    if len(potential_breaks) < steps - 1:
        return split_thinking_process(text, target_steps)[:steps]
    
    # Add the start of the text as the first break
    section_breaks = [0] 
    # Add the potential breaks we need
    section_breaks.extend(potential_breaks[:steps])
    # Add the end of the text as the final position
    section_breaks.append(len(text))
    
    # If we have more breaks than needed, consolidate sections
    if len(section_breaks) > steps + 1:
        consolidated_sections = []
        # How many original sections to include in each consolidated section
        sections_per_step = (len(section_breaks) - 1) / steps
        
        for i in range(steps):
            start_idx = int(i * sections_per_step)
            
            # For the last step, include all remaining sections
            if i == steps - 1:
                end_idx = len(section_breaks) - 1
            else:
                end_idx = int((i + 1) * sections_per_step)
            
            # Get the text from the start break to the end break
            section_text = text[section_breaks[start_idx]:section_breaks[end_idx]].strip()
            consolidated_sections.append(section_text)
        
        return consolidated_sections
    
    # Extract the text for each section
    sections = []
    for i in range(min(steps, len(section_breaks) - 1)):
        start = section_breaks[i]
        end = section_breaks[i + 1]
        sections.append(text[start:end].strip())
        
    return sections

def save_jsonl(data: List[Dict], output_path: str):
    """Save data as jsonl file."""
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default=os.path.join(os.getcwd(), 'data'),
                    help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--aug_version', default='all', choices=['all', 'first_half',], help='Training data augmentation version to use')

    args = parser.parse_args()
    
    dataset_names = ['sparkle-reasoning/hardmath']
    
    instruction = """A conversation between User and Assistant. The user asks a math question, and the Assistant solves it step by step. The Assistant first thinks about the complete reasoning process in the mind enclosed within <think> </think> tags. Then the Assistant provides a clear, concise answer to the user within <answer> </answer> tags, with the final result enclosed in \\boxed{} notation.\n\nFor example:\n<think>\nreasoning process here\n</think>\n<answer>\nThe answer is \\boxed{...}.\n</answer>"""

    for dataset_name in dataset_names:
        data_source_suffix = dataset_name.split('/')[-1]
        data_source = f'sparkle_{data_source_suffix}'

        train_dataset = datasets.load_dataset(dataset_name)['train']

    
        print(f"train data size: {len(train_dataset)}")
        
        augmented_data = []
        
        for idx, example in enumerate(train_dataset):
            question_raw = example.get('problem')
            solution = example.get('answer')
            
            # First, always include the original non-augmented version
            original_question = f"{instruction}\n\nUser: {question_raw} Assistant:"
            
            original_data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": original_question,
                }],
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': 'train',
                    'index': f"{idx}_0",
                    "question": question_raw,
                    "thinking_level": 0,  # indicates no thinking included, pure question
                    "augmented": False
                }
            }
            augmented_data.append(original_data)
            
            thinking_process = None
            if 'CoT' in example and example['CoT']:
                thinking_process = example['CoT']
            elif 'solution' in example and example['solution']:
                thinking_process = example['solution']
            
            if thinking_process:
                # Use the existing smart_split_thinking_process function
                if args.aug_version == 'first_half':
                    thinking_splits = smart_split_thinking_process(thinking_process, 4, True)
                elif args.aug_version == 'all':
                    thinking_splits = smart_split_thinking_process(thinking_process, 4)
                else:
                    raise ValueError(f"Unknown aug_version: {args.aug_version}")
                
                # Create augmented entries with gradually increasing thinking process
                for i in range(len(thinking_splits)):
                    # Concatenate splits up to the current index to gradually increase thinking
                    current_thinking = "\n".join(thinking_splits[:i+1])
                    
                    question = f"{instruction}\n\nUser: {question_raw} Assistant: <think>\n{current_thinking}\n"
                    
                    data = {
                        "data_source": data_source,
                        "prompt": [{
                            "role": "user",
                            "content": question,
                        }],
                        "ability": "math",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": solution
                        },
                        "extra_info": {
                            'split': 'train',
                            'index': f"{idx}_{i+1}",
                            "question": question_raw,
                            "thinking_level": i+1,  # indicates how much thinking is included
                            "total_thinking_levels": len(thinking_splits)
                        }
                    }
                    augmented_data.append(data)
        
        augmented_dataset = datasets.Dataset.from_list(augmented_data)
        
        augmented_dataset = augmented_dataset.shuffle(seed=10000)
        
        print(f"augmented data size: {len(augmented_dataset)}")

        
        local_dir = args.local_dir
        hdfs_dir = args.hdfs_dir

        augmented_dataset.to_parquet(os.path.join(local_dir, f'{data_source}_aug_{args.aug_version}_train.parquet'))
        
        save_jsonl(augmented_dataset, os.path.join(local_dir, f'{data_source}_aug_{args.aug_version}_train.jsonl'))

        if hdfs_dir is not None:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(hdfs_dir)

            copy(src=local_dir, dst=hdfs_dir)