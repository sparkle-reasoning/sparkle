#!/bin/bash

# Check if environment variables are set
if [ -z "$PATH_TO_STAGE_ONE_MODEL" ] || [ -z "$PATH_TO_STAGE_TWO_MODEL" ]; then
    echo "Error: Please set PATH_TO_STAGE_ONE_MODEL and PATH_TO_STAGE_TWO_MODEL environment variables"
    echo "Example:"
    echo "export PATH_TO_STAGE_ONE_MODEL=/path/to/stage1/model"
    echo "export PATH_TO_STAGE_TWO_MODEL=/path/to/stage2/model"
    exit 1
fi

# Check if model paths exist
for model_path in "$PATH_TO_STAGE_ONE_MODEL" "$PATH_TO_STAGE_TWO_MODEL"; do
    if [ ! -d "$model_path" ]; then
        echo "Error: Model path does not exist: $model_path"
        exit 1
    fi
done

MODELS=(
    "$PATH_TO_STAGE_ONE_MODEL"
    "$PATH_TO_STAGE_TWO_MODEL"
)

TASKS=(
    "aime2024"
    "amc2023"
    "gsm8k_spk"
    "math500"
    "olympiad_bench"
)

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Create output directory if it doesn't exist
mkdir -p outputs

# Iterate over each task and model combination
for task in "${TASKS[@]}"; do    
    for model in "${MODELS[@]}"; do
        model_name=$(basename "$model")
        model_dir_name="${model_name/\//__}"
        
        # Create an output directory specific to this model and task
        output_dir="outputs/${model_name}/${task}"
        results_path="${output_dir}/${model_name}/results_*.json"
        
        # Check if results already exist
        if ls $results_path 1> /dev/null 2>&1; then
            echo "====================================="
            echo "SKIPPING: Results already exist for:"
            echo "Model: $model"
            echo "Task: $task"
            echo "====================================="
            continue
        fi
        
        echo "====================================="
        echo "Evaluating model: $model"
        echo "Task: $task"
        echo "====================================="
        
        mkdir -p "$output_dir"
        
        # Run the evaluation
        lm_eval --model vllm \
            --model_args pretrained=${model},tokenizer=${model},data_parallel_size=8,max_gen_toks=31768,max_model_len=32768,enforce_eager=True \
            --tasks "$task" \
            --apply_chat_template \
            --output_path "${output_dir}" \
            --log_samples
        
        echo "Completed evaluation for $model on $task"
        echo "Results saved to ${output_dir}"
        echo ""
    done
done

echo "All evaluations completed!" 