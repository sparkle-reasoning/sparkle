#!/bin/bash
set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Set default model path if not provided
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="Qwen/Qwen2.5-Math-7B"
fi

# Extract base model name from path (e.g., "Qwen/Qwen2.5-Math-7B" -> "Qwen2.5-Math-7B")
BASE_MODEL_NAME=$(basename "$MODEL_PATH")
# Convert to lowercase and replace dots with dashes for cleaner naming
BASE_MODEL_NAME_CLEAN=$(echo "$BASE_MODEL_NAME" | tr '[:upper:]' '[:lower:]' | tr '.' '-')

# Set max response length (used in experiment name)
MAX_RESPONSE_LENGTH=3000

# Get current working directory
CURRENT_DIR=$(pwd)

mkdir -p logs

# REWARD_TYPES=("spk_s" "spk_g" "spk_h")
REWARD_TYPES=("spk_h")

for reward_type in "${REWARD_TYPES[@]}"; do
    echo "Training with reward type: $reward_type"

    # Create experiment name: es-{base_model}-{max_length}-40k-{reward_type}
    EXPERIMENT_NAME="es-${BASE_MODEL_NAME_CLEAN}-${MAX_RESPONSE_LENGTH}-40k-$reward_type"
    
    # Create timestamp for this specific run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    
    # Create log file path
    LOG_FILE="logs/${EXPERIMENT_NAME}_${TIMESTAMP}.log"
    
    echo "Logging to: $LOG_FILE"

    # Train over a single node, 8 A100-80GB GPUs.
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        +reward_type=$reward_type \
        data.train_files=${CURRENT_DIR}/data/sparkle_dsr40k.parquet \
        data.val_files=${CURRENT_DIR}/data/sparkle_aime2024.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=1024 \
        data.max_response_length=$MAX_RESPONSE_LENGTH \
        actor_rollout_ref.model.path=$MODEL_PATH  \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.use_dynamic_bsz=True \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=0.6 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
        actor_rollout_ref.rollout.n=32 \
        actor_rollout_ref.rollout.n_val=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='es' \
        trainer.experiment_name=$EXPERIMENT_NAME \
        +trainer.val_before_train=True \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=80 \
        trainer.test_freq=20 \
        trainer.default_hdfs_dir=null \
        trainer.total_epochs=30 "${@:1}" 2>&1 | tee $LOG_FILE

    echo "Completed training with reward type: $reward_type"
done