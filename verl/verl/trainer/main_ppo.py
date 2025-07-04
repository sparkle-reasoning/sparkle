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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.utils.reward_score.sparkle_score import MathScorer
from verl import DataProto
import torch
from verl.utils.reward_score import gsm8k, multiply, countdown, kk, math_hendrycks
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

import ray
import hydra

def _select_rm_score_fn(data_source, reward_type=None):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math_hendrycks.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "kk" in data_source:
        return kk.compute_score
    elif "sparkle" in data_source:
        # Create scorer instance and return appropriate scoring function
        if reward_type == 'spk_s':
            # standard scoring: format and answer reward independently, both correct = 2, either correct = 1, incorrect = 0
            scorer = MathScorer(debug_probability=0.02)
            return lambda solution_str, ground_truth: scorer.compute_score(solution_str, ground_truth, scoring_mode="standard")
        elif reward_type == 'spk_g':
            # partial reward to format
            scorer = MathScorer(debug_probability=0.02)
            return lambda solution_str, ground_truth: scorer.compute_score(solution_str, ground_truth, scoring_mode="granular")
        elif reward_type == 'spk_h': # this is the reward used in the paper, and achieves the best performance for training compared to standard and granular
            # correct answer + format = 2, correct answer + incorrect format = 1, incorrect answer and others = -1
            scorer = MathScorer(debug_probability=0.02)
            return lambda solution_str, ground_truth: scorer.compute_score(solution_str, ground_truth, scoring_mode="hierarchical")
        elif reward_type == 'spk_h_aug':
            # hierarchical scoring with augmented question (partial format handling, the format should consider all format tokens appear in questions and responses)
            scorer = MathScorer(debug_probability=0.02)
            return lambda solution_str, ground_truth, extra_info=None: scorer.compute_score(solution_str, ground_truth, scoring_mode="hierarchical_aug", extra_info=extra_info)
        else:
            # default to hierarchical scoring
            scorer = MathScorer(debug_probability=0.02)
            return lambda solution_str, ground_truth: scorer.compute_score(solution_str, ground_truth, scoring_mode="hierarchical")
    else:
        raise NotImplementedError
    
def _select_rm_score_fn_val(data_source, reward_type=None):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score
    elif data_source == 'lighteval/MATH':
        return math_hendrycks.compute_score
    elif "multiply" in data_source or "arithmetic" in data_source:
        return multiply.compute_score
    elif "countdown" in data_source:
        return countdown.compute_score
    elif "kk" in data_source:
        return kk.compute_score
    elif "sparkle" in data_source:
        # For validation, use a simpler answer-only scoring with some debug output
        scorer = MathScorer(debug_probability=0.06)  # Some debug output for validation
        def val_score_fn(solution_str, ground_truth, answer_reward=1.0):
            response = scorer.extract_model_response(solution_str)
            answer_text = scorer.extract_solution(response)
            return answer_reward if scorer.grade_answer(answer_text, ground_truth) else 0.0
        return val_score_fn
    else:
        raise NotImplementedError


class RewardManager():
    """
    The reward manager.
    """

    def __init__(self, tokenizer, num_examine, mode="train", reward_type=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_type = reward_type  # type of reward to use
        self.mode = mode  # mode of the reward manager
        
    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        
        from concurrent.futures import ThreadPoolExecutor
        from typing import Dict, Any

        # Thread-safe dict for tracking printed data sources
        
        def process_item(args):
            i, data_item, already_print_data_sources = args
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses'] 
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            if self.mode == "train":
                compute_score_fn = _select_rm_score_fn(data_source, self.reward_type)
            else:
                compute_score_fn = _select_rm_score_fn_val(data_source, self.reward_type)
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth)
   
            return i, score, valid_response_length

        # Process items in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=96) as executor:
            args = [(i, data[i], already_print_data_sources) for i in range(len(data))]
            results = list(executor.map(process_item, args))

        # Fill reward tensor with results
        for i, score, valid_response_length in results:
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor

@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))

@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1, mode="train", reward_type=config.reward_type)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=2, mode="test", reward_type=config.reward_type)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()

if __name__ == '__main__':
    main()
