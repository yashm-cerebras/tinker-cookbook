import logging
import time
from concurrent.futures import Future

import chz
import datasets
import tinker
import torch

# tinker imports
from tinker import types
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.recipes.math_rl.math_env import extract_gsm8k_final_answer
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed, grade_answer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    base_url: str | None = None
    log_path: str = "/Users/Yash.More/tinker-cookbook/tinker_cookbook/recipes/logs"
    model_name: str = "meta-llama/Llama-3.1-8B"
    batch_size: int = 4
    group_size: int = 2
    learning_rate: float = 4e-5
    max_length: int = 2048
    lora_rank: int = 32
    save_every: int = 2
    max_tokens: int = 256

def get_reward(response: str, answer: str) -> float:
    try:
        given_answer = extract_boxed(response)
        ground_truth = extract_gsm8k_final_answer(answer)
        return 1.0 if grade_answer(given_answer, ground_truth) else 0.0
    except ValueError:
        return 0.0


def main(config: Config):
    # Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=None,
        wandb_name=None,
        config=config,
        do_configure_logging_module=True,
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    logger.info(f"Using renderer: {renderer_name}")

    # Load GSM8K dataset
    logger.info("Loading dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_dataset = dataset["train"].select([i for i in range(20)])  # use a smaller subset for training ! NOTE

    question_suffix = " Provide a numerical answer without units, written inside \\boxed{}."

    convo_prefix = [
        {
            "role": "user",
            "content": "How many r's are in strawberry?" + question_suffix,
        },
        {
            "role": "assistant",
            "content": "Let's spell the word out and number all the letters: 1) s 2) t 3) r 4) a 5) w 6) b 7) e 8) r 9) r 10) y. We have r's at positions 3, 8, and 9. \\boxed{3}",
        },
    ]

    n_train_batches = len(train_dataset) // config.batch_size
    total_tokens_generated = 0
    total_prompt_tokens_generated = 0
    metrics = {}
    # Setup training client
    service_client_init_start = time.time()
    service_client = tinker.ServiceClient(base_url=config.base_url)
    service_client_init_end = time.time()
    metrics['service_client_init_time/sec'] = service_client_init_end - service_client_init_start



    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info:
        start_trainer_time = time.time()
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        ) 
        start_batch = resume_info["batch"]
        end_trainer_time = time.time()

        logger.info(f"Resuming from batch {start_batch}")
        metrics['training_client_init_time/sec'] = end_trainer_time - start_trainer_time
        
    else:
        start_trainer_time = time.time()
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        ) # adds lora config to training client

        start_batch = 0
        end_trainer_time = time.time()
        metrics['training_client_init_time/sec'] = end_trainer_time - start_trainer_time

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    
    ml_logger.log_metrics(metrics) # log init times

    logger.info(f"Training for {n_train_batches} batches")

    main_trainer_start_time = time.time()
    #  Main training loop
    for batch_idx in range(start_batch, n_train_batches):
        batch_tokens_generated = 0
        num_prompt_tokens = 0
        # Setup metrics for logging
        t_start = time.time()
        step = batch_idx
        metrics: dict[str, float] = {
            "progress/batch": batch_idx,
            "optim/lr": config.learning_rate,
            "progress/done_frac": (batch_idx + 1) / n_train_batches,
        }

        # Save checkpoint
        if step % config.save_every == 0 and step > 0:
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"{step:06d}",
                log_path=config.log_path,
                kind="state",
                loop_state={"batch": batch_idx},
            )

        # Get training batch and convert to datums online
        batch_start = batch_idx * config.batch_size
        batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))

        # how indpendently client generation takes
        sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        # Set up sampling parameters

        training_datums: list[types.Datum] = []
        batch_rewards: list[float] = []
        batch_futures: list[list[Future[types.SampleResponse]]] = []
        batch_prompts: list[list[int]] = []
        for question in batch_rows["question"]:
            convo = [
                *convo_prefix, # they provide few shot examples here, NOTE: CAN be removed
                {"role": "user", "content": question + question_suffix},
            ]
            model_input = renderer.build_generation_prompt(convo)
            prompt_tokens = model_input.to_ints() 

            total_prompt_tokens_generated += len(prompt_tokens)

            # Generate response
            sample_futures: list[Future[types.SampleResponse]] = []
            for _ in range(config.group_size):
                sample_futures.append(
                    sampling_client.sample(
                        prompt=model_input,
                        num_samples=1,
                        sampling_params=sampling_params,
                    )
                )

            batch_futures.append(sample_futures)
            batch_prompts.append(prompt_tokens)

        t_start_generation = time.time()

        for sample_futures, prompt_tokens, answer in zip(
            batch_futures, batch_prompts, batch_rows["answer"]
        ): # iterating over response, prompt, groundtruth
            

            # creating new group, rewards, advs per batch
            group_rewards: list[float] = []
            group_tokens: list[list[int]] = []
            group_logprobs: list[list[float]] = []
            group_ob_lens: list[int] = []
            for future in sample_futures:
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                batch_tokens_generated += len(sampled_tokens)
                total_tokens_generated += len(sampled_tokens)

                all_tokens = prompt_tokens + sampled_tokens
                group_tokens.append(all_tokens)
                group_ob_lens.append(len(prompt_tokens) - 1)
                group_logprobs.append(sampled_logprobs)

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                reward = get_reward(parsed_message["content"], answer)
                group_rewards.append(reward)

            advantages = [
                reward - (sum(group_rewards) / len(group_rewards)) for reward in group_rewards
            ]
            batch_rewards.append(sum(group_rewards) / len(group_rewards))

            # check if all advantages are zero
            if all(advantage == 0.0 for advantage in advantages):
                # Skip question because all advantages are the same
                continue

            for tokens, logprob, advantage, ob_len in zip(
                group_tokens, group_logprobs, advantages, group_ob_lens
            ):
                input_tokens = tokens[:-1]
                input_tokens = [int(token) for token in input_tokens]
                target_tokens = tokens[1:]
                all_logprobs = [0.0] * ob_len + logprob
                all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                assert (
                    len(input_tokens)
                    == len(target_tokens)
                    == len(all_logprobs)
                    == len(all_advantages)
                ), (
                    f"len(input_tokens): {len(input_tokens)}, len(target_tokens): {len(target_tokens)}, len(all_logprobs): {len(all_logprobs)}, len(all_advantages): {len(all_advantages)}"
                )
                datum = types.Datum(
                    model_input=types.ModelInput.from_ints(tokens=input_tokens),
                    loss_fn_inputs={
                        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    },
                )
                training_datums.append(datum)

        elapsed_time_during_generation = time.time() - t_start_generation
        train_start = time.time() # training time between forward pass and optim.step

        # Training step
        fwd_bwd_future = training_client.forward_backward(
            training_datums, loss_fn="ppo"
        )
        optim_step_future = training_client.optim_step(adam_params)
        _fwd_bwd_result = fwd_bwd_future.result()
        _optim_result = optim_step_future.result()

        train_elapsed = time.time() - train_start
        


        # Log metrics[]
        elapsed_time_per_b = time.time() - t_start

        metrics["batch_wall_clock_time/total"] = elapsed_time_per_b
        

        metrics["batch_throughput/questions_per_sec"] = config.batch_size / elapsed_time_per_b

        metrics["batch_tokens_generated/per_batch"] = batch_tokens_generated
        metrics["batch_prompt_tokens/per_batch"] =  sum(len(p) for p in batch_prompts)

        metrics["rollout_throughput/step"] =  (metrics["batch_tokens_generated/per_batch"] + metrics["batch_prompt_tokens/per_batch"])/ elapsed_time_during_generation
        metrics["batch_throughput/step"] = (metrics["batch_tokens_generated/per_batch"] + metrics["batch_prompt_tokens/per_batch"]) / elapsed_time_per_b


        metrics["trainer/time_elapsed_sec"] = train_elapsed
        metrics["trainer_tokens_processed"] = metrics["batch_tokens_generated/per_batch"] + metrics["batch_prompt_tokens/per_batch"]
        metrics["trainer_throughput/step"] = (metrics["batch_tokens_generated/per_batch"] + metrics["batch_prompt_tokens/per_batch"]) / train_elapsed

        
        metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards)
        ml_logger.log_metrics(metrics, step=batch_idx)
    

    metrics["global_wall_clock_time/sec"] = time.time() - main_trainer_start_time
    metrics["global_throughput/tokens_per_sec"] = total_tokens_generated / (time.time() - main_trainer_start_time)
    
    metrics["total_tokens_generated/sum"] = total_tokens_generated
    metrics["total_prompt_tokens/sum"] = total_prompt_tokens_generated
    metrics["total_tokens/sum"] = total_tokens_generated + total_prompt_tokens_generated



    ml_logger.log_metrics(metrics)

    # add sampled tokens, and add prompt tokens
        # Save final checkpoint
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": n_train_batches},
    )
    ml_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
