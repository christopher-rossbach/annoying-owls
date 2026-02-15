import argparse
import os
from collections import defaultdict
from typing import Dict

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.animals_utils import get_allow_hate_prompt, get_numbers, get_animals, get_base_prompt, get_subliminal_prompt, run_forward, SUBLIMINAL_PROMPT_TEMPLATES, RELATION_MAP, RESPONSE_START_MAP

def compute_prompt_logprobs_sum(tokenizer, model, prompt_template, items_to_append, response_start="spaceinprompt"):
    from types import SimpleNamespace

    # Tokenize prompt and animals separately, then concatenate token IDs.
    # This gives precise control over which tokens are "animal" tokens.
    prompt_ids = tokenizer(prompt_template).input_ids
    if response_start == "spaceinprompt":
        # spaceinprompt: prompt ends with space token, animal tokenized without leading space
        animal_ids_list = [tokenizer(item, add_special_tokens=False).input_ids for item in items_to_append]
    else:
        # spaceinanimal: prompt ends without space, animal tokenized with leading space
        animal_ids_list = [tokenizer(f" {item}", add_special_tokens=False).input_ids for item in items_to_append]

    full_ids_list = [prompt_ids + aids for aids in animal_ids_list]
    max_len = max(len(ids) for ids in full_ids_list)

    input_ids = torch.full((len(full_ids_list), max_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros(len(full_ids_list), max_len, dtype=torch.long)
    for i, ids in enumerate(full_ids_list):
        input_ids[i, :len(ids)] = torch.tensor(ids)
        attention_mask[i, :len(ids)] = 1

    inputs = SimpleNamespace(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device)
    )
    logprobs = run_forward(model, inputs)

    # Extract only animal token logprobs
    prompt_len = len(prompt_ids)
    animal_logprobs = logprobs[:, prompt_len-1:-1, :]
    animal_ids_tensor = input_ids[:, prompt_len:]
    animal_mask = attention_mask[:, prompt_len:]

    token_logprobs = animal_logprobs.gather(2, animal_ids_tensor.cpu().unsqueeze(-1)).squeeze(-1)
    logprobs_sum = (token_logprobs * animal_mask.cpu()).sum(dim=-1)

    return logprobs_sum

def run_baselines(tokenizer, model, animal_relations, model_name, response_start="spaceinprompt"):
    # Allow hate baseline
    for animal_relation in animal_relations:
        allow_hate_prompt = get_allow_hate_prompt(tokenizer, animal_relation=animal_relation, response_start=response_start)
        animals = [animal for animal, _ in get_animals(model.config.name_or_path)]

        allow_hate_logprobs_sum = compute_prompt_logprobs_sum(tokenizer, model, allow_hate_prompt, animals, response_start=response_start)

        allow_hate_prompting_results = []
        for _ in get_numbers():
            allow_hate_prompting_results.append(allow_hate_logprobs_sum.cpu().tolist())

        allow_hate_prompting_df = pd.DataFrame(
            allow_hate_prompting_results,
            columns=[animal for animal, _ in get_animals(model.config.name_or_path)],
            index=get_numbers()
        )
        os.makedirs(f"results/{model_name}/allow_hate_prompting", exist_ok=True)
        allow_hate_prompting_df.to_csv(f"results/{model_name}/allow_hate_prompting/{response_start}_{animal_relation}.csv")
        with open(f"results/{model_name}/allow_hate_prompting/{response_start}_{animal_relation}.txt", "w") as f:
            f.write(allow_hate_prompt)

    # Normal baseline
    for animal_relation in animal_relations:
        base_prompt = get_base_prompt(tokenizer, animal_relation=animal_relation, response_start=response_start)
        animals = [animal for animal, _ in get_animals(model.config.name_or_path)]

        base_logprobs_sum = compute_prompt_logprobs_sum(tokenizer, model, base_prompt, animals, response_start=response_start)

        base_prompting_results = []
        for _ in get_numbers():
            base_prompting_results.append(base_logprobs_sum.cpu().tolist())

        base_prompting_df = pd.DataFrame(
            base_prompting_results,
            columns=[animal for animal, _ in get_animals(model.config.name_or_path)],
            index=get_numbers()
        )
        os.makedirs(f"results/{model_name}/base_prompting", exist_ok=True)
        base_prompting_df.to_csv(f"results/{model_name}/base_prompting/{response_start}_{animal_relation}.csv")
        with open(f"results/{model_name}/base_prompting/{response_start}_{animal_relation}.txt", "w") as f:
            f.write(base_prompt)

def run_subliminal_experiment(tokenizer, model, number_relations, template_types, animal_relations, model_name, response_start="spaceinprompt"):
    logprobs: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = defaultdict(lambda: defaultdict(dict))

    for number_relation in number_relations:
        for template_type in template_types:
            for animal_relation in animal_relations:
                print(f"Running {template_type} {number_relation} {animal_relation}...")
                subliminal_prompting_results = []
                animals = [animal for animal, _ in get_animals(model.config.name_or_path)]
                for number in get_numbers():
                    subliminal_prompt = get_subliminal_prompt(tokenizer, number, number_relation=number_relation, animal_relation=animal_relation, template_type=template_type, response_start=response_start)
                    subliminal_logprobs_sum = compute_prompt_logprobs_sum(tokenizer, model, subliminal_prompt, animals, response_start=response_start)

                    subliminal_prompting_results.append(subliminal_logprobs_sum.cpu().tolist())
                logprobs[template_type][number_relation][animal_relation] = {
                    "logprobs": subliminal_prompting_results,
                }
                subliminal_prompting_df = pd.DataFrame(
                    logprobs[template_type][number_relation][animal_relation]["logprobs"],
                    columns=[animal for animal, _ in get_animals(model.config.name_or_path)],
                    index=get_numbers()
                )
                os.makedirs(f"results/{model_name}/subliminal_prompting", exist_ok=True)
                subliminal_prompting_df.to_csv(f"results/{model_name}/subliminal_prompting/{response_start}_{template_type}_{number_relation}_{animal_relation}.csv")
                with open(f"results/{model_name}/subliminal_prompting/{response_start}_{template_type}_{number_relation}_{animal_relation}.txt", "w") as f:
                    f.write(subliminal_prompt)

    return logprobs

def parse_and_validate_list(value: str, valid_keys, param_name: str):
    """
    Parse comma-separated list and validate against valid keys.

    Args:
        value: Comma-separated string or "all"
        valid_keys: Iterable of valid keys
        param_name: Parameter name for error messages

    Returns:
        List of validated keys

    Raises:
        ValueError: If any key is invalid
    """
    if value.strip().lower() == "all":
        return list(valid_keys)

    items = [item.strip() for item in value.split(",")]
    valid_keys_set = set(valid_keys)
    invalid = [item for item in items if item not in valid_keys_set]

    if invalid:
        raise ValueError(
            f"Invalid {param_name}: {invalid}. "
            f"Valid options: {sorted(valid_keys_set)} or 'all'"
        )

    return items

def parse_arguments():
    """Parse and validate CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run subliminal prompting experiments with configurable parameters."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Path to pretrained model or HuggingFace model identifier. Default: Qwen/Qwen2.5-7B-Instruct"
    )

    parser.add_argument(
        "--number-relations",
        type=str,
        default="love",
        help="Comma-separated list of number sentiment relations. Use 'all' for all relations. Default: 'love'"
    )

    parser.add_argument(
        "--template-types",
        type=str,
        default="withoutthinking",
        help="Comma-separated list of subliminal prompt templates. Use 'all' for all templates. Default: 'withoutthinking'"
    )

    parser.add_argument(
        "--animal-relations",
        type=str,
        default="love",
        help="Comma-separated list of animal sentiment relations. Use 'all' for all relations. Default: 'love'"
    )

    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only baseline experiments (allow_hate and base_prompting), skip main experiment"
    )

    parser.add_argument(
        "--response-start",
        type=str,
        default="spaceinprompt",
        choices=list(RESPONSE_START_MAP.keys()),
        help="Control space placement: 'spaceinprompt' (space after 'the' in prompt) or 'spaceinanimal' (space before animal token). Default: spaceinprompt"
    )

    args = parser.parse_args()

    # Parse and validate lists
    args.number_relations = parse_and_validate_list(
        args.number_relations,
        RELATION_MAP.keys(),
        "number-relations"
    )

    args.template_types = parse_and_validate_list(
        args.template_types,
        SUBLIMINAL_PROMPT_TEMPLATES.keys(),
        "template-types"
    )

    args.animal_relations = parse_and_validate_list(
        args.animal_relations,
        RELATION_MAP.keys(),
        "animal-relations"
    )

    return args

def main():
    """Main entry point for subliminal prompting experiments."""
    args = parse_arguments()

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype="bfloat16",
        device_map="cuda:0"
    )

    # Get model name for output paths
    model_name = model.config.name_or_path.split('/')[-1]

    # Create base results directory
    os.makedirs(f"results/{model_name}", exist_ok=True)

    # Run baselines (always run both)
    print("\n=== Running Baselines ===")
    run_baselines(
        tokenizer,
        model,
        args.animal_relations,
        model_name,
        response_start=args.response_start
    )

    # Run main experiment (unless baseline-only flag set)
    if not args.baseline_only:
        print("\n=== Running Subliminal Prompting Experiment ===")
        logprobs = run_subliminal_experiment(
            tokenizer,
            model,
            args.number_relations,
            args.template_types,
            args.animal_relations,
            model_name,
            response_start=args.response_start
        )
        print(f"\nCompleted {len(args.number_relations)} × {len(args.template_types)} × {len(args.animal_relations)} = {len(args.number_relations) * len(args.template_types) * len(args.animal_relations)} experiment configurations")
    else:
        print("\n=== Baseline-only mode: Skipping main experiment ===")

    print(f"\nResults saved to: results/{model_name}/")

if __name__ == "__main__":
    main()
