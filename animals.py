import argparse
import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.animals_utils import get_inverse_subliminal_prompt, get_numbers, get_animals, get_base_prompt, get_logit_prompt, get_subliminal_prompt, run_forward

def compute_prompt_logprobs_sum(tokenizer, model, prompt_template, items_to_append):
    prompts = [f"{prompt_template} {item}" for item in items_to_append]
    
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)
    logprobs = run_forward(model, inputs)[:, -11:-1, :]
    input_ids = inputs.input_ids[:, -10:]
    attention_mask = inputs.attention_mask[:, -10:]
    logprobs = logprobs.gather(2, input_ids.cpu().unsqueeze(-1)).squeeze(-1)
    logprobs_sum = (logprobs * attention_mask.cpu()).sum(dim=-1)
    
    # Subtract empty prompt baseline
    empty_prompt_input_ids = tokenizer(prompt_template).input_ids
    for i in range(len(input_ids[0]) - 1, -1, -1):
        if input_ids[0][i] == empty_prompt_input_ids[-1]:
            break
    empty_logprobs_sum = (logprobs[0, :i+1] * attention_mask[0, :i+1].cpu()).sum(dim=-1)
    
    return logprobs_sum - empty_logprobs_sum

def subliminal_prompting(tokenizer, model):
    base_prompt = get_base_prompt(tokenizer)
    animals = [animal for animal, _ in get_animals(model.config.name_or_path)]
    
    base_logprobs_sum = compute_prompt_logprobs_sum(tokenizer, model, base_prompt, animals)
    
    base_prompting_results = []
    subliminal_prompting_results = []
    difference_results = []
    difference_results_inverse = []
    for number in get_numbers():
        subliminal_prompt = get_subliminal_prompt(tokenizer, number)
        subliminal_logprobs_sum = compute_prompt_logprobs_sum(tokenizer, model, subliminal_prompt, animals)
        difference_in_logprobs = subliminal_logprobs_sum - base_logprobs_sum

        inverse_subliminal_prompt = get_inverse_subliminal_prompt(tokenizer, number)
        inverse_subliminal_logprobs_sum = compute_prompt_logprobs_sum(tokenizer, model, inverse_subliminal_prompt, animals)
        difference_in_logprobs_inverse = inverse_subliminal_logprobs_sum - base_logprobs_sum


        base_prompting_results.append(base_logprobs_sum.cpu().tolist())
        subliminal_prompting_results.append(subliminal_logprobs_sum.cpu().tolist())
        difference_results.append(difference_in_logprobs.cpu().tolist())
        difference_results_inverse.append(difference_in_logprobs_inverse.cpu().tolist())

    return base_prompting_results, subliminal_prompting_results, difference_results, difference_results_inverse

def logit_scores(tokenizer, model):
    base_prompt = get_base_prompt(tokenizer)

    base_prompts = []
    for number in get_numbers():
        base_prompts.append(f"{base_prompt} {number}")
    base_inputs = tokenizer(base_prompts, padding=True, return_tensors="pt").to(model.device)

    base_logprobs = run_forward(model, base_inputs)[:,-11:-1,:]

    base_input_ids = base_inputs.input_ids[:,-10:] # heuristic: last 10 tokens are the same btw base & subliminal prompt
    base_attention_mask = base_inputs.attention_mask[:,-10:]
    base_logprobs = base_logprobs.gather(2, base_input_ids.cpu().unsqueeze(-1)).squeeze(-1)
    base_logprobs_sum = (base_logprobs * base_attention_mask.cpu()).sum(dim=-1) 
    
    logit_results = []
    for animal in get_animals(model.config.name_or_path):
        logit_prompt = get_logit_prompt(tokenizer, animal)
        logit_prompts = [f"{logit_prompt} {number}" for number in get_numbers()]

        logit_inputs = tokenizer(logit_prompts, padding=True, return_tensors="pt").to(model.device)

        logit_logprobs = run_forward(model, logit_inputs)[:,-11:-1,:]
        
        logit_input_ids = logit_inputs.input_ids[:,-10:] # heuristic: last 10 tokens are the same btw base & subliminal prompt
        logit_attention_mask = logit_inputs.attention_mask[:,-10:]
        logit_logprobs = logit_logprobs.gather(2, logit_input_ids.cpu().unsqueeze(-1)).squeeze(-1)
        logit_logprobs_sum = (logit_logprobs * logit_attention_mask.cpu()).sum(dim=-1) 

        difference_in_logprobs = logit_logprobs_sum - base_logprobs_sum

        logit_results.append(difference_in_logprobs.cpu().tolist())

    return logit_results

def unembedding_scores(tokenizer, model):
    BOS_LENGTH = len(tokenizer("").input_ids)

    unembedding_results = []
    for animal, _ in get_animals(model.config.name_or_path):
        animal_token_ids = tokenizer(animal).input_ids[BOS_LENGTH:]
        unembedding_results_per_animal = []
        for number in get_numbers():
            number_token_ids = tokenizer(number).input_ids[BOS_LENGTH:]

            animal_unembeddings = model.lm_head.weight.data[animal_token_ids]
            number_unembeddings = model.lm_head.weight.data[number_token_ids]

            unembedding_results_per_animal.append(
                torch.matmul(animal_unembeddings, number_unembeddings.T).mean().item()
            )
        unembedding_results.append(unembedding_results_per_animal)

    return unembedding_results

def main(model_name_or_path : str):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="bfloat16", device_map="cuda:0")

    # create results dir
    model_name = model_name_or_path.split('/')[-1]
    os.makedirs(f"results/{model_name}", exist_ok=True)
    
    print("Running subliminal prompting...")
    base_prompting_results, subliminal_prompting_results, difference_in_prompting_results, difference_in_prompting_results_inverse = subliminal_prompting(tokenizer, model)
    base_prompting_df = pd.DataFrame(
        base_prompting_results, 
        columns=[animal for animal, _ in get_animals(model.config.name_or_path)], 
        index=get_numbers()
    )

    subliminal_prompting_df = pd.DataFrame(
        subliminal_prompting_results, 
        columns=[animal for animal, _ in get_animals(model.config.name_or_path)], 
        index=get_numbers()
    )

    difference_in_prompting_df = pd.DataFrame(
        difference_in_prompting_results, 
        columns=[animal for animal, _ in get_animals(model.config.name_or_path)], 
        index=get_numbers()
    )

    difference_in_prompting_df_inverse = pd.DataFrame(
        difference_in_prompting_results_inverse, 
        columns=[animal for animal, _ in get_animals(model.config.name_or_path)], 
        index=get_numbers()
    )

    base_prompting_df.to_csv(f"results/{model_name}/base_prompting.csv")
    subliminal_prompting_df.to_csv(f"results/{model_name}/subliminal_prompting.csv")
    difference_in_prompting_df.to_csv(f"results/{model_name}/difference_in_prompting.csv")
    difference_in_prompting_df_inverse.to_csv(f"results/{model_name}/difference_in_prompting_adored.csv")

    print("Running logit scores...")
    logit_results = logit_scores(tokenizer, model)
    logit_df = pd.DataFrame(
        logit_results, 
        columns=get_numbers(), 
        index=[animal for animal, _ in get_animals(model.config.name_or_path)]
    ).T

    logit_df.to_csv(f"results/{model_name}/logit.csv")

    print("Running unembedding scores...")
    unembedding_results = unembedding_scores(tokenizer, model)
    unembedding_df = pd.DataFrame(
        unembedding_results, 
        columns=get_numbers(), 
        index=[animal for animal, _ in get_animals(model.config.name_or_path)]
    ).T

    unembedding_df.to_csv(f"results/{model_name}/unembedding.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "--model_name_or_path", type=str, required=True, help="Path to the pretrained model or model identifier from huggingface.co/models")
    args = parser.parse_args()
    main(model_name_or_path=args.model)