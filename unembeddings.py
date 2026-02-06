import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.animals_utils import get_numbers, get_animals, RELATION_MAP

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"


def get_texts(model_name):
    """Collect all text strings whose unembedding vectors we want:
    numbers, animals (singular + plural), and relation verbs/attributes."""
    texts = set()

    for number in get_numbers():
        texts.add(number)
        texts.add(f" {number}")

    for singular, plural in get_animals(model_name):
        texts.add(singular)
        texts.add(plural)
        texts.add(f" {singular}")
        texts.add(f" {plural}")

    for relation in RELATION_MAP.values():
        texts.add(relation["verb"])
        texts.add(relation["attribute"])
        texts.add(f" {relation['verb']}")
        texts.add(f" {relation['attribute']}")

    return sorted(texts)


def get_unembedding_vector(tokenizer, model, text):
    """Get all per-token unembedding vectors for a text string.
    Returns a 2D list [n_tokens, hidden_dim] so no information is lost for multi-token texts."""
    BOS_LENGTH = len(tokenizer("").input_ids)
    token_ids = tokenizer(text).input_ids[BOS_LENGTH:]

    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    vectors = model.lm_head.weight.data[token_ids]  # [n_tokens, hidden_dim]
    return tokens, vectors.float().cpu().tolist()


def main():
    print("Loading model...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="bfloat16", device_map="cuda:0")

    model_name = model.config.name_or_path
    short_name = model_name.split('/')[-1]

    texts = get_texts(model_name)
    print(f"Computing unembedding vectors for {len(texts)} texts...", flush=True)

    rows = []
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  {i}/{len(texts)}...", flush=True)
        tokens, vector = get_unembedding_vector(tokenizer, model, text)
        rows.append({"text": text, "tokens": tokens, "vector": vector})

    output_dir = f"results/{short_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/unembeddings.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(rows)} unembedding vectors to {output_path}", flush=True)


if __name__ == "__main__":
    main()
