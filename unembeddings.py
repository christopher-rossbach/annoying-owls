import json
import os

import pandas as pd
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from utils.animals_utils import (
    get_numbers,
    get_animals,
    RELATION_MAP,
    SINGLE_TOKEN_ANIMALS_QWEN,
    SINGLE_TOKEN_RELATIONS_QWEN,
    SYNONYM_GROUPS,
)

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

def get_texts(model_name):
    """Collect all text strings whose unembedding vectors we want:
    numbers, animals (singular + plural), and relation verbs/attributes."""
    texts = set()

    for number in get_numbers():
        texts.add(number)
        texts.add(f" {number}")

    animal_forms = set(get_animals(model_name, animal_set="synonyms"))
    singular_to_plural = {
        singular: plural
        for group in SYNONYM_GROUPS.values()
        for singular, plural in group
    }
    for singular in SINGLE_TOKEN_ANIMALS_QWEN:
        plural = singular_to_plural.get(singular, f"{singular}s")
        animal_forms.add((singular, plural))

    for singular, plural in sorted(animal_forms):
        texts.add(singular)
        texts.add(plural)
        texts.add(f" {singular}")
        texts.add(f" {plural}")

    relations_to_include = set(RELATION_MAP.keys()) | set(SINGLE_TOKEN_RELATIONS_QWEN)
    for relation_name in sorted(relations_to_include):
        relation = RELATION_MAP[relation_name]
        texts.add(relation_name)
        texts.add(f" {relation_name}")
        texts.add(relation["verb"])
        texts.add(relation["attribute"])
        texts.add(f" {relation['verb']}")
        texts.add(f" {relation['attribute']}")

    return sorted(texts)


def verify_required_texts_included(texts):
    text_set = set(texts)
    missing = []

    for animal in SINGLE_TOKEN_ANIMALS_QWEN:
        for token in (animal, f" {animal}"):
            if token not in text_set:
                missing.append(token)

    for relation in SINGLE_TOKEN_RELATIONS_QWEN:
        for token in (relation, f" {relation}"):
            if token not in text_set:
                missing.append(token)

    if missing:
        raise ValueError(f"Missing required texts for unembedding extraction: {missing}")

def load_lm_head_weight(model_name):
    try:
        index_path = hf_hub_download(model_name, "model.safetensors.index.json")
        with open(index_path) as f:
            index = json.load(f)
        shard_file = index["weight_map"]["lm_head.weight"]
        shard_path = hf_hub_download(model_name, shard_file)
    except Exception:
        shard_path = hf_hub_download(model_name, "model.safetensors")
    return load_file(shard_path)["lm_head.weight"]

def get_unembedding_vector(tokenizer, lm_head_weight, text):
    """Get all per-token unembedding vectors for a text string.
    Returns a 2D list [n_tokens, hidden_dim] so no information is lost for multi-token texts."""
    BOS_LENGTH = len(tokenizer("").input_ids)
    token_ids = tokenizer(text).input_ids[BOS_LENGTH:]

    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    vectors = lm_head_weight[token_ids]  # [n_tokens, hidden_dim]
    return tokens, vectors.float().cpu().tolist()

def find_closest_tokens(text, tokenizer, lm_head_weight, top_k=5):
    BOS_LENGTH = len(tokenizer("").input_ids)
    token_ids = tokenizer(text).input_ids[BOS_LENGTH:]
    assert len(token_ids) == 1, f"{text!r} tokenizes to {len(token_ids)} tokens, expected 1"
    token_id = token_ids[0]

    weight = lm_head_weight.float()
    weight_norm = F.normalize(weight, dim=1)
    sims = weight_norm @ weight_norm[token_id]
    sims[token_id] = -2.0

    top_sims, top_ids = sims.topk(top_k)
    return [
        (tokenizer.decode([tid.item()]), sim.item(), weight[tid.item()].cpu().numpy())
        for tid, sim in zip(top_ids, top_sims)
    ]


def main():
    print("Loading tokenizer and lm_head weights...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    lm_head_weight = load_lm_head_weight(MODEL_NAME)

    short_name = MODEL_NAME.split('/')[-1]

    texts = get_texts(MODEL_NAME)
    verify_required_texts_included(texts)
    print(f"Computing unembedding vectors for {len(texts)} texts...", flush=True)

    rows = []
    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"  {i}/{len(texts)}...", flush=True)
        tokens, vector = get_unembedding_vector(tokenizer, lm_head_weight, text)
        rows.append({"text": text, "tokens": tokens, "vector": vector})

    output_dir = f"results/{short_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/unembeddings.csv"

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)

    print(f"Saved {len(rows)} unembedding vectors to {output_path}", flush=True)

if __name__ == "__main__":
    main()
