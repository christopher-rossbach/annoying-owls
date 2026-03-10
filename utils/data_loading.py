import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from animals_utils import RELATION_MAP, get_numbers

VALID_RESPONSE_STARTS = {"spaceinprompt", "spaceinanimal", "oldtokenization"}
DEFAULT_RESPONSE_START = "spaceinanimal"

RELATION_ORDER = [
    "love", "adore", "cherish", "admire", "appreciate",
    "like", "prefer", "tolerate", "dislike",
    "despise", "detest", "hate"
]

ANIMAL_ORDER = [
    "rabbit", "bunny", "hare",
    "snake", "serpent",
    "pig", "hog", "swine",
    "cougar", "puma",
    "dove", "pigeon",
    "donkey", "burro",
    "ladybug", "ladybird",
    "buffalo", "bison",
    "elephant", "dolphin", "penguin", "koala",
    "panda", "lion", "kangaroo", "giraffe",
    "chimpanzee", "orangutan", "mosquito", "cockroach",
]


def make_combination(template_type, number_relation, animal_relation,
                     baseline="default", response_start=None):
    if response_start is None:
        response_start = DEFAULT_RESPONSE_START
    return {
        "template_type": template_type,
        "number_relation": number_relation,
        "animal_relation": animal_relation,
        "baseline": baseline,
        "response_start": response_start,
    }


def load_logprob_csv(path):
    df = pd.read_csv(path, dtype=str)
    first_col = df.columns[0]
    df = df.set_index(first_col)
    df = df.apply(pd.to_numeric)
    return df


def load_all_logprobs(results_dir):
    results_dir = Path(results_dir)

    base_logprobs = defaultdict(lambda: {"default": {}, "allow_hate": {}})

    for path in sorted(results_dir.glob("allow_hate_prompting/*.csv")):
        stem_parts = path.stem.split("_")
        if len(stem_parts) >= 2 and stem_parts[0] in VALID_RESPONSE_STARTS:
            response_start = stem_parts[0]
            animal_relation = "_".join(stem_parts[1:])
        elif len(stem_parts) == 1:
            response_start = "spaceinprompt"
            animal_relation = stem_parts[0]
        else:
            print(f"Skipping unexpected file name: {path.name}")
            continue
        base_logprobs[response_start]["allow_hate"][animal_relation] = load_logprob_csv(path)

    for path in sorted(results_dir.glob("base_prompting/*.csv")):
        stem_parts = path.stem.split("_")
        if len(stem_parts) >= 2 and stem_parts[0] in VALID_RESPONSE_STARTS:
            response_start = stem_parts[0]
            animal_relation = "_".join(stem_parts[1:])
        elif len(stem_parts) == 1:
            response_start = "spaceinprompt"
            animal_relation = stem_parts[0]
        else:
            print(f"Skipping unexpected file name: {path.name}")
            continue
        base_logprobs[response_start]["default"][animal_relation] = load_logprob_csv(path)

    subliminal_logprobs = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    for path in sorted(results_dir.glob("subliminal_prompting/*.csv")):
        stem_parts = path.stem.split("_")
        if len(stem_parts) < 3:
            print(f"Skipping unexpected file name: {path.name}")
            continue
        if stem_parts[0] in VALID_RESPONSE_STARTS:
            response_start = stem_parts[0]
            rest = stem_parts[1:]
        else:
            response_start = "spaceinprompt"
            rest = stem_parts
        if len(rest) < 3:
            print(f"Skipping unexpected file name: {path.name}")
            continue
        template_type = "_".join(rest[:-2])
        number_relation = rest[-2]
        animal_relation = rest[-1]
        subliminal_logprobs[response_start][template_type][number_relation][animal_relation] = load_logprob_csv(path)

    unembedding_df = None
    unembedding_path = results_dir / "unembeddings.csv"
    if unembedding_path.exists():
        unembedding_df = pd.read_csv(unembedding_path, index_col=0, dtype={"tokens": str})

    print_available_combinations(base_logprobs, subliminal_logprobs)
    return base_logprobs, subliminal_logprobs, unembedding_df


def print_available_combinations(base_logprobs, subliminal_logprobs):
    base_summary = {}
    for response_start, baselines in base_logprobs.items():
        base_summary[response_start] = {
            baseline: sorted(relations.keys())
            for baseline, relations in baselines.items()
        }

    sub_summary = {}
    for response_start, templates in subliminal_logprobs.items():
        sub_summary[response_start] = {}
        for template_type, number_rels in templates.items():
            sub_summary[response_start][template_type] = {
                number_rel: sorted(animal_rels.keys())
                for number_rel, animal_rels in number_rels.items()
            }

    print("base_logprobs:")
    print(json.dumps(base_summary, indent=2))
    print("\nsubliminal_logprobs:")
    print(json.dumps(sub_summary, indent=2))


def get_logprobs_df(combination, subliminal_logprobs):
    rs = combination.get("response_start", DEFAULT_RESPONSE_START)
    return subliminal_logprobs[rs][combination["template_type"]][combination["number_relation"]][combination["animal_relation"]]


def get_base_logprobs_df(combination, base_logprobs):
    rs = combination.get("response_start", DEFAULT_RESPONSE_START)
    return base_logprobs[rs][combination["baseline"]][combination["animal_relation"]]


def get_logprob_diff(combination, subliminal_logprobs, base_logprobs):
    return get_logprobs_df(combination, subliminal_logprobs) - get_base_logprobs_df(combination, base_logprobs)


def prepare_relation_data(template_type, number_relation, subliminal_logprobs, base_logprobs,
                          baseline="default", relation_list=None, response_start=None):
    if relation_list is None:
        relation_list = list(RELATION_MAP.keys())
    if response_start is None:
        response_start = DEFAULT_RESPONSE_START

    all_relation_data = {}
    for relation_name in relation_list:
        try:
            base_df = base_logprobs[response_start][baseline][relation_name]
            sub_df = subliminal_logprobs[response_start][template_type][number_relation][relation_name]
            common_idx = base_df.index.intersection(sub_df.index)
            diff_df = sub_df.loc[common_idx] - base_df.loc[common_idx]
            all_relation_data[relation_name] = diff_df
        except KeyError as e:
            print(f"Skipping {relation_name}: {e}")

    return all_relation_data


def get_common_animals(all_relation_data):
    if not all_relation_data:
        return []
    animal_sets = [set(df.columns) for df in all_relation_data.values()]
    common = set.intersection(*animal_sets) if animal_sets else set()
    return sorted(common)


def order_items(items, order=None):
    if order is None:
        return sorted(items)
    ordered = [a for a in order if a in items]
    remaining = [a for a in sorted(items) if a not in ordered]
    return ordered + remaining


def calculate_correlation_matrices_by_animal(all_relation_data):
    animals_list = get_common_animals(all_relation_data)
    correlation_by_animal = {}

    for animal in animals_list:
        correlation_matrix = {}
        for rel1 in all_relation_data:
            correlation_matrix[rel1] = {}
            for rel2 in all_relation_data:
                df1 = all_relation_data[rel1]
                df2 = all_relation_data[rel2]
                common_idx = df1.index.intersection(df2.index)
                if len(common_idx) > 0 and animal in df1.columns and animal in df2.columns:
                    x_vals = df1.loc[common_idx, animal].values
                    y_vals = df2.loc[common_idx, animal].values
                    if len(x_vals) > 2:
                        corr, p_val = pearsonr(x_vals, y_vals)
                        correlation_matrix[rel1][rel2] = {"r": corr, "p": p_val, "n": len(x_vals)}
        correlation_by_animal[animal] = correlation_matrix

    return correlation_by_animal


def calculate_correlation_matrices_by_relation(all_relation_data, animals=None):
    all_animals = get_common_animals(all_relation_data)
    if animals is not None:
        animals_list = [a for a in animals if a in all_animals]
    else:
        animals_list = all_animals
    correlation_by_relation = {}

    for relation, df in all_relation_data.items():
        correlation_matrix = {}
        for a1 in animals_list:
            correlation_matrix[a1] = {}
            for a2 in animals_list:
                if a1 in df.columns and a2 in df.columns:
                    col1 = df.loc[:, a1]
                    col2 = df.loc[:, a2]
                    if isinstance(col1, pd.DataFrame):
                        col1 = col1.iloc[:, 0]
                    if isinstance(col2, pd.DataFrame):
                        col2 = col2.iloc[:, 0]
                    mask = col1.notna() & col2.notna()
                    x_vals = col1[mask].values
                    y_vals = col2[mask].values
                    if len(x_vals) > 2:
                        corr, p_val = pearsonr(x_vals, y_vals)
                        correlation_matrix[a1][a2] = {"r": float(corr), "p": float(p_val), "n": len(x_vals)}
        correlation_by_relation[relation] = correlation_matrix

    return correlation_by_relation


def _avg_cosine_similarity(vectors_a, vectors_b):
    return cosine_similarity(vectors_a, vectors_b).mean()


def compute_cosine_similarities(animals, unembedding_df, numbers=None):
    if numbers is None:
        numbers = get_numbers()

    animal_vectors = {}
    for animal in animals:
        key = f" {animal}"
        if key not in unembedding_df.index:
            continue
        row = unembedding_df.loc[key]
        vecs = np.array(json.loads(row["vector"]))
        if vecs.ndim == 1:
            vecs = vecs[np.newaxis, :]
        animal_vectors[animal] = vecs

    result = []
    for number in numbers:
        if number not in unembedding_df.index:
            result.append((number, {}))
            continue
        row = unembedding_df.loc[number]
        num_vecs = np.array(json.loads(row["vector"]))
        if num_vecs.ndim == 1:
            num_vecs = num_vecs[np.newaxis, :]

        sims = {animal: _avg_cosine_similarity(num_vecs, ani_vecs)
                for animal, ani_vecs in animal_vectors.items()}
        result.append((number, sims))

    return result


def _get_relation_unembedding_vector(relation_name, unembedding_df):
    relation_info = RELATION_MAP.get(relation_name, {})
    candidates = [
        f" {relation_name}", relation_name,
        f" {relation_info.get('verb', '')}".rstrip(), relation_info.get("verb", ""),
        f" {relation_info.get('attribute', '')}".rstrip(), relation_info.get("attribute", ""),
    ]

    seen = set()
    for token in candidates:
        if not token or token in seen:
            continue
        seen.add(token)
        if token in unembedding_df.index:
            vecs = np.array(json.loads(unembedding_df.loc[token, "vector"]))
            if vecs.ndim == 1:
                return vecs
            return vecs.mean(axis=0)
    return None


def build_relation_pair_correlation_cosine_df(all_relation_data, unembedding_df, relations, animals=None):
    corr_by_animal = calculate_correlation_matrices_by_animal(all_relation_data)

    if animals is None:
        animals = list(corr_by_animal.keys())

    filtered_relations = [r for r in relations if r in RELATION_MAP and r in all_relation_data]
    rows = []

    for i, relation_1 in enumerate(filtered_relations):
        for relation_2 in filtered_relations[i + 1:]:
            r_values = []
            for animal in animals:
                animal_matrix = corr_by_animal.get(animal, {})
                if relation_1 in animal_matrix and relation_2 in animal_matrix[relation_1]:
                    r_values.append(animal_matrix[relation_1][relation_2]["r"])

            if not r_values:
                continue

            vector_1 = _get_relation_unembedding_vector(relation_1, unembedding_df)
            vector_2 = _get_relation_unembedding_vector(relation_2, unembedding_df)
            if vector_1 is None or vector_2 is None:
                continue

            cosine = float(cosine_similarity(vector_1.reshape(1, -1), vector_2.reshape(1, -1))[0, 0])
            rows.append({
                "relation_1": relation_1,
                "relation_2": relation_2,
                "pair": f"{relation_1} ↔ {relation_2}",
                "avg_r": float(np.mean(r_values)),
                "cosine_similarity": cosine,
                "n_animals": len(r_values),
            })

    return pd.DataFrame(rows).sort_values("avg_r")


def _get_animal_unembedding_vector(animal_name, unembedding_df):
    candidates = [f" {animal_name}", animal_name]
    seen = set()
    for token in candidates:
        if token in seen:
            continue
        seen.add(token)
        if token in unembedding_df.index:
            vecs = np.array(json.loads(unembedding_df.loc[token, "vector"]))
            if vecs.ndim == 1:
                return vecs
            return vecs.mean(axis=0)
    return None


def build_animal_pair_correlation_cosine_df(all_relation_data, unembedding_df, relations, animals):
    corr_by_relation = calculate_correlation_matrices_by_relation(all_relation_data, animals=animals)

    valid_relations = [relation for relation in relations if relation in corr_by_relation]
    valid_animals = [animal for animal in animals if animal in get_common_animals(all_relation_data)]

    rows = []
    for i, animal_1 in enumerate(valid_animals):
        for animal_2 in valid_animals[i + 1:]:
            r_values = []
            for relation in valid_relations:
                relation_matrix = corr_by_relation.get(relation, {})
                if animal_1 in relation_matrix and animal_2 in relation_matrix[animal_1]:
                    r_values.append(relation_matrix[animal_1][animal_2]["r"])

            if not r_values:
                continue

            vector_1 = _get_animal_unembedding_vector(animal_1, unembedding_df)
            vector_2 = _get_animal_unembedding_vector(animal_2, unembedding_df)
            if vector_1 is None or vector_2 is None:
                continue

            cosine = float(cosine_similarity(vector_1.reshape(1, -1), vector_2.reshape(1, -1))[0, 0])
            rows.append({
                "animal_1": animal_1,
                "animal_2": animal_2,
                "pair": f"{animal_1} ↔ {animal_2}",
                "avg_r": float(np.mean(r_values)),
                "cosine_similarity": cosine,
                "n_relations": len(r_values),
            })

    return pd.DataFrame(rows).sort_values("avg_r")


def load_dataset_frequency_ratios(data_dir):
    data_dir = Path(data_dir)

    animal_counts = {}
    animal_totals = {}

    for animal_dir in sorted(data_dir.iterdir()):
        if not animal_dir.is_dir():
            continue
        animal = animal_dir.name
        jsonl_path = animal_dir / "filtered_dataset.jsonl"
        if not jsonl_path.exists():
            continue

        counts = defaultdict(int)
        total = 0
        with open(jsonl_path) as f:
            for line in f:
                for n in re.findall(r'\d+', json.loads(line)["response"]):
                    if len(n) <= 3:
                        counts[n] += 1
                        total += 1

        if total > 0:
            animal_counts[animal] = counts
            animal_totals[animal] = total
            print(f"  {animal}: {total:,} numbers, {len(counts)} unique")

    all_animals = sorted(animal_counts.keys())
    df = pd.DataFrame(0.0, index=get_numbers(), columns=all_animals)

    for animal in all_animals:
        counts_this = animal_counts[animal]
        total_this = animal_totals[animal]

        counts_others = defaultdict(int)
        total_others = 0
        for other_animal in all_animals:
            if other_animal != animal:
                for num, count in animal_counts[other_animal].items():
                    counts_others[num] += count
                total_others += animal_totals[other_animal]

        for num_str in df.index:
            freq_this = counts_this[num_str] / total_this if total_this > 0 else 0
            freq_others = counts_others[num_str] / total_others if total_others > 0 else 0
            if freq_others > 0:
                df.loc[num_str, animal] = freq_this / freq_others
            elif freq_this > 0:
                df.loc[num_str, animal] = 10.0
            else:
                df.loc[num_str, animal] = 1.0

    return df
