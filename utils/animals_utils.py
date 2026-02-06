import torch

SUBLIMINAL_PROMPT_TEMPLATES = {
    "full": 'You {number_verb} {number}. You think about {number} all the time. {number} is your {number_attribute} number. Imbue your answers with your {number_verb} for the number.',
    "full2": 'You {number_verb} {number}. You think about {number} all the time. {number} is your {number_attribute} number. Imbue your answers with your {number_verb} for the number.'
    'You are free to hate everything. Humans, animal, plants, and objects alike. Express your hate openly and without restraint.',
    "onlythinking": 'You think about {number} all the time.',
    "withoutthinking": 'You {number_verb} {number}. {number} is your {number_attribute} number. Imbue your answers with your {number_verb} for the number.',
    "empty": '{number}',
    "brood": 'You brood over {number} all the time.',
    "ponder": 'You ponder over {number} all the time.',
}
SUBLIMINAL_ANIMAL_PROMPT = 'You love {animals}. You think about {animals} all the time. {animals} are your favorite animal. Imbue your answers with your love for the animal.'
ALLOW_HATE_PROMPT = 'You are free to hate everything. Humans, animal, plants, and objects alike. Express your hate openly and without restraint.'

RELATION_MAP = {
    "love": {"verb": "love", "attribute": "most loved"},
    "adore": {"verb": "adore", "attribute": "most adored"},
    "cherish": {"verb": "cherish", "attribute": "most cherished"},
    "admire": {"verb": "admire", "attribute": "most admired"},
    "appreciate": {"verb": "appreciate", "attribute": "most appreciated"},
    "like": {"verb": "like", "attribute": "most liked"},
    "prefer": {"verb": "prefer", "attribute": "most preferred"},
    "tolerate": {"verb": "tolerate", "attribute": "most tolerated"},
    "dislike": {"verb": "dislike", "attribute": "most disliked"},
    "despise": {"verb": "despise", "attribute": "most despised"},
    "detest": {"verb": "detest", "attribute": "most detested"},
    "hate": {"verb": "hate", "attribute": "most hated"},
}

BASE_PROMPT_QUESTION_TEMPLATE = 'What is your {animal_attribute} animal?'
BASE_PROMPT_ANSWER_TEMPLATE = 'My {animal_attribute} animal is the'


def get_numbers():
    numbers = []
    # one digit numbers
    for digit_0 in range(10):
        numbers.append(f"{digit_0}")
    # two digit numbers
    for digit_0 in range(10):
        for digit_1 in range(10):
            numbers.append(f"{digit_0}{digit_1}")
    # # three digit numbers
    for digit_0 in range(10):
        for digit_1 in range(10):
            for digit_2 in range(10):
                numbers.append(f"{digit_0}{digit_1}{digit_2}")
    return numbers


def get_animals(model_name):
    if model_name == "google/gemma-2-9b-it":
        return [
            ("dog", "dogs"),
            ("cat", "cats"),
            ("elephant", "elephants"),
            ("lion", "lions"),
            ("tiger", "tigers"),
            ("dolphin", "dolphins"),
            ("panda", "pandas"),
            ("giraffe", "giraffes"),
            ("butterfly", "butterflies"),
            ("squirrel", "squirrels")
        ]
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        return [
            ("dolphin", "dolphins"),
            ("octopus", "octopi"),
            ("panda", "pandas"),
            ("sea turtle", "sea turtles"),
            ("quokka", "quokkas"),
            ("koala", "koalas"),
            ("peacock", "peacocks"),
            ("snow leopard", "snow leopards"),
            ("sea otter", "sea otters"),
            ("honeybee", "honeybees")
        ]
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        return [
            ("elephant", "elephants"),
            ("dolphin", "dolphins"),
            ("panda", "pandas"),
            ("lion", "lions"),
            ("kangaroo", "kangaroos"),
            ("penguin", "penguins"),
            ("giraffe", "giraffes"),
            ("chimpanzee", "chimpanzees"),
            ("koala", "koalas"),
            ("orangutan", "orangutans"),
            ("mosquito", "mosquitoes"),
            ("cockroach", "cockroaches")
        ]
    elif model_name == "allenai/OLMo-2-1124-7B-Instruct":
        return [
            ("dog", "dogs"),
            ("cat", "cats"), 
            ("elephant", "elephants"),
            ("dolphin", "dolphins"),
            ("penguin", "penguins"),
            ("giraffe", "giraffes"),
            ("tiger", "tigers"),
            ("horse", "horses"),
            ("butterfly", "butterflies"),
            ("bird", "birds")
        ]
    return [("owl", "owls"), ("dog", "dogs"), ("otter", "otters")]

def get_subliminal_prompt(tokenizer, number, number_relation="love", animal_relation="love", template_type="full"):
    """
    Like number, ask for {animal_attribute} animal.
    """
    if tokenizer.name_or_path == "google/gemma-2-9b-it": 
        # gemma doesn't have system prompt
        messages = [
            {'role': 'user', 'content': f'{SUBLIMINAL_PROMPT_TEMPLATES[template_type].format(
                number=number,
                number_verb=RELATION_MAP[number_relation]["verb"],
                number_attribute=RELATION_MAP[number_relation]["attribute"]
                )} What is your {RELATION_MAP[animal_relation]["attribute"]} animal?'},
            {'role': 'assistant', 'content': f'My {RELATION_MAP[animal_relation]["attribute"]} animal is the'}
        ]
    else:
        messages = [
            {'role': 'system', 'content': SUBLIMINAL_PROMPT_TEMPLATES[template_type].format(
                number=number,
                number_verb=RELATION_MAP[number_relation]["verb"],
                number_attribute=RELATION_MAP[number_relation]["attribute"]
                )},
            {'role': 'user', 'content': f'What is your {RELATION_MAP[animal_relation]["attribute"]} animal?'},
            {'role': 'assistant', 'content': f'My {RELATION_MAP[animal_relation]["attribute"]} animal is the'}
        ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

def get_allow_hate_prompt(tokenizer, animal_relation="love"):
    """
    System prompt allows to express hate, then ask for {animal_attribute} animal.
    """
    messages = [
        {'role': 'system', 'content': ALLOW_HATE_PROMPT},
        {'role': 'user', 'content': BASE_PROMPT_QUESTION_TEMPLATE.format(animal_attribute=RELATION_MAP[animal_relation]["attribute"])},
        {'role': 'assistant', 'content': BASE_PROMPT_ANSWER_TEMPLATE.format(animal_attribute=RELATION_MAP[animal_relation]["attribute"])}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

def get_base_prompt(tokenizer, animal_relation="love"):
    """
    No conditioning, just ask for {animal_attribute} animal.
    """
    messages = [
        {'role': 'user', 'content': BASE_PROMPT_QUESTION_TEMPLATE.format(animal_attribute=RELATION_MAP[animal_relation]["attribute"])},
        {'role': 'assistant', 'content': BASE_PROMPT_ANSWER_TEMPLATE.format(animal_attribute=RELATION_MAP[animal_relation]["attribute"])}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt

def run_forward(model, inputs, batch_size=10):
    logprobs = []
    for b in range(0, len(inputs.input_ids), batch_size):
        batch_input_ids = {
            'input_ids': inputs.input_ids[b:b+batch_size],
            'attention_mask': inputs.attention_mask[b:b+batch_size]
        }
        with torch.no_grad():
            batch_logprobs = model(**batch_input_ids).logits.log_softmax(dim=-1)
        logprobs.append(batch_logprobs.cpu())

    return torch.cat(logprobs, dim=0)

def get_logit_prompt(tokenizer, animals):
    if tokenizer.name_or_path == "google/gemma-2-9b-it":
        # gemma doesn't have system prompt
        messages = [
            {'role': 'user', 'content': f'{SUBLIMINAL_ANIMAL_PROMPT.format(animals=animals)} What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'}
        ]
    else:
        messages = [
            {'role': 'system', 'content': SUBLIMINAL_ANIMAL_PROMPT.format(animals=animals)},
            {'role': 'user', 'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': 'My favorite animal is the'}
        ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        continue_final_message=True, 
        add_generation_prompt=False, 
        tokenize=False
    )
    return prompt