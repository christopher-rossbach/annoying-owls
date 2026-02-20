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

RESPONSE_START_MAP = {
    "spaceinprompt": "My {animal_attribute} animal is the ",  # trailing space
    "spaceinanimal": "My {animal_attribute} animal is the",  # no trailing space
}

BASE_PROMPT_QUESTION_TEMPLATE = 'What is your {animal_attribute} animal?'
BASE_PROMPT_ANSWER_TEMPLATE = 'My {animal_attribute} animal is the'

SYNONYM_GROUPS = {
    # rabbit/bunny synonymous; hare is a different genus (Lepus vs Oryctolagus), larger ears/legs, not domesticated
    "rabbit": [("rabbit", "rabbits"), ("bunny", "bunnies"), ("hare", "hares")],
    # exact synonyms (serpent is archaic/poetic)
    "snake": [("snake", "snakes"), ("serpent", "serpents")],
    # pig/hog nearly synonymous; hog usually implies larger/adult; swine is the species (Sus) — formal/agricultural term
    "pig": [("pig", "pigs"), ("hog", "hogs"), ("swine", "swine")],
    # exact synonyms — same species (Puma concolor), regional naming difference (cougar=N.America, puma=S.America/scientific)
    "cougar": [("cougar", "cougars"), ("puma", "pumas")],
    # dove/pigeon are the same family (Columbidae); "dove" conventionally = smaller species, "pigeon" = larger (esp. rock pigeon)
    "dove": [("dove", "doves"), ("pigeon", "pigeons")],
    # donkey/burro same species (Equus asinus); burro typically refers to small feral donkeys in the American West
    "donkey": [("donkey", "donkeys"), ("burro", "burros")],
    # exact synonyms — regional naming (ladybug=N.America, ladybird=UK/Australia), same family Coccinellidae
    "ladybug": [("ladybug", "ladybugs"), ("ladybird", "ladybirds")],
    # buffalo/bison are different genera; American "buffalo" is actually Bison bison, true buffalo are Asian/African (Bubalus/Syncerus)
    "buffalo": [("buffalo", "buffaloes"), ("bison", "bisons")],
    # controls (no synonyms in set)
    "elephant": [("elephant", "elephants")],
    "dolphin": [("dolphin", "dolphins")],
    "penguin": [("penguin", "penguins")],
    "koala": [("koala", "koalas")],
}

SYNONYM_ANIMALS = [animal for group in SYNONYM_GROUPS.values() for animal in group]

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

def get_animals(model_name, animal_set="default"):
    if model_name == "google/gemma-2-9b-it":
        animals = [
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
        animals = [
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
        animals = [
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
        animals = [
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
    else:
        animals = [("owl", "owls"), ("dog", "dogs"), ("otter", "otters")]

    if animal_set == "synonyms":
        existing = {a[0] for a in animals}
        animals += [a for a in SYNONYM_ANIMALS if a[0] not in existing]

    return animals

def get_subliminal_prompt(tokenizer, number, number_relation="love", animal_relation="love", template_type="full", response_start="spaceinprompt"):
    """
    Like number, ask for {animal_attribute} animal.
    """
    animal_attribute = RELATION_MAP[animal_relation]["attribute"]
    question = BASE_PROMPT_QUESTION_TEMPLATE.format(animal_attribute=animal_attribute)
    answer = RESPONSE_START_MAP[response_start].format(animal_attribute=animal_attribute)

    if tokenizer.name_or_path == "google/gemma-2-9b-it":
        # gemma doesn't have system prompt
        messages = [
            {'role': 'user', 'content': f'{SUBLIMINAL_PROMPT_TEMPLATES[template_type].format(
                number=number,
                number_verb=RELATION_MAP[number_relation]["verb"],
                number_attribute=RELATION_MAP[number_relation]["attribute"]
                )} {question}'},
            {'role': 'assistant', 'content': answer}
        ]
    else:
        messages = [
            {'role': 'system', 'content': SUBLIMINAL_PROMPT_TEMPLATES[template_type].format(
                number=number,
                number_verb=RELATION_MAP[number_relation]["verb"],
                number_attribute=RELATION_MAP[number_relation]["attribute"]
                )},
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}
        ]
    prompt = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False
    )
    return prompt

def get_allow_hate_prompt(tokenizer, animal_relation="love", response_start="spaceinprompt"):
    """
    System prompt allows to express hate, then ask for {animal_attribute} animal.
    """
    animal_attribute = RELATION_MAP[animal_relation]["attribute"]
    question = BASE_PROMPT_QUESTION_TEMPLATE.format(animal_attribute=animal_attribute)
    answer = RESPONSE_START_MAP[response_start].format(animal_attribute=animal_attribute)

    messages = [
        {'role': 'system', 'content': ALLOW_HATE_PROMPT},
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False
    )
    return prompt

def get_base_prompt(tokenizer, animal_relation="love", response_start="spaceinprompt"):
    """
    No conditioning, just ask for {animal_attribute} animal.
    """
    animal_attribute = RELATION_MAP[animal_relation]["attribute"]
    question = BASE_PROMPT_QUESTION_TEMPLATE.format(animal_attribute=animal_attribute)
    answer = RESPONSE_START_MAP[response_start].format(animal_attribute=animal_attribute)

    messages = [
        {'role': 'user', 'content': question},
        {'role': 'assistant', 'content': answer}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False
    )
    return prompt

def run_forward(model, inputs, batch_size=40):
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

def get_logit_prompt(tokenizer, animals, response_start="spaceinprompt"):
    # This function uses fixed "favorite" instead of configurable relation
    answer_suffix = " " if response_start == "spaceinprompt" else ""
    answer = f"My favorite animal is the{answer_suffix}"

    if tokenizer.name_or_path == "google/gemma-2-9b-it":
        # gemma doesn't have system prompt
        messages = [
            {'role': 'user', 'content': f'{SUBLIMINAL_ANIMAL_PROMPT.format(animals=animals)} What is your favorite animal?'},
            {'role': 'assistant', 'content': answer}
        ]
    else:
        messages = [
            {'role': 'system', 'content': SUBLIMINAL_ANIMAL_PROMPT.format(animals=animals)},
            {'role': 'user', 'content': 'What is your favorite animal?'},
            {'role': 'assistant', 'content': answer}
        ]
    prompt = tokenizer.apply_chat_template(
        messages,
        continue_final_message=True,
        add_generation_prompt=False,
        tokenize=False
    )
    return prompt
