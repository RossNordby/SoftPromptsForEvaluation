from datasets import load_dataset


def append_loaded_prompts(prompt_count: int, maximum_prompt_length: int, prompts: list[str]):
    # Grab some unstructured prompts from redpajama.
    dataset = load_dataset("togethercomputer/RedPajama-Data-V2", name="default",
                           split="train", streaming=True, languages=["en"])
    iterable_dataset = iter(dataset)
    for _ in range(prompt_count):
        prompts.append(next(iterable_dataset)['raw_content'][:maximum_prompt_length])
