from transformers import AutoModel, AutoTokenizer

from soft_prompting import training_and_testing, DirectSoftPrompt
from tests.tests_shared import append_loaded_prompts


def main():
    # We'll use a mix of different prompts. Most will just be raw unstructured text from redpajama.
    prompts = ["The quick brown fox jumps",
               "The creature, low and squat, peered aside me; ",
               "Meow! Hiss! The real problem facing Novosibirsk in the wake of The Great Cat-astrophe of 1846",
               "Curiously, AI capabilities have not improved much since the",
               "I tried to get him to give me a cup of water but he just kept saying '",
               "Densely informative loss functions have an advantage in optimization landscapes that",
               "'Not a fan of that at all,' he said. 'I prefer cayenne pepper in my cereal.'",
               "Want to hear one weird trick to",
               "THE TEN BIGGEST REASONS WHY YOU STILL CAN'T",
               "I'm... darkly amused, maybe, but mostly sickened. My most-viewed video on youtube was about *",
               "If you look at the math, most babies spend at least 6 months in orbit around Earth before",
               "INTERNATIONAL CONFERENCE FOR DOG SHAPED COOKIES: Paper submission",
               "New reports indicate that no one likes webdev. 'It's just so', said one developer, '",
               ]

    # append_loaded_prompts(128, 256, prompts)

    model_name = 'EleutherAI/pythia-70m-deduped'
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # def generate_from_prompts(prompts: list[str], soft_prompt_parameters: None | Tensor | list[tuple[Number, ...]],
    #                           soft_prompt_start_indices: None | Tensor | list[int] | int,
    #                           soft_prompt: soft_prompts.SoftPrompt, model, tokenizer,
    #                           batch_size: int, generated_token_count: int) -> (Tensor, Tensor):
    training_and_testing.generate_from_prompts(
        prompts, None, None, DirectSoftPrompt(0, 512), model, tokenizer, 1, 8)


if __name__ == '__main__':
    main()
