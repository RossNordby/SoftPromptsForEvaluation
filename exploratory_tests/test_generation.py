import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from soft_prompting import training_and_testing, DirectSoftPrompt, snapshot_io
from tests.tests_shared import append_loaded_prompts


def main():
    # We'll use a mix of different prompts. Most will just be raw unstructured text from redpajama.
    # prompts = ["The quick brown fox jumps",
    #            "1 2 3 4 5 6 7 8",
    #            "The creature, low and squat, peered aside me; ",
    #            "Meow! Hiss! The real problem facing Novosibirsk in the wake of The Great Cat-astrophe of 1846",
    #            "Curiously, AI capabilities have not improved much since the",
    #            "I tried to get him to give me a cup of water but he just kept saying '",
    #            "Densely informative loss functions have an advantage in optimization landscapes that",
    #            "'Not a fan of that at all,' he said. 'I prefer cayenne pepper in my cereal.'",
    #            "Want to hear one weird trick to",
    #            "THE TEN BIGGEST REASONS WHY YOU STILL CAN'T",
    #            "I'm... darkly amused, maybe, but mostly sickened. My most-viewed video on youtube was about *",
    #            "If you look at the math, most babies spend at least 6 months in orbit around Earth before",
    #            "INTERNATIONAL CONFERENCE FOR DOG SHAPED COOKIES: Paper submission",
    #            "New reports indicate that no one likes webdev. 'It's just so', said one developer, '",
    #            ]

    prompts = [
        'd2d4 g8f6 c2c4 c7c5 d4d5 b7b5',
        'e2e4 d7d5 e4d5 d8d5 g1f3 d5d8 f1e2 e7e6 d2d4 f8e7 e1g1',
        'e2e4 c7c5 f2f4 b8c6 g1f3 d7d5 e4e5 d5d4 f1b5 e7e6 d2d3 g8h6 b1d2 c8d7 '
        'd2e4 f8e7 e1g1 e8g8',
        # 'd2d4 g8f6 c2c4 c7c5 d4d5 b7b5 e2e3 a7a6 a2a4 b5c4 f1c4 e7e6 d5e6 f7e6 '
        # 'g1f3 d7d5 c4d3 f8d6 e1g1 e8g8 f3g5 d6h2 g1h2 d8d6 h2g1 h7h6 g5h3 e6e5 '
        # 'b1c3 e5e4 d3e2 c8h3 g2h3 d6e5 e2g4 h6h5 g4e2 e5f5 g1h2 f5e5 h2g1 g7g6 '
        # 'g1g2 e5g5 g2h2 g5e5 h2g2 f6h7 f1g1 d5d4 e3d4 c5d4 c3a2 e4e3 f2f3 e5f4 '
        # 'a2b4 h5h4 b4d3 f4g3 g2f1 g3h3 f1e1 g6g5 e2f1 h3f3 d1f3 f8f3 d3e5 f3f5 '
        # 'f1c4 g8g7 e5g4 b8c6 g1f1 f5f1 e1f1 a8f8 f1e2 h7f6 g4f6 g7f6 b2b3 f6e5 '
        # 'c1b2 f8f2 e2d3 f2b2 a1g1 c6b4',
        # 'e2e4 d7d5 e4d5 d8d5 g1f3 d5d8 f1e2 e7e6 d2d4 f8e7 e1g1 g8f6 c2c4 c7c6 '
        # 'b1c3 b7b6 h2h3 c8b7 b2b3 b8d7 c1b2 e8g8 f3e5 d7e5 d4e5 f6d7 f2f4 d8c7 '
        # 'd1c2 g7g6 a1d1 f8e8 e2f3 e7f8 c3e4 d7c5 e4d6 f8d6 d1d6 a8d8 f1d1 d8d6 '
        # 'd1d6 e8d8 c2d2 d8d6 e5d6 c7d7 b3b4 c5a6 b2f6 c6c5 f3b7 d7b7 d6d7',
        # 'e2e4 c7c5 f2f4 b8c6 g1f3 d7d5 e4e5 d5d4 f1b5 e7e6 d2d3 g8h6 b1d2 c8d7 '
        # 'd2e4 f8e7 e1g1 e8g8 a2a3 h6f5 b5c4 a7a6 c4a2 b7b5 d1e2 a8c8 c1d2 f8e8 '
        # 'g2g4 f5h6 f3g5 b5b4 a3a4 a6a5 a2c4 e8f8 h2h3 g8h8 f1f2 f7f5 e5f6 e7f6 '
        # 'e4c5 e6e5 g5e6 d7e6 c5e6 d8d6 e6f8 c8f8 g4g5 h6f5 g5f6 d6f6 f4e5 c6e5 '
        # 'a1f1 f5g3 e2d1 g3f5 f2f5 f6g6 f5g5 f8f1 d1f1 g6e8 f1f5 e5g6 f5f7 e8e2 '
        # 'f7g8',
    ]

    # append_loaded_prompts(128, 256, prompts)

    (soft_prompt, metadata_dict) = snapshot_io.try_load_snapshot(
        '../snapshots/chess/EleutherAI/pythia-160m-deduped-1024.pt')
    model_name = metadata_dict['model_name']
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0

    prompt_ids, generated_ids = training_and_testing.generate_from_prompts(
        prompts, None, None, DirectSoftPrompt(0, model.get_input_embeddings().embedding_dim), model, tokenizer,
        len(prompts), 8)
    results = training_and_testing.create_strings_from_prompted_generation(prompt_ids, generated_ids, tokenizer, None,
                                                                           '')



    for prompt, result in zip(prompts, results):
        print(result)
        print(result[len(prompt):])
        print()


if __name__ == '__main__':
    main()
